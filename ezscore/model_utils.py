# model_utils.py
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
#     Copyright (c) 2025 The Johns Hopkins University Applied Physics Laboratory LLC
#     Author: William G. Coon, PhD
#     author email: will.coon@jhuapl.edu
#     repo: https://github.com/coonwg1/ezscore/tree/main
#     from "Coon WG, Zerr P, Milsap G, Sikder N, Smith M, Dresler M, Reid M. 
#           ezscore-f: A Set of Freely Available, Validated Sleep Stage Classifiers for Forehead EEG. 
#           bioRxiv, 2025, doi: 10.1101/2025.06.02.657451"
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import os
os.environ["TF_USE_LEGACY_KERAS"] = "True"

import tensorflow as tf
import mne
import numpy as np
from pathlib import Path


def load_zmax( edf_path ):
    """Load ZMax-style left/right EDF pair and return MNE Raw object."""
    fs=64  #for now, model expects data sampled at 64Hz
    rawL = mne.io.read_raw_edf(edf_path, preload=True).resample(fs).filter(l_freq=0.5, h_freq=None)
    rawR = mne.io.read_raw_edf(Path(str(edf_path)[:-5] + "R.edf"), preload=True).resample(fs).filter(l_freq=0.5, h_freq=None)
    dataL = rawL.get_data().flatten()
    dataR = rawR.get_data().flatten()
    info = mne.create_info(['eegl', 'eegr'], sfreq=fs, ch_types=['eeg', 'eeg'])
    raw = mne.io.RawArray(np.vstack([dataL, dataR]), info)

    return raw


def preproc( raw, normalize=True ):
    raw = raw.resample(sfreq=64).filter(l_freq=0.5, h_freq=None) #redundant, but ensures data is at 64Hz, then high-pass filters at 0.5Hz
    if normalize:
        sdata = raw.get_data()
        for ch in range(sdata.shape[0]):
            sig = sdata[ch]
            mad = np.median(np.abs(sig - np.median(sig)))  
            norm = (sig - np.median(sig)) / mad
            iqr = np.subtract(*np.percentile(norm, [75, 25]))
            sdata[ch] = np.clip(norm, -20 * iqr, 20 * iqr)
        raw._data = sdata
        scale = 1 
    else:
        scale = 1_000_000
    eegL = sdata[0,:] * scale
    eegR = sdata[1,:] * scale
    data_as_array = np.vstack((eegL.reshape(1, -1), eegR.reshape(1, -1)))

    return data_as_array, raw 


def ezpredict(model, data):
    """Run ezscore model inference and remap labels to final hypnogram integers.
    Accepts either preprocessed array data or an MNE Raw object.
    """
    if isinstance(data, mne.io.BaseRaw):
        data_ar, _ = preproc(data)  # apply preprocessing if MNE Raw object passed
    else:
        data_ar = data  # assume data is already a preprocessed array
    input_array = reshape_for_decoder(data_as_array=data_ar, fs=64)
    print("Input array shape for model:", input_array.shape)

    num_full_seqs = input_array.shape[0] - 1
    last_seq = input_array[-1]
    last_seq_valid_epochs = np.sum(~np.isnan(last_seq[:, 0, 0]))  # Count non-NaN epochs

    # Case: full last sequence, no padding
    if last_seq_valid_epochs == 100:
        ypred = model.predict(input_array, verbose=1)
        ypred = ypred.reshape(-1, 6)  # Flatten to (epochs, categories)

    else: #partial last sequence, handle separately to avoid NaNs smearing over predictions from the RNN's backward pass
        # Predict full-length sequences first
        ypred_main = model.predict(input_array[:num_full_seqs], verbose=1)
        ypred_main = ypred_main.reshape(-1, 6)  # shape (num_full_seqs * 100, 6)

        # Predict the valid portion of the final (partial) sequence
        valid_last = last_seq[:last_seq_valid_epochs]  # (valid_epochs, 1920, 2)
        valid_last = np.expand_dims(valid_last, axis=0)  # shape (1, valid_epochs, 1920, 2)
        ypred_tail = model.predict(valid_last, verbose=0).reshape(-1, 6)  # (valid_epochs, 6)

        # Concatenate full + partial predictions
        ypred = np.concatenate([ypred_main, ypred_tail], axis=0)  # shape (total_epochs, 6)

    # Clip output to match number of full 30s epochs in original signal
    num_epochs = np.arange(0, data.shape[1], 64 * 30).shape[0]
    ypred = ypred[:num_epochs, :]

    # Reorder columns to match final label schema: [W, N1, N2, N3, REM, ART]
    ypred = tf.reshape(ypred, shape=(-1, 6))
    ypred = tf.gather(ypred, [2, 1, 0, 3, 4, 5], axis=1)  # Reorder to match final label schema
    probs = ypred
    hypnogram = tf.argmax(ypred.numpy(), axis=1).numpy() + 1

    return probs, hypnogram

        # |-------------------- FYI: HERE'S AN EXPLAINER ON WHAT WE'RE DOING WITH THE UNMODIFIED MODEL OUTPUT TO GET IT INTO INTUITIVE ORDERING... ---------------------|

        #   ====================================================================================
        #   ORIGINAL MODEL OUTPUT (Softmax Columns, Before Reordering)
        #   ====================================================================================

        #     Softmax Column Index : Sleep Stage
        #     ----------------------|----------------
        #         0                : N3 <-- not a mistake
        #         1                : N2
        #         2                : N1 <-- not a mistake
        #         3                : REM
        #         4                : Wake
        #         5                : Artifact (ART)    

        #       # re-arrange to intuitive ordering
        #       new_order       = [2, 1, 0, 3, 4, 5]
        #       ypred           = tf.gather(ypred, new_order, axis=1)
        #       hyp             = np.array(tf.argmax( ypred, axis=1 )) + 1 #predicted sleep stage (1-based)

        #     Now we have:
        #       argmax = 0 → N1   → label 1
        #       argmax = 1 → N2   → label 2
        #       argmax = 2 → N3   → label 3
        #       argmax = 3 → REM  → label 4
        #       argmax = 4 → Wake → label 5
        #       argmax = 5 → ART  → label 6

        #   ====================================================================================
        #   FINAL LABEL MAPPING (used in hypnogram 'hyp')
        #   ====================================================================================
        #     Final Integer Label  : Sleep Stage
        #     ----------------------|----------------
        #         1                : N1
        #         2                : N2
        #         3                : N3
        #         4                : REM
        #         5                : Wake
        #         6                : Artifact (ART)

        #   ====================================================================================


def reshape_for_decoder( data_as_array, 
                         fs: int=64,
                         seq_lenth: int=100, 
                         ):     
    import numpy as np
    import tensorflow as tf
    '''
    This function takes a 2D array as input, of either (CHANNELS x TIME) or (TIME x CHANNELS), with 
    time in units of samples at sampling rate 'fs'.  It returns a reshaped array of shape: 
    
        (NUM-SEQUENCES x NUM-EPOCHS-PER-SEQUENCE x NUM-SAMPLES-PER-EPOCH x NUM-CHANNELS)

    This is the format required for input to the CNN-RNN or CNN-transformer-RNN models based on
    Coon & Pubjabi (2021).
    '''

    data = data_as_array 

    # Ensure 'data' is 2-dimensional
    if data.ndim != 2:
        raise ValueError("Input data must be a 2-dimensional array.")
    
    # If the first dimension is larger than the second, transpose the array
    if data.shape[0] > data.shape[1]:
        data = data.T

    # Initialize epoched data array
    num_channels        = data.shape[0]
    num_epochs          = int(np.floor(data.shape[1]/30/fs))
    epoch_length        = 30 * fs
    epoched_data        = np.full((num_channels, num_epochs, epoch_length), np.nan)

    # Epoch the EEG data into CH x EPOCH x TIME
    tidxs = np.arange(0, data.shape[1] - epoch_length + 1, epoch_length)
        
    for ch_idx in np.arange(num_channels):
        e_idx = 0
        for tidx in tidxs:
            epoched_data[ch_idx, e_idx, :] = data[ch_idx, tidx:tidx + epoch_length]
            e_idx += 1

    # Slice into 100-epoch sequences for RNN
    seq_len = seq_lenth

    # Calculate the number of full sequences and one additional for the modulo part if it exists
    num_full_seqs = (epoched_data.shape[1] - 1) // seq_len
    last_seq_start = num_full_seqs * seq_len
    num_full_seqs = epoched_data.shape[1] // seq_len
    remainder_epochs = epoched_data.shape[1] % seq_len
    num_seqs = num_full_seqs + (1 if remainder_epochs > 0 else 0)

    # Adjust the seqdat initialization to accommodate the potentially smaller last sequence
    # Note: The size for the last sequence might be smaller than seq_len
    seqdat = np.full((num_seqs, seq_len, epoched_data.shape[2], epoched_data.shape[0]), np.nan)

    # Fill in full sequences
    for ct in range(num_full_seqs):            
        idx_start = ct * seq_len
        idx_end = idx_start + seq_len
        seqdat[ct, 0:seq_len, :, :] = np.transpose(epoched_data[:, idx_start:idx_end, :], (1, 2, 0))

    # Handle the last sequence if it exists and is smaller than seq_len
    if remainder_epochs > 0:
        idx_start = num_full_seqs * seq_len
        idx_end = epoched_data.shape[1]
        seqdat[num_full_seqs, 0:remainder_epochs, :, :] = np.transpose(epoched_data[:, idx_start:idx_end, :], (1, 2, 0))

    return seqdat
 


def ezspectgm(raw, sfreq=64, win_sec=30, overlap_sec=15, fmin=0, fmax=30, trim_perc=5):
    """Compute left and right channel spectrograms from MNE Raw object."""
    from lspopt import spectrogram_lspopt
    dataL = raw.get_data(picks=[0]).T.flatten()
    dataR = raw.get_data(picks=[1]).T.flatten()
    nperseg = int(win_sec * sfreq)

    f, tt, SxxL = spectrogram_lspopt(dataL, sfreq, nperseg=nperseg, noverlap=overlap_sec)
    SxxL = 10 * np.log10(SxxL)
    f, tt, SxxR = spectrogram_lspopt(dataR, sfreq, nperseg=nperseg, noverlap=overlap_sec)
    SxxR = 10 * np.log10(SxxR)

    good_freqs = np.logical_and(f >= fmin, f <= fmax)
    f_trimmed = f[good_freqs]
    SxxL = SxxL[good_freqs, :]
    SxxR = SxxR[good_freqs, :]
    tt_hours = tt / 3600

    from types import SimpleNamespace
    ezs = SimpleNamespace()
    ezs.f = f_trimmed
    ezs.tt = tt_hours
    ezs.SxxL = SxxL
    ezs.SxxR = SxxR
    ezs.trim_perc = trim_perc

    return ezs #f_trimmed, tt_hours, SxxL, SxxR, trim_perc



def plot_summary(hyp, hypdens, spctgm_object, titl="ezscore-f"):
    """Plot hypnogram, class probabilities, and dual-channel spectrograms."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import Normalize

    f=spctgm_object.f
    tt=spctgm_object.tt 
    SxxL=spctgm_object.SxxL
    SxxR=spctgm_object.SxxR

    allf = [['a', 'ar'], 
            ['b', 'br'], 
            ['c', 'cr'], 
            ['d', 'nr']]
    sns.set(style="darkgrid")
    sns.set(font_scale=1)
    sns.set_theme()
    axs = plt.figure(figsize=(19, 7)).subplot_mosaic(
        allf,
        empty_sentinel=None,
        gridspec_kw={"height_ratios": [1, 1, 1, 1], "hspace": 0.3, "width_ratios": [15*(19/16), 1], "wspace": -0.05},
    )

    # Time axis (hours)
    t = np.arange(0, hyp.shape[0]) / 2 / 60

    # Plot hypnogram
    hyp_plot = hyp.copy()
    hyp_plot = np.where(hyp_plot == 3, -2, hyp_plot)  # just for plotting purposes, so that N3 is the first from the bottom, then  N2 is 2nd, then N1 is third,...
    hyp_plot = np.where(hyp_plot == 1, 3, hyp_plot) 
    hyp_plot = np.where(hyp_plot == -2, 1, hyp_plot)
    axs['b'].plot(t, hyp_plot )
    axs['b'].set_xlabel('Time (hrs)')
    axs['b'].set_ylabel('Hypnogram')
    axs['b'].set_title(f'{titl} Stages', fontweight='bold')
    axs['b'].scatter(t[hyp_plot == 4], hyp_plot[hyp_plot == 4], color=[1/255, 121/255, 51/255], s=35, zorder=2)  # REM is forest green
    axs['b'].scatter(t[hyp_plot == 1], hyp_plot[hyp_plot == 1], color=[118/255, 214/255, 255/255], s=35, zorder=2)  # N3 is light blue
    axs['b'].scatter(t[hyp_plot == 2], hyp_plot[hyp_plot == 2], color=[255/255, 127/255, 0/255], s=35, zorder=2)  # N2 is Dutch Orange
    axs['b'].scatter(t[hyp_plot == 6], hyp_plot[hyp_plot == 6], color='k', s=17.5, zorder=2)  # MT/ART is Black
    axs['b'].set_yticks([1, 2, 3, 4, 5, 6])
    axs['b'].set_yticklabels(['N3', 'N2', 'N1', 'REM', 'WAKE', 'ART'])
    axs['b'].set_xlim(t.min(), t.max())
    axs['b'].set_ylim(axs['b'].get_ylim()[0] - 0.125, axs['b'].get_ylim()[1] + 0.125)

    # Plot softmax probabilities
    probs_df = pd.DataFrame(hypdens, columns=['N1', 'N2', 'N3', 'R', 'W', 'ART'])
    palette = [[134/255, 46/255, 119/255, .8], [255/255, 127/255, 0/255, .8],
               [118/255, 214/255, 255/255, .8], [1/255, 121/255, 51/255, .8],
               [234/255, 234/255, 242/255, 0], [10/255, 67/255, 122/255, .65]]
    probs_df.plot(kind="area", color=palette, stacked=True, lw=0, ax=axs['a'])
    axs['a'].set_xlim(0, len(hypdens))
    axs['a'].set_ylim(0, 1)
    axs['a'].set_ylabel("Probability")
    axs['a'].set_yticks([0, .25, .5, .75, 1])
    axs['a'].set_yticklabels([0, '', .5, '', 1])
    tickpos = np.arange(0, len(hypdens)+1, 1*2*60)
    axs['a'].set_xticks(ticks=tickpos, labels=[f"{int(h)}h" for h in np.arange(0, int(len(hypdens)/2/60)+1)])
    axs['a'].set_xlabel('')
    axs['a'].grid(True, which='both', axis='both', color='white')
    axs['a'].set_facecolor([234/255, 234/255, 242/255])
    axs['a'].legend(loc="right")

    # Spectrograms
    for tag, Sxx, label in zip(['c', 'd'], [SxxL, SxxR], ['EEG-L', 'EEG-R']):
        vmin, vmax = np.percentile(Sxx, [5, 95])
        norm = Normalize(vmin=vmin, vmax=vmax)
        axs[tag].pcolormesh(tt, f, Sxx, norm=norm, cmap='Spectral_r', shading="auto")
        axs[tag].set_ylabel(f"{label}\nFrequency [Hz]")
        axs[tag].set_xlabel("Time (hrs)")
        axs[tag].set_xlim(tt.min(), tt.max())

    # Hide unused plot spaces
    for extra in ['ar', 'br', 'cr', 'nr']:
        axs[extra].set_visible(False)

    return axs


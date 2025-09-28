# ezscore_demo.py
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
#     Copyright: 3 July (c) 2025 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL)
#     License:   MIT License
#     Author: William G. Coon, PhD
#     author email: will.coon@jhuapl.edu
#     repo: https://github.com/coonwg1/ezscore/tree/main
#     from "Coon WG, Zerr P, Milsap G, Sikder N, Smith M, Dresler M, Reid M. 
#           ezscore-f: A Set of Freely Available, Validated Sleep Stage Classifiers for Forehead EEG. 
#           bioRxiv, 2025, doi: 10.1101/2025.06.02.657451"
#     doi: https://doi.org/10.1101/2025.06.02.657451
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
# ez6rt=ez6rt.97

import os
import numpy as np
import pandas as pd
from pathlib import Path
import mne
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

# Import ezscore utilities
from src.ezscore.model_utils import (
    load_zmax,     # loads paired ZMax EDF files into MNE Raw
    load_dcm,      # loads single-file DCM EDF files into MNE Raw
    preproc,       # preprocesses Raw object: minimal filtering, normalization, unit scaling
    ezpredict,     # runs model prediction and returns hypnogram + softmax
    ezspectgm,     # computes multitaper spectrograms
    plot_summary,   # generates hypnogram/softmax/spectrogram plot
    download_ez6moe
)

# ====================================================================================================
#                                USER SETTINGS: SELECT MODEL & PATHS
# ====================================================================================================

# Choose which pretrained model to use: 
#   'ez6'    → expects normalized input (offline analysis, fast)
#   'ez6rt'  → expects raw microvolt input (useful for real-time scoring, fast)
#   'ez6moe' → mixture-of-experts model that averages predictions from an ensemble of differently trained 'ez6' models  NOTE: THIS WILL LOAD SLOWLY BUT *CAN* BE MORE ACCURATE
mdl = 'ez6'

# Directory with ZMax EDF files. Must include both *L.edf and *R.edf files.
data_dir = Path('data/zmax')

# Directory to save output figures
figoutdir = Path('figs')
figoutdir.mkdir(parents=True, exist_ok=True)

# Directory to save hypnogram CSV files
figoutdir = Path('hypno_csvs')
figoutdir.mkdir(parents=True, exist_ok=True)

# Automatically find all ZMax-style left-channel EDFs
edf_file_fullpaths = list(data_dir.rglob('*.edf'))

# Determine whether to apply normalization based on selected model
normalize = True if mdl in ('ez6', 'ez6moe') else False

# Ensure ez6moe model is downloaded if selected
if mdl == 'ez6moe' and not os.path.exists("model/ez6moe"):
    download_ez6moe()

# ====================================================================================================
#                                       LOAD MODEL
# ====================================================================================================

# Load trained ezscore model (from model/ez6 or model/ez6rt)
print(f"Loading {mdl} model...")
model = load_model( f"model/{mdl}" )

# ====================================================================================================
#                            PROCESS EACH FILE IN THE EDF DIRECTORY
# ====================================================================================================

# For demonstration, we loop through all detected EDF pairs
for edf_path in edf_file_fullpaths:

    if edf_path.name.lower().endswith('r.edf'):
        continue  #skip right-channel files (we'll load ZMax pairs w.r.t. the L file only)

    print(f"Running ezscore demo on: {edf_path.name}")

    # ------------------------------------------------------------------------
    # STEP 1: LOAD AND PREPROCESS EEG
    # ------------------------------------------------------------------------

    # Load EEG L+R channels as MNE Raw object
    # Route: L.edf => paired ZMax; anything else => single-file DCM
    if edf_path.name.lower().endswith("l.edf"):
        raw = load_zmax(edf_path)     # uses ...L.edf and its matching ...R.edf
    else:
        raw = load_dcm(edf_path)      # single-file EDF (e.g., DCM)

    # Preprocess the data:
    #   - Resample to 64 Hz
    #   - High-pass filter at 0.5 Hz
    #   - Normalize if using ez6
    # Returns both raw and the model-ready 2-channel array
    data_array, raw = preproc(raw, normalize=normalize)

    # ------------------------------------------------------------------------
    # STEP 2: RUN MODEL INFERENCE
    # ------------------------------------------------------------------------

    # Run prediction: returns
    #   - ypred: (N_epochs x 6) softmax probabilities
    #   - hyp: integer-coded hypnogram  (1=N1, 2=N2, 3=N3, 4=REM, 5=W, 6=ART)
    hyp, ypred = ezpredict( model=model, data=data_array )

    # ------------------------------------------------------------------------
    # STEP 3: COMPUTE SPECTROGRAMS
    # ------------------------------------------------------------------------

    # Use multitaper method to compute L/R EEG spectrograms from raw signal
    ezs = ezspectgm( raw )

    # ------------------------------------------------------------------------
    # STEP 4: PLOT RESULTS
    # ------------------------------------------------------------------------

    # Plot softmax, hypnogram, and spectrogram summary
    axs = plot_summary( hyp=hyp, hypdens=ypred, spctgm_object=ezs, titl=mdl )

    # Save hypnogram CSV and display the figure
    csv_save_path = figoutdir / f"hypnos_{mdl}_{edf_path.name}.csv"
    pd.DataFrame(np.asarray(hyp, dtype=int)).to_csv(csv_save_path, index=False, header=False)
    fig_save_path = figoutdir / f"hypnos_{mdl}_{edf_path.name}.png"
    plt.savefig( fig_save_path, format='png', dpi=150, bbox_inches='tight' )
    plt.show()


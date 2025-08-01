import os
import numpy as np
import mne
from huggingface_hub import snapshot_download
from tensorflow.keras.models import load_model
from ezscore.model_utils import ezpredict  # adjust path as needed


def ezscore( data_matrix, sfreq=64, model_name="ez6moe" ):

    """
    Applies the ezscore-f sleep classifier to a 2D EEG array.

    Parameters
    ----------
    data_matrix : np.ndarray
        EEG data of shape (channels, time_samples).
    sfreq : int, optional
        Sampling frequency (Hz). Default is 64 Hz. If another sfreq is used, the function will automatically resample internally.
    model_name : str, optional
        Which model to use (default "ez6moe"). The first time ez6moe is called, the function will download the model from Hugging Face.
        ez6moe is a mixture-of-experts model that averages predictions from an ensemble of differently trained 'ez6' models.
        Other options include the lightweight 'ez6' for normalized input and 'ez6rt' for raw microvolt input (useful for real-time scoring).

    Returns
    -------
    hypnogram : np.ndarray
        Sleep stage predictions (integers per 30s epoch).
    yprobs : np.ndarray
        Softmax class probabilities per epoch.
    """

    # Validate input shape
    if data_matrix.ndim != 2:
        raise ValueError("data_matrix must be a 2D array (channels x time).")
    if data_matrix.shape[0] > data_matrix.shape[1]:
        data_matrix = data_matrix.T  # ensure (channels, time)

    # Create MNE RawArray object
    info = mne.create_info(ch_names=["Ch1", "Ch2"][:data_matrix.shape[0]],
                           sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_matrix, info, verbose=False)
    raw.filter(l_freq=0.5, h_freq=None, verbose=False)
    raw.resample(sfreq, verbose=False)

    # Download model if not present
    model_dir = os.path.join(os.path.expanduser("~"), ".ezscore_models", model_name)
    model_file = os.path.join(model_dir, "model.keras")
    if not os.path.exists(model_file):
        print(f"Downloading '{model_name}' model from Hugging Face...")
        snapshot_download(repo_id=f"coonwg1/{model_name}",
                          local_dir=model_dir,
                          repo_type="model")

    # Load model
    model = load_model(model_file, compile=False)

    # Predict
    hypnogram, yprobs = ezpredict(raw, model=model, mdl=model_name)
    return hypnogram, yprobs

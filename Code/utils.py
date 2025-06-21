import os
import sys
import soundfile as sf
import librosa

def read_wav_file(file_path: str, print_info: bool = True, sr: int = None) -> tuple:
    """
    Reads a WAV file using librosa and returns the sample rate and audio data.

    Args:
        file_path (str): Path to the WAV file.
        print_info (bool): Whether to print file information.
        sr (int or None): Target sample rate. If None, uses the file's original sample rate.

    Returns:
        tuple: Sample rate (int) and audio data (numpy array).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid audio file or can't be read.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        raise ValueError(f"Error loading audio file with librosa: {file_path}") from e

    if print_info:
        print("WAV file read successfully with librosa...")
        print(f"\tSample Rate: {sample_rate} Hz")
        print(f"\tAudio Data Shape: {audio_data.shape}")
        print(f"\tAudio Data Duration: {len(audio_data)/sample_rate:.2f} seconds")

    return audio_data, sample_rate

def save_wav_file(file_path: str, audio_data, sample_rate: int) -> None:
    """
    Saves an audio signal to a WAV file using the given sample rate.

    Args:
        file_path (str): Path where the WAV file will be saved.
        audio_data (np.ndarray): The audio signal data (mono or stereo).
        sample_rate (int): The sample rate in Hz.

    Raises:
        ValueError: If saving fails due to unsupported format or invalid data.
    """
    try:
        sf.write(file_path, audio_data, sample_rate)
        print(f"WAV file saved successfully at: {file_path}")
    except Exception as e:
        raise ValueError(f"Error saving WAV file: {e}")

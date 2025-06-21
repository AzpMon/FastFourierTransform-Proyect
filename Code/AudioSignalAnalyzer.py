import librosa
import plotly.graph_objects as go
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns



class AudioAnalyzer:
    """
    A utility class for plotting and analyzing audio signals and their spectrograms.
    """

    def __init__(self):
        pass



    def plot_waveform(self, audio, sr, color="royalblue", name="Audio"):
        """
        Plot the waveform of a single audio signal.
        """
        time = np.arange(len(audio)) / sr  # tiempo en segundos

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=audio,
            mode='lines',
            name=name,
            line=dict(color=color)
        ))

        fig.update_layout(
            title=f"Waveform of {name}",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            template="plotly_white",
            width=1000,
            height=400
        )

        fig.show()





    def plot_spectrogram(self, audio, sr, name="Audio"):
        """
        Plot the spectrogram of a single audio signal.
        """
        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        S_dB = librosa.amplitude_to_db(S, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)

        fig = go.Figure(data=go.Heatmap(
            z=S_dB,
            x=times,
            y=freqs,
            colorscale='Viridis',
            colorbar=dict(title='dB'),
        ))

        fig.update_layout(
            title=f"Spectrogram (STFT) of {name}",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            template="plotly_white",
            width=1000,
            height=500
        )

        fig.show()


    def plot_waveform_seaborn(self, audio, sr, name="Audio", color="royalblue"):
        """
        Plot the waveform using Seaborn + Matplotlib.
        """
        time = np.arange(len(audio)) / sr  # tiempo en segundos

        plt.figure(figsize=(12, 3))
        sns.lineplot(x=time, y=audio, color=color)
        plt.title(f"Waveform of {name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    def plot_spectrogram_seaborn(self, audio, sr, name="Audio"):
        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        S_dB = librosa.amplitude_to_db(S, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)

        plt.figure(figsize=(12, 4))
        ax = sns.heatmap(
            S_dB,
            cmap="viridis",
            cbar_kws={'label': 'dB'},
            xticklabels=False,
            yticklabels=False
        )
        ax.set_title(f"Spectrogram (STFT) of {name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        ax.set_xticks(np.linspace(0, S_dB.shape[1], 6))
        ax.set_xticklabels([f"{t:.1f}" for t in np.linspace(times[0], times[-1], 6)])

        ax.set_yticks(np.linspace(0, S_dB.shape[0], 6))
        ax.set_yticklabels([f"{f:.0f}" for f in np.linspace(freqs[0], freqs[-1], 6)])

        ax.invert_yaxis()

        plt.tight_layout()
        plt.show()



    def plot_fft_seaborn(self, fft_result, sr, name="FFT Magnitude", color="orange"):
        """
        Plot the magnitude spectrum of the FFT result using Seaborn.

        Args:
            fft_result (list[complex]): Result from fft_1d.
            sr (int): Sampling rate.
        """
        n = len(fft_result)
        freqs = np.fft.fftfreq(n, d=1/sr)
        magnitude = np.abs(fft_result)

        # Solo la mitad positiva
        half = n // 2
        freqs = freqs[:half]
        magnitude = magnitude[:half]

        plt.figure(figsize=(12, 3))
        sns.lineplot(x=freqs, y=magnitude, color=color)
        plt.title(f"{name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_fft_plotly(self, fft_result, sr, name="FFT Magnitude", color="orange"):
        """
        Plot the magnitude spectrum of the FFT result using Plotly.

        Args:
            fft_result (list[complex]): Result from fft_1d.
            sr (int): Sampling rate.
            name (str): Plot title.
            color (str): Line color.
        """
        import plotly.graph_objects as go
        import numpy as np

        n = len(fft_result)
        freqs = np.fft.fftfreq(n, d=1/sr)
        magnitude = np.abs(fft_result)

        # Solo la mitad positiva
        half = n // 2
        freqs = freqs[:half]
        magnitude = magnitude[:half]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs,
            y=magnitude,
            mode='lines',
            name=name,
            line=dict(color=color)
        ))

        fig.update_layout(
            title=f"{name}",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            template="plotly_white",
            width=1000,
            height=400
        )

        fig.show()

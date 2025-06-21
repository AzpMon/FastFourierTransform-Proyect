import librosa
import plotly.graph_objects as go
import numpy as np
import librosa
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns



class AudioSignalAnalyzer:
    """
    A utility class for plotting and analyzing audio signals and their spectrograms.
    """

    def __init__(self):
        pass

    def plot_waveform(self, audio, sr, color="royalblue", name="Audio"):
        """
        Plot the waveform of a single audio signal.
        """
        time = librosa.times_like(audio, sr=sr)

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

    def plot_two_waveforms(self, audio1, sr1, audio2, sr2,
                           color1="royalblue", color2="orange",
                           name1="Audio 1", name2="Audio 2"):
        """
        Plot the waveforms of two audio signals in vertical subplots.
        """
        time1 = librosa.times_like(audio1, sr=sr1)
        time2 = librosa.times_like(audio2, sr=sr2)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(name1, name2))

        fig.add_trace(go.Scatter(
            x=time1, y=audio1, mode='lines',
            name=name1, line=dict(color=color1)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=time2, y=audio2, mode='lines',
            name=name2, line=dict(color=color2)
        ), row=2, col=1)

        fig.update_layout(
            title="Comparison of Two Audio Signals",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=700,
            width=1000,
            template="plotly_white",
            showlegend=False
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

    def plot_two_spectrograms(self, audio1, sr1, audio2, sr2,
                              name1="Audio 1", name2="Audio 2"):
        """
        Plot the spectrograms of two audio signals in vertical subplots.
        """
        S1 = np.abs(librosa.stft(audio1, n_fft=2048, hop_length=512))
        S1_dB = librosa.amplitude_to_db(S1, ref=np.max)
        freqs1 = librosa.fft_frequencies(sr=sr1, n_fft=2048)
        times1 = librosa.frames_to_time(np.arange(S1.shape[1]), sr=sr1, hop_length=512)

        S2 = np.abs(librosa.stft(audio2, n_fft=2048, hop_length=512))
        S2_dB = librosa.amplitude_to_db(S2, ref=np.max)
        freqs2 = librosa.fft_frequencies(sr=sr2, n_fft=2048)
        times2 = librosa.frames_to_time(np.arange(S2.shape[1]), sr=sr2, hop_length=512)

        fig = make_subplots(rows=2, cols=1, subplot_titles=(f"Spectrogram of {name1}", f"Spectrogram of {name2}"))

        fig.add_trace(go.Heatmap(
            z=S1_dB,
            x=times1,
            y=freqs1,
            colorscale='Viridis',
            showscale=False
        ), row=1, col=1)

        fig.add_trace(go.Heatmap(
            z=S2_dB,
            x=times2,
            y=freqs2,
            colorscale='Viridis',
            colorbar=dict(title="dB")
        ), row=2, col=1)

        fig.update_layout(
            title="Comparison of Spectrograms",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            template="plotly_white",
            height=800,
            width=1000
        )

        fig.show()


    def plot_waveform_seaborn(self, audio, sr, name="Audio", color="royalblue"):
        """
        Plot the waveform using Seaborn + Matplotlib.
        """
        time = librosa.times_like(audio, sr=sr)

        plt.figure(figsize=(12, 3))
        sns.lineplot(x=time, y=audio, color=color)
        plt.title(f"Waveform of {name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_two_waveforms_seaborn(self, audio1, sr1, audio2, sr2,
                                    name1="Audio 1", name2="Audio 2",
                                    color1="royalblue", color2="orange"):
        """
        Plot two waveforms in vertical subplots using Seaborn + Matplotlib.
        """
        time1 = librosa.times_like(audio1, sr=sr1)
        time2 = librosa.times_like(audio2, sr=sr2)

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        sns.lineplot(x=time1, y=audio1, ax=axes[0], color=color1)
        axes[0].set_title(name1)
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True)

        sns.lineplot(x=time2, y=audio2, ax=axes[1], color=color2)
        axes[1].set_title(name2)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_spectrogram_seaborn(self, audio, sr, name="Audio"):
        """
        Plot the spectrogram using Seaborn + Matplotlib.
        """
        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        S_dB = librosa.amplitude_to_db(S, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)

        plt.figure(figsize=(12, 4))
        sns.heatmap(S_dB, xticklabels=100, yticklabels=100, cmap="viridis", cbar_kws={'label': 'dB'})
        plt.title(f"Spectrogram (STFT) of {name}")
        plt.xlabel("Time (frames)")
        plt.ylabel("Frequency bins")
        plt.tight_layout()
        plt.show()

    def plot_two_spectrograms_seaborn(self, audio1, sr1, audio2, sr2,
                                      name1="Audio 1", name2="Audio 2"):
        """
        Plot two spectrograms in vertical subplots using Seaborn + Matplotlib.
        """
        S1 = np.abs(librosa.stft(audio1, n_fft=2048, hop_length=512))
        S1_dB = librosa.amplitude_to_db(S1, ref=np.max)

        S2 = np.abs(librosa.stft(audio2, n_fft=2048, hop_length=512))
        S2_dB = librosa.amplitude_to_db(S2, ref=np.max)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))

        sns.heatmap(S1_dB, ax=axes[0], cmap="viridis", xticklabels=100, yticklabels=100, cbar=False)
        axes[0].set_title(f"Spectrogram of {name1}")
        axes[0].set_ylabel("Freq bins")

        sns.heatmap(S2_dB, ax=axes[1], cmap="viridis", xticklabels=100, yticklabels=100, cbar_kws={'label': 'dB'})
        axes[1].set_title(f"Spectrogram of {name2}")
        axes[1].set_xlabel("Time (frames)")
        axes[1].set_ylabel("Freq bins")

        plt.tight_layout()
        plt.show()
        
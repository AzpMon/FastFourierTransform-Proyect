
# Audio FFT Analysis

This repository provides tools for analyzing audio signals using the **Fast Fourier Transform (FFT)** and its inverse. It includes utilities to visualize waveforms, spectrograms, and apply low-pass filtering to audio signals. The implementation is built using Python, with support for both Plotly and Seaborn visualizations.

---

## 📁 Project Structure

```
├── Experiments and analysis/
│   └── Jupyter notebooks demonstrating audio transformations,
│       waveform and spectrogram plots (before and after FFT),
│       and results of filtering using low-pass filters.
│
├── Data and results/
│   └── Contains audio samples (`.wav`) and corresponding images showing
│       waveform and spectrograms before and after FFT/filtering.
│
├── fft.py
│   └── Contains the `FastFourierTransforms` class with:
│       - Recursive FFT and iFFT
│       - Configurable low-pass filter
│
├── audio_analysis.py
│   └── Contains the `AudioSignalAnalyzer` class with:
│       - Waveform and spectrogram plotting (Plotly & Seaborn)
│
├── requirements.txt
│   └── Lists required Python packages and versions
```

---

## ⚙️ Features

### `FastFourierTransforms` class
- ✅ Recursive 1D Fast Fourier Transform (FFT)
- ✅ Inverse FFT (iFFT)
- ✅ Low-pass filtering with adjustable frequency retention

### `AudioSignalAnalyzer` class
- ✅ Plot audio waveforms using Plotly and Seaborn
- ✅ Plot spectrograms with customizable layout
- ✅ Compare original and filtered signals visually

---

## 🧪 Example Use Cases

- Visualize how an audio signal changes after applying FFT.
- Remove high-frequency noise using a low-pass filter.
- Compare original and reconstructed audio signals using waveform and spectrogram plots.

---

## 📦 Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Main libraries used:
- `numpy`
- `librosa`
- `plotly`
- `matplotlib`
- `seaborn`

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/audio-fft-analysis.git
   cd audio-fft-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebooks in the `Experiments and analysis/` folder to visualize FFT transformations and filtering results.

---

## 🔍 Visualization Examples

- 📈 Waveform plots before and after applying FFT or filters
- 🎛️ Spectrogram heatmaps to analyze frequency content
- 🎧 Playback and comparison of original and processed `.wav` files

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
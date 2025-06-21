import cmath

class FastFourierTransforms():
    """
    A class to compute the 1D Fast Fourier Transform (FFT) and its inverse (iFFT)
    using the recursive Cooley-Tukey algorithm.
    """

    def __init__(self):
        """
        Initializes the FastFourierTransforms class.
        """
        self.audio = None

    def fft_1d(self, a: list[complex]) -> list[complex]:
        """
        Computes the 1D Fast Fourier Transform (FFT) of a sequence of complex numbers
        using the recursive Cooley-Tukey algorithm.

        Args:
            a (list[complex]): A list of complex numbers representing the input signal.
                               The length of the list must be a power of two.

        Returns:
            list[complex]: The FFT of the input signal as a list of complex numbers.
        """
        n = len(a)
        if n == 1:
            return a

        omega_n = cmath.exp(-2j * cmath.pi / n)
        omega = 1

        a_even = a[0::2]
        a_odd = a[1::2]

        y_even = self.fft_1d(a_even)
        y_odd = self.fft_1d(a_odd)

        y = [0] * n
        for k in range(n // 2):
            t = omega * y_odd[k]
            y[k] = y_even[k] + t
            y[k + n // 2] = y_even[k] - t
            omega *= omega_n

        return y

    def ifft_1d(self, a: list[complex]) -> list[complex]:
        """_summary_

        Args:
            a (list[complex]): _description_

        Returns:
            list[complex]: _description_
        """
        n = len(a)
        a_conj = [x.conjugate() for x in a]
        y = self.fft_1d(a_conj)
        return [x.conjugate() / n for x in y]
    


    def lowPassFilter(self, a: list[complex], preserve_ratio: float = 0.5) -> list[complex]:
        """
        Applies a low-pass filter to the FFT of a signal by zeroing out
        high-frequency components based on a percentage of frequency spectrum to preserve.

        Args:
            a (list[complex]): The FFT of the signal.
            preserve_ratio (float): Fraction of frequency components to preserve (0 < ratio <= 1).
                                    For example, 0.25 preserves the lowest 25% of frequencies.

        Returns:
            list[complex]: The filtered FFT.
        """
    
        if not (0 < preserve_ratio <= 1):
            raise ValueError("preserve_ratio must be between 0 and 1.")

        n = len(a)
        cutoff = int((n * preserve_ratio) // 2)

        # Copy FFT to avoid modifying the original
        filtered = a.copy()

        # Zero out frequencies beyond the preserve band
        for i in range(cutoff, n - cutoff):
            filtered[i] = 0

        return filtered
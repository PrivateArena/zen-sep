package audio

import (
	"math"
)

func ApplyHannWindow(data []float64) {
	n := len(data)
	// periodic=True Hann window matches torch.hann_window
	for i := 0; i < n; i++ {
		data[i] *= 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
}

func GetSTFT(wavData []float32, n_fft int, hopSize int) [][]complex128 {
	// Center=True padding
	pad := n_fft / 2
	paddedData := make([]float32, len(wavData)+2*pad)
	copy(paddedData[pad:], wavData)

	numFrames := (len(paddedData) - n_fft) / hopSize + 1
	stftResult := make([][]complex128, numFrames)
	
	window := make([]float64, n_fft)
	for i := range window { window[i] = 1.0 }
	ApplyHannWindow(window)

	for i := 0; i < numFrames; i++ {
		start := i * hopSize
		frame := make([]complex128, n_fft)
		for j := 0; j < n_fft; j++ {
			frame[j] = complex(float64(paddedData[start+j])*window[j], 0)
		}
		stftResult[i] = FFT(frame)
	}
	return stftResult
}

func GetISTFT(spectrogram [][]complex128, n_fft int, hopSize int, length int) []float32 {
	numFrames := len(spectrogram)
	outputLength := (numFrames-1)*hopSize + n_fft
	
	output := make([]float64, outputLength)
	overlap := make([]float64, outputLength)
	
	window := make([]float64, n_fft)
	for i := range window { window[i] = 1.0 }
	ApplyHannWindow(window)

	for i := 0; i < numFrames; i++ {
		start := i * hopSize
		// ISTFT: IFFT(frame) * window
		frame := IFFT(spectrogram[i])
		for j := 0; j < n_fft; j++ {
			if start+j < outputLength {
				output[start+j] += real(frame[j]) * window[j]
				overlap[start+j] += window[j] * window[j]
			}
		}
	}
	
	// Normalize by overlap-add energy
	res := make([]float32, length)
	pad := n_fft / 2
	for i := 0; i < length; i++ {
		t := i + pad
		if t < outputLength && overlap[t] > 1e-10 {
			res[i] = float32(output[t] / overlap[t])
		}
	}
	return res
}

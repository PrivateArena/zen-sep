package audio

import (
	"math"
	"math/cmplx"
	"testing"
)

// ── helpers ───────────────────────────────────────────────────────────────────

func makeComplexSlice(n int) []complex128 {
	s := make([]complex128, n)
	for i := range s {
		s[i] = complex(math.Sin(float64(i)*0.01), math.Cos(float64(i)*0.01))
	}
	return s
}

func makeFloat32Slice(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(math.Sin(float64(i) * 0.001))
	}
	return s
}

// ── FFT benchmarks ────────────────────────────────────────────────────────────

// BenchmarkFFT_5120 measures bluesteinFFT (MDX-Net hot path).
// Each call allocates 4 slices + recomputes all chirp twiddles.
func BenchmarkFFT_5120(b *testing.B) {
	x := makeComplexSlice(5120)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FFT(x)
	}
}

// BenchmarkFFT_4096 measures radix2FFT (Spleeter / Demucs hot path).
func BenchmarkFFT_4096(b *testing.B) {
	x := makeComplexSlice(4096)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FFT(x)
	}
}

// BenchmarkIFFT_5120 measures IFFT cost (GetISTFT calls one per frame).
func BenchmarkIFFT_5120(b *testing.B) {
	x := makeComplexSlice(5120)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		IFFT(x)
	}
}

// BenchmarkBluesteinAlloc isolates allocation cost inside bluesteinFFT.
func BenchmarkBluesteinAlloc(b *testing.B) {
	n := 5120
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := nextPowerOf2(2*n - 1)
		_ = make([]complex128, n)
		_ = make([]complex128, m)
		_ = make([]complex128, n)
		_ = make([]complex128, m) // pad(a,m)
	}
}

// BenchmarkBluesteinTwiddle isolates the cmplx.Exp twiddle-factor loop.
func BenchmarkBluesteinTwiddle(b *testing.B) {
	n := 5120
	b.ResetTimer()
	for iter := 0; iter < b.N; iter++ {
		for i := 0; i < n; i++ {
			ang := math.Pi * float64(i*i) / float64(n)
			_ = cmplx.Exp(complex(0, -ang))
		}
	}
}

// ── STFT benchmarks ───────────────────────────────────────────────────────────

// BenchmarkGetSTFT_MDX benchmarks a realistic MDX-Net STFT call (~3s at 44.1kHz).
func BenchmarkGetSTFT_MDX(b *testing.B) {
	samples := makeFloat32Slice(44100 * 3) // 3s mono
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = GetSTFT(samples, 5120, 1280)
	}
}

// BenchmarkGetSTFT_Spleeter benchmarks Spleeter's n_fft=4096.
func BenchmarkGetSTFT_Spleeter(b *testing.B) {
	samples := makeFloat32Slice(44100 * 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = GetSTFT(samples, 4096, 1024)
	}
}

// BenchmarkSTFT_FrameAlloc benchmarks the per-frame make([]complex128, n_fft)
// allocation that occurs inside the parallel goroutines of GetSTFT.
func BenchmarkSTFT_FrameAlloc(b *testing.B) {
	n := 5120
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = make([]complex128, n)
	}
}

// ── ISTFT benchmarks ──────────────────────────────────────────────────────────

// BenchmarkGetISTFT_MDX benchmarks reconstruction from a typical MDX STFT output.
func BenchmarkGetISTFT_MDX(b *testing.B) {
	numSamples := 44100 * 3
	n_fft := 5120
	hopSize := 1280
	// Pre-build a spectrogram matching MDX dimensions
	pad := n_fft / 2
	padded := numSamples + 2*pad
	numFrames := (padded-n_fft)/hopSize + 1
	spec := make([][]complex128, numFrames)
	for i := range spec {
		spec[i] = makeComplexSlice(n_fft)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = GetISTFT(spec, n_fft, hopSize, numSamples)
	}
}

// BenchmarkOLALoop benchmarks the serial overlap-add accumulation without IFFT cost.
func BenchmarkOLALoop(b *testing.B) {
	numFrames := 300
	n_fft := 5120
	hopSize := 1280
	outputLength := (numFrames-1)*hopSize + n_fft
	frames := make([][]complex128, numFrames)
	for i := range frames {
		frames[i] = makeComplexSlice(n_fft)
	}
	window := make([]float64, n_fft)
	for i := range window {
		window[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n_fft)))
	}
	b.ResetTimer()
	for iter := 0; iter < b.N; iter++ {
		output := make([]float64, outputLength)
		overlap := make([]float64, outputLength)
		for i := 0; i < numFrames; i++ {
			start := i * hopSize
			for j := 0; j < n_fft; j++ {
				if start+j < outputLength {
					output[start+j] += real(frames[i][j]) * window[j]
					overlap[start+j] += window[j] * window[j]
				}
			}
		}
		_ = output
		_ = overlap
	}
}

// ── Window benchmarks ─────────────────────────────────────────────────────────

// BenchmarkHannWindow measures recomputation cost (called on each STFT/ISTFT call).
func BenchmarkHannWindow(b *testing.B) {
	n := 5120
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := make([]float64, n)
		for j := range w {
			w[j] = 1.0
		}
		ApplyHannWindow(w)
	}
}

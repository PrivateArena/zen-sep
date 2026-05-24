package audio

import (
	"math"
	"runtime"
	"sync"
)

// hannCache stores precomputed Hann windows keyed by length.
// Fix #6: avoids recomputing O(n_fft) math.Cos calls on every STFT/ISTFT call.
var hannCache sync.Map // map[int][]float64

func getHannWindow(n int) []float64 {
	if v, ok := hannCache.Load(n); ok {
		return v.([]float64)
	}
	w := make([]float64, n)
	for i := 0; i < n; i++ {
		w[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	hannCache.Store(n, w)
	return w
}

// ApplyHannWindow multiplies data in-place by a periodic Hann window.
// Kept for compatibility; internal callers use getHannWindow.
func ApplyHannWindow(data []float64) {
	n := len(data)
	w := getHannWindow(n)
	for i := range data {
		data[i] *= w[i]
	}
}

// framePools maps n_fft → *sync.Pool of []complex128 scratch buffers.
// Fix #5: eliminates per-frame heap allocation inside parallel goroutines.
var framePools sync.Map // map[int]*sync.Pool

func getFramePool(n int) *sync.Pool {
	if v, ok := framePools.Load(n); ok {
		return v.(*sync.Pool)
	}
	p := &sync.Pool{
		New: func() any {
			buf := make([]complex128, n)
			return &buf
		},
	}
	framePools.Store(n, p)
	return p
}

func GetSTFT(wavData []float32, n_fft int, hopSize int) [][]complex128 {
	// Center=True padding
	pad := n_fft / 2
	paddedData := make([]float32, len(wavData)+2*pad)
	copy(paddedData[pad:], wavData)

	numFrames := (len(paddedData)-n_fft)/hopSize + 1
	stftResult := make([][]complex128, numFrames)

	window := getHannWindow(n_fft) // cached — no alloc after first call
	pool := getFramePool(n_fft)

	numWorkers := runtime.NumCPU()
	var wg sync.WaitGroup
	chunk := (numFrames + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end && i < numFrames; i++ {
				startIdx := i * hopSize

				// Borrow scratch buffer from pool (Fix #5)
				framep := pool.Get().(*[]complex128)
				frame := *framep
				for j := 0; j < n_fft; j++ {
					frame[j] = complex(float64(paddedData[startIdx+j])*window[j], 0)
				}
				out := FFT(frame)
				// FFT returns a new slice; return scratch to pool
				pool.Put(framep)
				stftResult[i] = out
			}
		}(w*chunk, (w+1)*chunk)
	}
	wg.Wait()

	return stftResult
}

func GetISTFT(spectrogram [][]complex128, n_fft int, hopSize int, length int) []float32 {
	numFrames := len(spectrogram)
	outputLength := (numFrames-1)*hopSize + n_fft

	output := make([]float64, outputLength)
	overlap := make([]float64, outputLength)

	window := getHannWindow(n_fft) // cached

	// Pre-compute IFFTs in parallel
	iffts := make([][]complex128, numFrames)
	numWorkers := runtime.NumCPU()
	var wg sync.WaitGroup
	chunkSize := (numFrames + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end && i < numFrames; i++ {
				iffts[i] = IFFT(spectrogram[i])
			}
		}(w*chunkSize, (w+1)*chunkSize)
	}
	wg.Wait()

	// Serial Overlap-Add
	for i := 0; i < numFrames; i++ {
		start := i * hopSize
		frame := iffts[i]
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

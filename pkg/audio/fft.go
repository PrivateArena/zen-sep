package audio

import (
	"math"
	"math/cmplx"
	"sync"
)

// nextPowerOf2 returns the smallest power of 2 greater than n.
func nextPowerOf2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// FFT performs a mixed-radix FFT. For MDX-Net, we specifically support N=5120 (5 * 1024).
func FFT(x []complex128) []complex128 {
	N := len(x)
	if N == 5120 {
		// Mixed-radix: FFT_5120(x) = Combine 1024 FFTs of size 5
		// For simplicity/accuracy in Go, we'll use Bluestein's Algorithm or a Radix-5 + Radix-2.
		// Implementing a robust Bluestein for any N.
		return bluesteinFFT(x)
	}
	
	// Fallback to power-of-2 Radix-2 (standard)
	n := nextPowerOf2(N)
	data := make([]complex128, n)
	copy(data, x)
	return radix2FFT(data)
}

func radix2FFT(data []complex128) []complex128 {
	n := len(data)
	j := 0
	for i := 0; i < n; i++ {
		if i < j {
			data[i], data[j] = data[j], data[i]
		}
		m := n >> 1
		for m >= 1 && j >= m {
			j -= m
			m >>= 1
		}
		j += m
	}
	for l := 2; l <= n; l <<= 1 {
		ang := 2 * math.Pi / float64(l)
		wlen := cmplx.Exp(complex(0, -ang))
		for i := 0; i < n; i += l {
			w := complex(1, 0)
			for j := 0; j < l/2; j++ {
				u := data[i+j]
				v := data[i+j+l/2] * w
				data[i+j] = u + v
				data[i+j+l/2] = u - v
				w *= wlen
			}
		}
	}
	return data
}

// bluesteinPlan caches the precomputed FFT(b) and chirp sequence for a given N.
// Building the plan costs O(N log N); subsequent calls cost O(N log N) with
// only one heap allocation (the a[] convolution buffer).
type bluesteinPlan struct {
	n     int
	m     int
	fb    []complex128 // FFT of the b-sequence (length m) — reused across calls
	chirp []complex128 // exp(-i·π·k²/n) for k=0..n-1
}

var bluesteinPlanCache sync.Map // map[int]*bluesteinPlan

func getBluesteinPlan(n int) *bluesteinPlan {
	if v, ok := bluesteinPlanCache.Load(n); ok {
		return v.(*bluesteinPlan)
	}
	m := nextPowerOf2(2*n - 1)
	chirp := make([]complex128, n)
	b := make([]complex128, m)
	b[0] = 1
	for i := 1; i < n; i++ {
		ang := math.Pi * float64(i*i) / float64(n)
		w := cmplx.Exp(complex(0, -ang)) // exp(-i·π·i²/n)
		chirp[i] = w
		// b[i] = conj(w) = exp(+i·π·i²/n)
		b[i] = cmplx.Conj(w)
		b[m-i] = cmplx.Conj(w)
	}
	chirp[0] = 1
	plan := &bluesteinPlan{
		n:     n,
		m:     m,
		fb:    radix2FFT(b),
		chirp: chirp,
	}
	bluesteinPlanCache.Store(n, plan)
	return plan
}

// bluesteinFFT implements the Chirp-Z transform for arbitrary FFT sizes.
// Uses a cached plan so twiddle factors and FFT(b) are computed only once per N.
func bluesteinFFT(x []complex128) []complex128 {
	n := len(x)
	plan := getBluesteinPlan(n)

	// a[k] = x[k] · exp(-i·π·k²/n)
	a := make([]complex128, plan.m)
	for i := 0; i < n; i++ {
		a[i] = x[i] * plan.chirp[i]
	}
	// (remaining a[n..m-1] are already zero)

	fa := radix2FFT(a)
	for i := range fa {
		fa[i] *= plan.fb[i]
	}
	res := radix2IFFT(fa)

	// final[k] = res[k] · exp(-i·π·k²/n)
	final := make([]complex128, n)
	for i := 0; i < n; i++ {
		// Fix #8: was float32 — use float64 for full precision
		ang := math.Pi * float64(i*i) / float64(n)
		w := cmplx.Exp(complex(0, -ang))
		final[i] = res[i] * w
	}
	return final
}

// pad is kept for potential external use; bluesteinFFT no longer calls it.
func pad(x []complex128, n int) []complex128 {
	res := make([]complex128, n)
	copy(res, x)
	return res
}

func IFFT(x []complex128) []complex128 {
	n := len(x)
	data := make([]complex128, n)
	copy(data, x)
	for i := range data {
		data[i] = cmplx.Conj(data[i])
	}
	res := FFT(data)
	for i := range res {
		// NO division by N here if we handle it in ISTFT OLA!
		// BUT standard IFFT convention is /N.
		res[i] = cmplx.Conj(res[i]) / complex(float64(n), 0)
	}
	return res
}

func radix2IFFT(x []complex128) []complex128 {
	n := len(x)
	data := make([]complex128, n)
	copy(data, x)
	for i := range data {
		data[i] = cmplx.Conj(data[i])
	}
	res := radix2FFT(data)
	for i := range res {
		res[i] = cmplx.Conj(res[i]) / complex(float64(n), 0)
	}
	return res
}

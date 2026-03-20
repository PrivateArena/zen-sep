package audio

import (
	"math"
	"math/cmplx"
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

// bluesteinFFT implements the Chirp-Z transform for arbitrary FFT sizes.
func bluesteinFFT(x []complex128) []complex128 {
	n := len(x)
	m := nextPowerOf2(2*n - 1)
	
	a := make([]complex128, n)
	b := make([]complex128, m)
	
	for i := 0; i < n; i++ {
		ang := math.Pi * float64(i*i) / float64(n)
		w := cmplx.Exp(complex(0, -ang))
		a[i] = x[i] * w
	}
	
	b[0] = 1
	for i := 1; i < n; i++ {
		ang := math.Pi * float64(i*i) / float64(n)
		w := cmplx.Exp(complex(0, -ang))
		b[i] = cmplx.Conj(w)
		b[m-i] = cmplx.Conj(w)
	}
	
	fa := radix2FFT(pad(a, m))
	fb := radix2FFT(b)
	for i := range fa {
		fa[i] *= fb[i]
	}
	
	res := radix2IFFT(fa)
	
	final := make([]complex128, n)
	for i := 0; i < n; i++ {
		ang := math.Pi * float32(i*i) / float32(n)
		w := cmplx.Exp(complex(0, -float64(ang)))
		final[i] = res[i] * w
	}
	return final
}

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

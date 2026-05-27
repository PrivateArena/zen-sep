package audio

import (
	"math"
)

// Resample resamples the given single-channel audio from inRate to outRate.
// It uses a windowed sinc filter for high quality sample rate conversion.
func Resample(in []float32, inRate, outRate int) []float32 {
	if inRate == outRate {
		return in
	}

	// Sinc filter with a support radius of 16 input samples
	const radius = 16
	ratio := float64(outRate) / float64(inRate)
	
	// Cutoff frequency for lowpass filter
	var cutoff float64
	if ratio < 1.0 {
		cutoff = ratio * 0.95 // Downsampling
	} else {
		cutoff = 0.95        // Upsampling
	}

	outLen := int(math.Floor(float64(len(in)) * ratio))
	out := make([]float32, outLen)

	for i := 0; i < outLen; i++ {
		t := float64(i) / ratio
		center := int(math.Floor(t))
		
		var sum float64
		var weightSum float64
		
		for j := center - radius; j <= center + radius; j++ {
			if j < 0 || j >= len(in) {
				continue
			}
			dx := t - float64(j)
			
			// Hann window
			w := 0.5 + 0.5*math.Cos(math.Pi*dx/float64(radius))
			
			// Sinc
			var s float64
			piX := math.Pi * dx * cutoff
			if math.Abs(piX) < 1e-9 {
				s = 1.0
			} else {
				s = math.Sin(piX) / piX
			}
			
			weight := w * s
			sum += float64(in[j]) * weight
			weightSum += weight
		}
		
		if weightSum > 0 {
			out[i] = float32(sum / weightSum)
		} else {
			out[i] = 0
		}
	}
	
	return out
}

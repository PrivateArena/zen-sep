package audio

import (
	"os"
	"github.com/youpy/go-wav"
)

func LoadWav(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	d := wav.NewReader(f)
	format, err := d.Format()
	if err != nil {
		return nil, 0, err
	}

	channels := int(format.NumChannels)
	// Pre-allocate a reasonable buffer to reduce re-allocations
	samples := make([]float32, 0, 1024*1024) 
	
	for {
		s, err := d.ReadSamples()
		if err != nil {
			break
		}
		for _, v := range s {
			for c := 0; c < channels; c++ {
				samples = append(samples, float32(v.Values[c])/32768.0)
			}
		}
	}

	return samples, channels, nil
}

func SaveWav(path string, data []float32, sampleRate int, channels int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	numFrames := len(data) / channels
	w := wav.NewWriter(f, uint32(numFrames), uint16(channels), uint32(sampleRate), 16)
	
	// Pre-allocate the entire sample buffer to write in ONE batch
	// This is the single biggest performance win for I/O
	samples := make([]wav.Sample, numFrames)
	
	for i := 0; i < numFrames; i++ {
		var s wav.Sample
		for c := 0; c < channels; c++ {
			val := int(data[i*channels+c] * 32768.0)
			if val > 32767 {
				val = 32767
			} else if val < -32768 {
				val = -32768
			}
			s.Values[c] = val
		}
		samples[i] = s
	}

	return w.WriteSamples(samples)
}

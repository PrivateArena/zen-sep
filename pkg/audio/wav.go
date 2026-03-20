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
	samples := make([]float32, 0)
	for {
		s, err := d.ReadSamples()
		if err != nil {
			break
		}
		for _, v := range s {
			// Correct Stereo handling: append all available channels from the frame
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

	w := wav.NewWriter(f, uint32(len(data)/channels), uint16(channels), uint32(sampleRate), 16)
	
	// Process frames [L, R, L, R...]
	for i := 0; i < len(data); i += channels {
		var s wav.Sample
		for c := 0; c < channels; c++ {
			val := int(data[i+c] * 32767)
			if val > 32767 { val = 32767 } else if val < -32768 { val = -32768 }
			s.Values[c] = val
		}
		if err := w.WriteSamples([]wav.Sample{s}); err != nil {
			return err
		}
	}

	return nil
}

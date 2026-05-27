# zen-sep: Code, Performance & Feature Review

## Executive Summary

The codebase is a solid, well-structured Go audio source separator. The DSP layer
(`fft.go`, `stft.go`) is particularly well-crafted — Bluestein/Chirp-Z for n=5120,
cached Hann windows, pooled scratch buffers, parallel STFT/ISTFT. The main areas
for improvement are **error handling hygiene**, **the hardcoded lib path**, **MDX
chunking correctness**, **missing post-processing**, and several features that
would make this competitive with Python-based dataset pipelines.

---

## 1. Code Quality

### 1.1 Critical: Hardcoded libonnxruntime path (`main.go:43`)
```go
libPath := "/media/jang/home/Deve/transcribe-models/onnxruntime-linux-x64-1.24.2/..."
```
This is the single biggest usability blocker. It will fail on every machine
that isn't yours. Fix:

```go
libPath := os.Getenv("ONNXRUNTIME_LIB")
if libPath == "" {
    libPath = "/usr/lib/libonnxruntime.so" // reasonable default
}
```
Or expose it as a `--lib` flag. Also expose `--output-dir`.

---

### 1.2 Ignored errors on tensor creation
Throughout `pipeline.go`, tensor creation errors are silently dropped:
```go
inputTensor, _ := ort.NewTensor(inputShape, inputBuf)   // _ drops error
outputTensor, _ := ort.NewEmptyTensor[float32](inputShape)
```
An OOM or shape mismatch will produce a nil tensor and a cryptic panic downstream.
Every `_, _` on a tensor constructor should be a proper `if err != nil { ... }`.

---

### 1.3 `Separate()` swallows errors from sub-pipelines
```go
case ModelTypeMDXNet:
    s.processMDXNet(data, inputFile)  // return value ignored
```
`processMDXNet` / `processDemucs` / `processSpleeter` return nothing, so any
internal failure (bad inference, failed save) is only logged, never propagated.
Change their signatures to `error` and propagate.

---

### 1.4 `saveStereo` vs `saveStereoFull` — inconsistent normalization
`saveStereo` (MDX, Spleeter) peak-normalizes. `saveStereoFull` (Demucs) does
not — raw model floats may clip. Apply the same normalization pass uniformly,
or make it a configurable option.

---

### 1.5 Spleeter's `n_fft` and `hopSize` are duplicated, not from `Config`
`processSpleeter` hard-codes `n_fft := 4096` and `hopSize := 1024` even though
those values exist in `Config`. Same for `demucs_n_fft` / `demucs_hop` in
`processDemucs`. They should come from `s.Config.N_FFT` / `s.Config.HopSize`.

---

### 1.6 `LoadWav` returns hardcoded `channels = 2`
```go
return samples, 2, nil   // always 2 regardless of actual file
```
The actual channel count is available in `format.NumChannels`. This masks bugs
when feeding mono files. Return the real value; let `main.go` enforce the
stereo requirement explicitly.

---

### 1.7 `SaveWav` sample-rate is always 44100
`saveStereo` calls `audio.SaveWav(..., 44100, 2)` but MDX-Net operates at
44100 while the input might not be. Thread `sampleRate` through from
`Config.SampleRate` (and default to the loaded file's sample rate).

---

### 1.8 Spleeter batches the entire track into one giant tensor
```go
inputShape := ort.NewShape(2, int64(numSplits), int64(freqBins), int64(splitSize))
```
For a 10-minute file at 44100 Hz this allocates ~2 GB in a single contiguous
block. ONNX Runtime has a practical limit; this will OOM long files. Spleeter
should iterate chunks the same way MDX does.

---

### 1.9 MDX: last chunk silently dropped when `endFrame > numFrames`
```go
if endFrame > numFrames {
    return  // tail of file discarded
}
```
The final partial chunk is never processed, potentially dropping the last
`chunkSize/2` frames (~1.5 s at default settings). The correct fix is to
zero-pad the last chunk to `chunkSize` and process it normally.

---

### 1.10 `progress` atomic counter is incremented but never read/displayed
`atomic.AddInt32(&progress, ...)` is accumulated in MDX but there is no
progress bar for it, unlike Demucs which uses `progressbar`. Add a
`progressbar.Default` for MDX and Spleeter too.

---

### 1.11 `opts` is stored in `Separator` but never destroyed
```go
Opts *ort.SessionOptions
```
`SessionOptions` should be freed after the session is created with
`opts.Destroy()` (or `defer opts.Destroy()` inside `NewSeparator`).

---

## 2. Performance

### 2.1 MDX concurrency cap of 4 is conservative for a 9700X
```go
sem := make(chan struct{}, 4)
```
You have 8 cores / 16 threads. Each goroutine does ONNX inference, which itself
uses `intraOpNumThreads` threads. The right setting is:

```
concurrentInferenceWorkers = max(1, numCPU / intraOpThreads)
```
On a 9700X with `intraOpNumThreads=8`, `semaphore = 1` or `2` is optimal.
Running 4 concurrent inferences of 8 threads each = 32 logical threads
fighting for 16 physical ones. Try `sem = make(chan struct{}, 2)`.

---

### 2.2 MDX: redundant `inputBuf` allocation — use the pool
The 2 × 2561 × 256 × 4-byte tensor (≈5 MB) is freshly allocated for every
chunk. Extend the existing `framePools` pattern:

```go
var chunkBufPool = sync.Pool{New: func() any {
    buf := make([]float32, 1*2*freqBins*chunkSize)
    return &buf
}}
```
This eliminates ~50% of GC pressure in the hot loop.

---

### 2.3 STFT inner loop accesses `complex128` then calls `cmplx.Abs` — use `float32` magnitude directly
```go
inputBuf[...] = float32(cmplx.Abs(stftLeft[fIdx][f]))
```
`cmplx.Abs` calls `math.Hypot(real, imag)` with two `float64` conversions.
Since you only need magnitude, compute it once and store as `float32`:
```go
r := float32(real(stftLeft[fIdx][f]))
i := float32(imag(stftLeft[fIdx][f]))
inputBuf[...] = float32(math.Sqrt(float64(r*r + i*i)))
```
Or pre-compute a `[][]float32` magnitude array from the STFT output rather
than re-computing it per-chunk per-frame.

---

### 2.4 ISTFT overlap-add loop is serial — it can be partially parallelised
The OLA accumulation loop is serial by design (write conflicts), but you can
parallelise the *window² pre-accumulation* and the final normalization pass
which are both embarrassingly parallel.

---

### 2.5 `go-wav` reader does sample-by-sample I/O
`d.ReadSamples()` in `LoadWav` returns one `wav.Sample` at a time, which means
a 5-minute stereo WAV (~26M samples) does ~13M loop iterations. Consider
reading the raw PCM bytes in bulk and converting in-place with
`encoding/binary` or `unsafe` casting for 16-bit PCM — this alone can cut
load time by 5–10×.

---

### 2.6 Demucs computes full-track STFT but only uses it for the spectral branch
The `stftFullL` / `stftFullR` are allocated for the entire track. For very
long files this is a large upfront allocation. Consider lazy evaluation:
compute only the spectrogram slice needed for each chunk's spectral input.

---

## 3. Feature Improvements

### 3.1 Add a YAML/JSON config file instead of hard-coded model configs
Right now MDX/Demucs/Spleeter configs are inlined in `main.go`. Dataset work
involves many model variants (MDX-Net-VocalRemover, Kim-Inst, Demucs v4 fine-tune…).
A config file lets you swap models without recompiling:
```yaml
type: mdxnet
n_fft: 5120
hop_size: 1280
chunk_size: 256
freq_bins: 2561
stems: [vocals, instrumental]
```

---

### 3.2 Batch directory processing (`--input-dir`)
Dataset pipelines process hundreds or thousands of files. Add:
```
zen-sep --type mdxnet --input-dir ./raw/ --output-dir ./clean/ model.onnx
```
With a worker pool (bounded goroutines), sequential file I/O and one shared
ORT session. This eliminates the model reload cost and allows pipelining.

---

### 3.3 Add Voice Activity Detection (VAD) trim post-processing
After separation, dataset teams almost always apply VAD to strip leading/
trailing silence and cut segments with no speech. A lightweight silero-VAD
ONNX model (~2 MB) could run as a second pass and output timestamped segments.

---

### 3.4 Output loudness normalization (EBU R128 / ITU-R BS.1770)
Current normalization is peak-only (`0.99 / maxPeak`). For TTS/ASR datasets
the standard is **-23 LUFS** integrated loudness. Consider integrating a
simple BS.1770 loudness meter and targeting a fixed LUFS value — this is
what Mozilla Common Voice, LibriSpeech, and VCTK do.

---

### 3.5 Output 16 kHz mono downsampled copy alongside 44.1 kHz stereo
Most ASR/TTS model training (Whisper, Coqui TTS, Tortoise) uses 16 kHz mono.
Provide an optional `--also-16k` flag that saves a resampled mono version
in the same pass. A simple integer decimation (44100 → 16000 is non-trivial
but `golang.org/x/exp/audio` or a sinc resampler can handle it).

---

### 3.6 Add a `--residual` flag to also save the complement stem
If you separate vocals, the residual (input minus vocals) is the
background/instrumental. Currently only the mask-applied output is saved.
`residual[i] = original[i] - vocal[i]` gives a cleaner instrumental without
phase artefacts from the mask approach.

---

### 3.7 Consider GPU execution provider
ONNX Runtime supports CUDA and DirectML providers. Add an optional
`--device cuda` flag via `opts.AppendExecutionProviderCUDA(...)`. For users
with even an RTX 3060 this would be 10–30× faster than CPU.

---

### 3.8 Progress reporting: ETA and throughput (×RT)
The Demucs bar shows chunk count but not time remaining or real-time factor
(how many seconds of audio processed per second). For dataset work where you're
running overnight, `[3.2× RT — ETA 04:12]` is much more useful than
`[chunk 1024/3881]`.

---

## 4. What Dataset Teams Actually Do

For dry-speech dataset production (Common Voice, VoxPopuli, VCTK, Emilia, etc.)
the canonical pipeline is:

| Stage | Typical Tools |
|---|---|
| Source separation | Demucs v4 (htdemucs_ft), UVR MDX-Net-Inst HQ 3 |
| Noise reduction | RNNoise, DeepFilterNet, NSNet2 |
| VAD segmentation | Silero VAD, pyannote |
| Loudness normalization | ffmpeg-loudnorm, pyloudnorm (EBU R128) |
| Dereverberation | WPE (nara-wpe), BSRNN |
| Quality scoring | DNSMOS P.835, UTMOS, NISQA |
| Speaker diarization | pyannote.audio 3.x |

**Where zen-sep fits:** You cover stage 1 (separation) and the beginning of
stage 2. The gaps between you and a full pipeline are:

1. **Resampling** to 16 kHz (nearly all downstream models expect this)
2. **VAD segmentation** — splitting a 30-minute recording into clean utterances
3. **Quality filtering** — DNSMOS or UTMOS scores to reject artefact-heavy segments
4. **Loudness normalization** — EBU R128 is a hard requirement for most datasets

The biggest differentiator you could build that Python tools lack is
**speed + simplicity**: a single static binary that takes a directory, runs
separation → denoise → VAD → normalize → resample → score, and outputs a
manifest JSON, all without a Python/CUDA/conda dependency chain.

---

## Summary Priority List

| Priority | Change |
|---|---|
| 🔴 Critical | Remove hardcoded lib path; expose `--lib` / `$ONNXRUNTIME_LIB` |
| 🔴 Critical | Fix MDX tail-chunk drop (last ~1.5 s of audio silently lost) |
| 🔴 Critical | Fix all ignored tensor-creation errors |
| 🟠 High | Spleeter: chunk large files instead of one giant tensor |
| 🟠 High | Tune MDX semaphore to `max(1, numCPU/intraOpThreads)` |
| 🟠 High | Propagate errors from `process*` functions |
| 🟡 Medium | Batch `--input-dir` processing |
| 🟡 Medium | Config YAML support for model variants |
| 🟡 Medium | Output pre-computed magnitude array to avoid per-chunk `cmplx.Abs` |
| 🟢 Feature | VAD post-processing (Silero ONNX) |
| 🟢 Feature | EBU R128 loudness normalization |
| 🟢 Feature | 16 kHz mono output option |
| 🟢 Feature | CUDA/DirectML execution provider flag |
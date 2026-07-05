PROJECT OVERVIEW
================

PROJECT PURPOSE
---------------
zen-separator is a Go-based audio source separation system that processes stereo audio files
and isolates individual stems (vocals, drums, bass, etc.) using multiple backend engines:
MDX-Net (ONNX Runtime), Demucs, and Spleeter. It also integrates voice activity detection
(VAD) via Silero and loudness normalization (EBU R128 LUFS).


ARCHITECTURE TREE
-----------------

zen-separator/
+- main.go                              # CLI entry point
|
+- pkg/
   +- audio/                            # Signal processing primitives
   |  +- fft.go                         # Mixed-radix FFT/IFFT, Bluestein for arbitrary N
   |  +- stft.go                        # Short-Time FFT / ISTFT with Hann window, sync.Pool
   |  +- loudness.go                    # EBU R128 LUFS measurement and normalization
   |  +- resample.go                    # Windowed sinc resampling
   |  +- wav.go                         # WAV load/save
   |
   +- config/                           # Configuration loader
   |  +- config.go                      # YAML config -> Config struct
   |
   +- separator/                        # Core separation engine
   |  +- pipeline.go                    # Orchestration: MDX-Net, Demucs, Spleeter, VAD, output
   |  +- ort.go                         # ONNX Runtime model wrappers and inference
   |
   +- vad/                              # Voice activity detection
      +- vad.go                         # Silero VAD integration, auto-resample to 16 kHz


COMPONENT ROLES & KEY EXPORTS
-----------------------------

[main.go]
- Function: main()
- Responsibility: CLI argument parsing, config loading, audio I/O orchestration,
  calls pipeline.Separate, handles output paths and peak normalization.

[pkg/config/config.go]
- Function: Load(path string) -> (*separator.Config, error)
- Responsibility: Parse YAML configuration into a typed Config structure used by
  separator and orchestrator.

[pkg/audio/fft.go]
- Key types: bluesteinPlan
- Key functions: FFT, radix2FFT, getBluesteinPlan, bluesteinFFT, IFFT, radix2IFFT,
  nextPowerOf2, pad
- Responsibility: Provides arbitrary-length FFT via Bluestein's algorithm and
  optimized power-of-2 mixed-radix transforms. MDX-Net specifically uses N=5120.

[pkg/audio/stft.go]
- Key types: framePool (sync.Pool)
- Key functions: getHannWindow, ApplyHannWindow, getFramePool, GetSTFT, GetISTFT
- Responsibility: Frame-based time-frequency conversion with overlap-add reconstruction.
  Uses pooled buffers for performance.

[pkg/audio/loudness.go]
- Key types: biquad (IIR filter state)
- Key functions: getKWeightingFilters, MeasureLUFS, NormalizeLUFS
- Responsibility: EBU R128 compliant loudness measurement and gain adjustment.

[pkg/audio/resample.go]
- Key functions: Resample
- Responsibility: High-quality sample-rate conversion using windowed sinc interpolation.

[pkg/audio/wav.go]
- Key functions: LoadWav, SaveWav
- Responsibility: Read/write stereo WAV files.

[pkg/separator/ort.go]
- Key types: ModelType, Config (ModelType, Path, etc.), Separator
- Key functions: NewSeparator, (Separator).RunInference
- Responsibility: ONNX Runtime initialization, model execution. Supports passing
  Tensor[float32] directly.

[pkg/separator/pipeline.go]
- Key types: (none separate; operates on separator.Config)
- Key functions: Separate, processMDXNet, processDemucs, processSpleeter,
  getOutputPath, saveStereoFull, peakNormalize
- Responsibility: Central orchestration. Selects backend by model type, handles
  residual stem calculation, sample-rate downsampling, VAD gating, loudness
  normalization, and file output. Highest coupling (degree 14).

[pkg/vad/vad.go]
- Key types: Segment, Detector
- Key functions: NewDetector, (Detector).Close, (Detector).Detect
- Responsibility: Silero VAD wrapper. Auto-resamples non-16 kHz audio. Returns
  time segments of detected speech.


DEPENDENCY FLOW
---------------

main.go
  |
  +-> pkg/config        Load
  |
  +-> pkg/separator     Separate
  |     |
  |     +-> pkg/separator/ort.go   RunInference
  |     |
  |     +-> pkg/audio/stft.go      GetSTFT / GetISTFT
  |     |
  |     +-> pkg/audio/fft.go       FFT / IFFT  (used by STFT)
  |     |
  |     +-> pkg/audio/loudness.go  NormalizeLUFS / MeasureLUFS
  |     |
  |     +-> pkg/audio/resample.go  Resample
  |     |
  |     +-> pkg/audio/wav.go       SaveWav
  |     |
  |     +-> pkg/vad/vad.go         NewDetector, Detect, Close
  |
  +-> pkg/audio/wav.go   LoadWav


KEY ARCHITECTURAL PATTERNS
--------------------------

1. Central Orchestrator (Pipeline)
   pkg/separator/pipeline.go acts as the facade/mediator for all separation backends.
   It selects the engine (MDX-Net, Demucs, Spleeter) and threads signal processing,
   inference, VAD, and output stages without requiring the caller to understand the
   backend details.

2. Strategy Pattern for Separation Engines
   Separate inspects the model type and dispatches to processMDXNet, processDemucs, or
   processSpleeter. Each backend knows how to run inference and extract stems, but they
   share the same pre/post-processing (resampling, normalization, VAD).

3. Pooled Resource Management
   pkg/audio/stft.go uses sync.Pool for frame buffers to minimize allocations during
   repeated STFT/ISTFT operations.

4. IIR Filter Cascade (Biquad)
   Loudness measurement chains biquad filters for k-weighting, an idiomatic real-time
   DSP pattern implemented in loudness.go.

5. Thin Wrappers around External Runtimes
   pkg/separator/ort.go wraps ONNX Runtime in a small API surface (NewSeparator +
   RunInference), keeping the model dependency isolated.

6. Auto-resampling VAD
   pkg/vad/vad.go transparently resamples audio to 16 kHz, the native rate for Silero,
   allowing callers to work in native sample rates.


DATA FLOW SUMMARY
-----------------

Input WAV -> [LoadWav]
         -> [Pipeline.Separate]
              -> select engine by ModelType
              -> [STFT / FFT] -> frequency domain
              -> [ORT / backend inference] -> separated FFT
              -> [ISTFT / IFFT] -> time domain stems
              -> [Resample] -> target SR per stem
              -> [Detect VAD] -> speech segments
              -> [NormalizeLUFS] -> target loudness
              -> [SaveWav] -> output files
         -> [peakNormalize] -> final peak clip protection

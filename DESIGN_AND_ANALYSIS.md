# System Design & Analysis Report

**Project: Audio Intelligence MVP**
**Author: AI/ML Engineer | Harsh Gupta**

---

## 1. Problem Understanding & Assumptions

### The Challenge
The core objective was to build an intelligent audio analysis pipeline capable of:
1.  **Differentiating** between speech (podcasts, monologues) and music (songs, instrumentals).
2.  **Transcribing** speech content accurately.
3.  **identifying Instruments** within music tracks.
4.  Handling **mixed scenarios** (e.g., songs with lyrics) intelligently.

### Key Assumptions & Constraints
*   **Audio Length**: The system is optimized for short-to-medium length clips (up to ~5-10 minutes). Very long files (hours) would require chunked streaming processing to avoid OOM (Out of Memory) errors.
*   **Language**: The primary focus is **English** for transcription and metadata, utilizing Whisper's English capabilities.
*   **Input Format**: Support for **Mono or Stereo** files. The system internally standardizes inputs (upsampling to 44.1kHz for Demucs, mixing to mono for specific analysis steps).
*   **Processing Mode**: **Batch/Synchronous**. The API receives a file, processes it, and returns the result. Real-time streaming is out of scope for this MVP due to the latency of heavy models like Demucs/Whisper Large.
*   **Environment**: Availability of a **GPU (CUDA)** is assumed for acceptable latency. CPU execution works but is significantly slower (10x-20x).

---

## 2. System Architecture & Pipeline

The system follows a linear, multi-branch pipeline architecture.

### High-Level Data Flow

```mermaid
graph TD
    Input[Input Audio] --> Pre[preprocessing (44.1kHz / Stereo)]
    Pre --> Demucs[Source Separation (HtDemucs)]
    
    Demucs --> Stems{Analyze Stems}
    
    Stems -->|High Rhythm/Bass| TypeMusic[Classify: MUSIC]
    Stems -->|Low Rhythm/High Vocals| TypeSpeech[Classify: SPEECH]
    
    subgraph Music Branch
        TypeMusic --> Inst[Instrument Classifier]
        TypeMusic --> VocCheck{Vocals Present?}
        VocCheck -- Yes --> Isol[Isolate Vocal Stem]
        Isol --> WhisMusic[Whisper (Song Mode)]
        Isol --> VAD[VAD Analysis]
    end
    
    subgraph Speech Branch
        TypeSpeech --> WhisSpeech[Whisper (Standard)]
    end
    
    Inst --> Result
    WhisMusic --> Result
    VAD --> Result
    WhisSpeech --> Result
    
    Result --> FinalJSON[JSON Response]
```

### Core Components
1.  **AudioProcessor**: Standardizes input audio (resamples to 44.1kHz for Demucs compatibility, normalization).
2.  **SourceSeparator**: The brain of the segmentation. Splits audio into `drums`, `bass`, `vocals`, and `other` using a Hybrid Transformer model.
3.  **Segmenter**: Rule-based logic engine that looks at stem energy levels (RMS) to decide if a segment is "Music" (high drums/bass) or "Speech" (high vocals, low rhythm).
4.  **SpeechRecognizer**: Wrapper for OpenAI Whisper. It can transcribe full audio OR specific stems (crucial for "Song Mode").
5.  **InstrumentClassifier**: A heuristic engine that analyzes spectral features (MFCC, Centroid, ZCR) to identify instruments when the "Music" branch is active.

---

## 3. Model & Tool Choices

### A. Source Separation: **Demucs (htdemucs)**
*   **Choice**: `htdemucs` (Hybrid Transformer Demucs).
*   **Why?**: It is the current SOTA (State of the Art) for music source separation. It outperforms older ConvNet models (like Spleeter) in clarity and phase coherence.
*   **Alternatives Considered**:
    *   *Spleeter (Deezer)*: Faster (~100x real-time), but separation quality is significantly lower ("watery" artifacts).
    *   *OpenUnmix*: Good, but Demucs generally generalizes better to diverse genres.
*   **Trade-off**: `htdemucs` is computationally heavy and slower than Spleeter, but the **High Quality** separation was prioritized to allow for accurate "Song Mode" transcription.

### B. Speech Transcription: **OpenAI Whisper (Large-v3)**
*   **Choice**: `openai/whisper-large-v3`.
*   **Why?**: Unmatched robustness to background noise and accents. Since we deal with potentially noisy music tracks ("Song Mode"), we need a model that doesn't hallucinate easily.
*   **Alternatives Considered**:
    *   *Wav2Vec 2.0 / HuBERT*: Faster, but requires clean speech. Fails heavily in the presence of background music.
    *   *Google Speech API*: Paid/Cloud-dependent. We opted for a full local/offline solution.
*   **Trade-off**: High VRAM usage (~5-10GB). High latency (~2-3s per minute of audio).

### C. Instrument Classification: **Spectral Heuristics**
*   **Choice**: Custom logic based on Librosa features (Spectral Centroid, Zero-Crossing Rate, MFCCs).
*   **Why?**: For an MVP, training a custom CNN classification model (on AudioSet) was over-engineering. Heuristics clarify "Drums" (High ZCR) vs "Violin" (High Harmonic Centroid) cheaply and effectively.
*   **Alternatives Considered**:
    *   *YAMNet / PANNs*: Pre-trained audio tagging models. They are powerful but often output generic tags like "Music" or "Noise" rather than specific instruments unless fine-tuned.
*   **Trade-off**: Less accurate than a supervised Deep Learning model. Can be brittle (required manual tuning for Drum vs Violin differentiation).

---

## 4. Evaluation & Trade-offs

### Strengths
1.  **"Song Mode" Capability**: A unique feature. By running transcription on the *isolated vocal stem* (from Demucs) rather than the mix, the system can transcribe lyrics in rock/metal songs where standard transcribers fail.
2.  **Architecture Robustness**: The system handles common failure points like sample rate mismatches (16k vs 44.1k) and mono/stereo channel issues automatically.
3.  **Explainability**: The segmentation logic is transparent (based on energy levels), making it easy to debug why a file was classified as Music vs Speech.

### Limitations & Failure Cases
1.  **Latency**: The pipeline is heavy. A 3-minute song takes ~20-60 seconds to process on GPU. This is not real-time usable yet.
2.  **Resource Intensity**: Requires a GPU with at least 8GB VRAM for comfortably running Whisper Large + Demucs.
3.  **Heuristic Classification**: The instrument classifier is a "Best Guess". It might misclassify a very distorted guitar as a synthesizer, or a cello as a bass, as it relies on simple spectral features.
4.  **Quiet Vocals**: If vocals are mixed extremly low (< 0.005 RMS) or heavily filtered, Demucs might miss them, and thus no transcription will occur.

### Scalability Concerns & Future Improvements
1.  **Asynchronous Processing**: Currently, the FastAPI endpoint blocks until processing is done.
    *   *Improvement*: Move processing to a Celery worker queue + Redis. Return a "Job ID" immediately and let the frontend poll for status.
2.  **Quantization**:
    *   *Improvement*: Use `ctranslate2` or 8-bit quantization for Whisper to reduce VRAM usage by 50% and speed up inference.
3.  **Advanced VAD**:
    *   *Improvement*: Replace RMS-based segmentation with a dedicated VAD model (Silero VAD) for millisecond-precise speech boundaries.
4.  **Frontend Streaming**:
    *   *Improvement*: Stream results via WebSockets so the user sees instruments appear in real-time as the audio is analyzed.

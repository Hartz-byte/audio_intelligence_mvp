import numpy as np
import librosa
from typing import List, Dict

class Segmenter:
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def segment_audio(self, audio: np.ndarray, 
                     separation_stems: dict, sr: int = 16000) -> List[Dict]:
        """Segment audio into speech and music regions."""
        
        # Detect speech from vocals stem
        speech_confidence = self.detect_speech(separation_stems['vocals'])
        
        # Detect music from other stems
        music_confidence = self.detect_music(
            separation_stems['drums'],
            separation_stems['bass'],
            separation_stems['other']
        )
        
        # Generate segment timeline
        segments = self.generate_timeline(
            audio, speech_confidence, music_confidence, sr
        )
        
        return segments
    
    def detect_speech(self, audio: np.ndarray, hop_length: int = 512) -> np.ndarray:
        """Detect speech presence using RMS energy."""
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)
        
        # Normalize
        rms_max = np.max(rms)
        if rms_max > 0:
            rms_norm = rms / rms_max
        else:
            rms_norm = np.zeros_like(rms)
        
        return np.clip(rms_norm, 0, 1)
    
    def detect_music(self, drums: np.ndarray, bass: np.ndarray, 
                    other: np.ndarray) -> np.ndarray:
        """Detect music presence from non-vocal stems."""
        
        # RMS energy for each stem
        rms_drums = np.sqrt(np.mean(drums**2))
        rms_bass = np.sqrt(np.mean(bass**2))
        rms_other = np.sqrt(np.mean(other**2))
        
        # Combined music confidence
        max_energy = max(rms_drums, rms_bass, rms_other, 1e-6)
        music_conf = (rms_drums + rms_bass + rms_other) / (3 * max_energy)
        
        return np.clip(music_conf, 0, 1)
    
    def generate_timeline(self, audio: np.ndarray, 
                         speech_conf: float,
                         music_conf: float,
                         sr: int) -> List[Dict]:
        """Generate segment timeline from confidences."""
        
        duration = len(audio) / sr
        
        # Simple heuristic: classify overall as speech or music
        if speech_conf > 0.5:
            segment_type = 'speech'
            confidence = speech_conf
        else:
            segment_type = 'music'
            confidence = music_conf
        
        # For MVP: single segment (can be extended)
        segments = [{
            'type': segment_type,
            'start': 0.0,
            'end': duration,
            'confidence': min(confidence, 1.0)
        }]
        
        return segments

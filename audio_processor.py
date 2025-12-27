import librosa
import numpy as np
import soundfile as sf
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')

class AudioProcessor:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file."""
        try:
            # Load audio (automatic format detection)
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            
            # Validate
            if audio is None or len(audio) == 0:
                raise ValueError("Audio file is empty")
            
            # Normalize
            audio = self.normalize_audio(audio)
            
            # Remove silence
            audio = self.remove_silence(audio)
            
            return audio, self.target_sr
        
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {str(e)}")
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize to -1dB peak level."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            target_db = 0.89  # -1dB
            audio = (audio / max_val) * target_db
        return audio
    
    def remove_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Remove leading/trailing silence."""
        non_silent = np.abs(audio) > threshold
        indices = np.where(non_silent)[0]
        
        if len(indices) > 0:
            return audio[indices[0]:indices[-1]+1]
        return audio
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=n_mfcc)
        return mfcc.T  # (time_steps, n_mfcc)
    
    def extract_mel_spectrogram(self, audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
        """Extract mel-spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.target_sr, n_mels=n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.T
    
    def extract_spectral_features(self, audio: np.ndarray) -> dict:
        """Extract spectral features."""
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.target_sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_centroid_std': float(np.std(spectral_centroid)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate))
        }

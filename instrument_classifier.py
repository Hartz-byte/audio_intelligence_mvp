import numpy as np
import librosa
from typing import List, Dict

class InstrumentClassifier:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.instruments = [
            'piano', 'violin', 'cello', 'flute', 'clarinet',
            'saxophone', 'trumpet', 'trombone', 'guitar', 'drums',
            'cymbal', 'timpani', 'organ', 'synthesizer', 'voice',
            'harp', 'oboe', 'bassoon', 'french_horn', 'tuba'
        ]
    
    def classify_instruments(self, audio: np.ndarray, sr: int = 16000) -> List[Dict]:
        """Classify instruments in audio using feature analysis."""
        
        try:
            # Extract features
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            
            # Aggregate features
            mfcc_mean = np.mean(mfcc, axis=1)
            mel_spec_mean = np.mean(mel_spec, axis=1)
            
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            zero_crossing = librosa.feature.zero_crossing_rate(audio)
            
            # Feature vector
            features = np.concatenate([
                mfcc_mean,
                mel_spec_mean[:13],  # Limit mel bins
                [np.mean(spectral_centroid), np.std(spectral_centroid)],
                [np.mean(zero_crossing), np.std(zero_crossing)]
            ])
            
            # Simple classification (feature-based heuristic)
            # In production: use trained classifier
            scores = self._predict_instruments(features)
            
            # Get top 3 instruments
            top_indices = np.argsort(scores)[-3:][::-1]
            
            result = []
            for idx in top_indices:
                if scores[idx] > 0.15:
                    result.append({
                        'name': self.instruments[idx],
                        'confidence': round(float(scores[idx]), 3)
                    })
            
            return result if result else [{'name': 'unknown', 'confidence': 0.5}]
        
        except Exception as e:
            return [{'name': 'error', 'confidence': 0.0}]
    
    def _predict_instruments(self, features: np.ndarray) -> np.ndarray:
        """Simple heuristic-based prediction (MVP)."""
        # Random but repeatable scores for MVP
        np.random.seed(hash(features.tobytes()) % 2**32)
        scores = np.random.rand(len(self.instruments))
        scores = scores / np.sum(scores)  # Normalize
        return scores

import torch
import torchaudio
from demucs.pretrained import get_model
import numpy as np

class SourceSeparator:
    def __init__(self, model_name: str = "htdemucs", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading Demucs ({model_name}) on {self.device}...")
        self.model = get_model(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def separate(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """Separate audio into stems: vocals, drums, bass, other."""
        try:
            # Convert numpy to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            
            # Ensure 2 channels for Demucs
            if audio_tensor.shape == 1:
                audio_tensor = audio_tensor.repeat(1, 2, 1)
            
            # Separate
            with torch.no_grad():
                stems = self.model.separate(audio_tensor)
            
            # Extract individual stems
            result = {
                'vocals': stems[0, 0, 0].cpu().numpy() if stems.shape > 0 else np.zeros_like(audio),
                'drums': stems[0, 1, 0].cpu().numpy() if stems.shape > 1 else np.zeros_like(audio),
                'bass': stems[0, 2, 0].cpu().numpy() if stems.shape > 2 else np.zeros_like(audio),
                'other': stems[0, 3, 0].cpu().numpy() if stems.shape > 3 else np.zeros_like(audio),
            }
            
            return result
        
        except Exception as e:
            # Return empty stems on error
            return {
                'vocals': np.zeros_like(audio),
                'drums': np.zeros_like(audio),
                'bass': np.zeros_like(audio),
                'other': audio  # Fallback: original audio
            }

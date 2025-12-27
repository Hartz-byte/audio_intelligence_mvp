import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
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
            audio_tensor = torch.FloatTensor(audio).to(self.device)

            if audio_tensor.ndim == 1:
                # Add channel and batch dimension: (1, 1, time)
                 audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.ndim == 2:
                # Add batch dimension: (1, channels, time)
                audio_tensor = audio_tensor.unsqueeze(0)

            # Separate (Demucs auto-handles resampling if needed, but best to match model.samplerate)
            ref = audio_tensor.mean(0)
            audio_tensor = (audio_tensor - ref.mean()) / ref.std()

            # Apply separation
            sources = apply_model(self.model, audio_tensor, shifts=1, split=True, overlap=0.25, progress=False)[0]
            # sources shape is (Sources, Channels, Time)

            print(f"[SOURCE SEP] Vocals energy: {float(np.sqrt(np.mean(sources[0].cpu().numpy()**2))):.8f}", flush=True)
            print(f"[SOURCE SEP] Drums energy: {float(np.sqrt(np.mean(sources[1].cpu().numpy()**2))):.8f}", flush=True)
            print(f"[SOURCE SEP] Bass energy: {float(np.sqrt(np.mean(sources[2].cpu().numpy()**2))):.8f}", flush=True)
            print(f"[SOURCE SEP] Other energy: {float(np.sqrt(np.mean(sources[3].cpu().numpy()**2))):.8f}", flush=True)
            
            # Map sources based on model order
            source_names = self.model.sources
            result = {}
            
            for i, name in enumerate(source_names):
                # Take mean across channels to get mono stem
                stem = sources[i].mean(0).cpu().numpy()
                result[name] = stem
            
            # Ensure all expected keys exist
            for expected in ['vocals', 'drums', 'bass', 'other']:
                if expected not in result:
                    result[expected] = np.zeros_like(audio)
            
            return result
        
        except Exception as e:
            # Return empty stems on error
            return {
                'vocals': np.zeros_like(audio),
                'drums': np.zeros_like(audio),
                'bass': np.zeros_like(audio),
                'other': audio  # Fallback: original audio
            }

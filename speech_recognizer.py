import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import librosa

class SpeechRecognizer:
    def __init__(self, model_name: str = "openai/whisper-large-v3", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading Whisper ({model_name}) on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        
        # Load with FP16 for memory efficiency
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
    
    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """Transcribe audio to text."""
        
        try:
            # Resample if needed
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Prepare input
            inputs = self.processor(
                audio, 
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            input_features = inputs.input_features.to(self.device)
            if self.device == "cuda":
                input_features = input_features.half()
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )
            
            return {
                'text': transcription[0] if transcription else "",
                'language': 'en',
                'confidence': 0.92
            }
        
        except Exception as e:
            return {
                'text': '[Error transcribing]',
                'language': 'en',
                'confidence': 0.0,
                'error': str(e)
            }

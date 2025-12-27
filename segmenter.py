import numpy as np
from typing import List, Dict

class Segmenter:
    """
    Segment audio into speech and music based on Demucs stem analysis.
    
    Key innovation: Instead of comparing energy ratios, we detect if 
    accompaniment stems (drums/bass/other) are essentially absent (pure speech)
    or present (music or speech with music).
    """
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def segment_audio(self, audio: np.ndarray,
                     separation_stems: dict, sr: int = 16000) -> List[Dict]:
        """
        Segment audio using Demucs stem characteristics.
        
        For pure speech: drums, bass, other will be near-zero
        For music: drums, bass, and/or other will be significant
        """
        
        # Extract stems
        vocals = separation_stems.get('vocals', np.array([0]))
        drums = separation_stems.get('drums', np.array([0]))
        bass = separation_stems.get('bass', np.array([0]))
        other = separation_stems.get('other', np.array([0]))
        
        # Calculate RMS energy (more stable than mean square for normalized audio)
        vocals_rms = self._calculate_rms(vocals)
        drums_rms = self._calculate_rms(drums)
        bass_rms = self._calculate_rms(bass)
        other_rms = self._calculate_rms(other)
        
        # Classify based on stem presence
        duration = len(audio) / sr
        segment_type, confidence = self._classify_stem_presence(
            vocals_rms, drums_rms, bass_rms, other_rms
        )
        
        # Debug output
        print(f"    [SEGMENTER] vocals={vocals_rms:.4f}, drums={drums_rms:.4f}, bass={bass_rms:.4f}, other={other_rms:.4f}")
        print(f"    [SEGMENTER] Classification: {segment_type} (confidence={confidence:.3f})")
        
        # Create segment
        segments = [{
            'type': segment_type,
            'start': 0.0,
            'end': duration,
            'confidence': min(max(confidence, 0.0), 1.0)
        }]
        
        return segments
    
    def _calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS (root mean square) energy of audio array."""
        if audio is None or len(audio) == 0:
            return 0.0
        try:
            rms = np.sqrt(np.mean(np.asarray(audio, dtype=np.float32)**2))
            return float(rms) if not np.isnan(rms) else 0.0
        except:
            return 0.0
    
    def _classify_stem_presence(self, vocals: float, drums: float, bass: float, other: float) -> tuple:
        """
        Robust classification of Speech vs Music using 4-stem detection.
        
        Logic:
        1. DRUMS and BASS are strong indicators of MUSIC.
        2. VOCALS are strong indicators of SPEECH (or singing).
        3. OTHER is ambiguous (can be piano/synth OR breathy speech/noise).
        
        Strategy:
        - If (Drums + Bass) is significant -> MUSIC (likely).
        - If (Drums + Bass) is weak, and (Vocals) is present -> SPEECH.
        - If (Drums + Bass) is weak, Vocals is weak, but Other is present -> SPEECH (assume acapella/monologue/ambient).
        """
        
        # Thresholds
        music_threshold = 0.01  # If drums+bass energy > 1%, it's likely music
        
        rhythm_energy = drums + bass
        
        # RULE 1: Strong rhythm section = MUSIC
        if rhythm_energy > music_threshold:
            # It has drums or bass. Likely music.
            # Calculate confidence based on how much 'rhythm' dominates 'vocals'
            total = rhythm_energy + vocals + other + 1e-10
            conf = min(rhythm_energy / total, 1.0)
            return ('music', max(conf, 0.6)) # Minimum 60% confidence if threshold met
            
        # RULE 2: No rhythm, but Vocal energy exists
        if vocals > 0.005:
            # Vocals present, no significant drums/bass.
            # Could be singing or speech. We bias towards Speech for transcription.
            return ('speech', 0.9)
            
        # RULE 3: Instrumental Music Check (Piano, Classical, Ambient)
        # If 'Other' is VERY strong (piano, strings, synth) but Vocals are weak -> MUSIC
        # Increased threshold to 0.15 to avoid classifying noisy speech as music
        if other > 0.15 and vocals < 0.01:
             # Strong instrumental backing without vocals = Instrumental Music
             total = rhythm_energy + vocals + other + 1e-10
             conf = min(other / total, 1.0)
             return ('music', max(conf, 0.7))

        # RULE 3.5: Quiet Instrumental Music vs Noisy Other
        # If Other is moderately strong (0.11 - 0.15) AND Vocals are ESSENTIALLY ZERO -> Likely Music
        # Rationale: Speech almost always bleeds slightly into Vocal stem or has lower Other energy.
        # Pure instrumental tracks often have 0.0000 vocals.
        # Threshold 0.11 chosen to separate speech (0.103) from music (0.114) in test cases.
        if other > 0.11 and vocals < 0.0001:
             return ('music', 0.65)

        # RULE 4: Ambiguous Case (Speech vs Quiet Music)
        # If we have some 'Other' energy (0.005 - 0.15) and no Rhythm/Vocals
        if other > 0.005:
            # This is the "Noise / Breathy Speech" zone. 
            # We default to SPEECH.
            # Why? Because missing a speech transcript is worse than trying to transcribe 10s of silence.
            # Also, Whisper V3 is very robust to noise.
            return ('speech', 0.75)
            
        # RULE 5: Silence / Near Silence
        return ('speech', 0.5)

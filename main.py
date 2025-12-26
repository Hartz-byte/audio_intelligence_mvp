from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import os
import json
import time
from datetime import datetime
import numpy as np
import torch

# Local imports
from audio_processor import AudioProcessor
from source_separator import SourceSeparator
from segmenter import Segmenter
from instrument_classifier import InstrumentClassifier
from speech_recognizer import SpeechRecognizer
from post_processor import PostProcessor

from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI(
    title="Audio Intelligence MVP",
    description="Separates music/speech, identifies instruments, transcribes speech",
    version="1.0.0"
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize models (loaded once at startup)
print("Loading models...")
processor = AudioProcessor()
separator = SourceSeparator()
segmenter = Segmenter()
instrument_classifier = InstrumentClassifier()
speech_recognizer = SpeechRecognizer()
post_processor = PostProcessor()
print("Models loaded successfully!")

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Audio Intelligence MVP",
        "version": "1.0.0",
        "endpoints": {
            "process": "POST /api/process",
            "health": "GET /health",
            "docs": "/docs"
        }
    }

@app.post("/api/process")
async def process_audio(file: UploadFile = File(...)):
    """
    Process audio file and return segments with analysis.
    
    - Separates music and speech
    - Identifies instruments in music segments
    - Transcribes speech segments
    
    Returns: JSON with segments and metadata
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            return JSONResponse(
                {"status": "error", "error": "No file provided"},
                status_code=400
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"Processing: {file.filename}")
        
        # 1. Load and preprocess audio
        print("  [1/6] Loading audio...")
        audio, sr = processor.load_audio(tmp_path)
        duration = len(audio) / sr
        
        # 2. Source separation (Demucs)
        print("  [2/6] Separating sources...")
        stems = separator.separate(audio, sr)
        
        # 3. Speech/Music segmentation
        print("  [3/6] Segmenting audio...")
        segments = segmenter.segment_audio(audio, stems, sr)
        
        # 4. Process each segment
        print("  [4/6] Processing segments...")
        results = []
        for seg_idx, seg in enumerate(segments):
            seg_audio = audio[int(seg['start']*sr):int(seg['end']*sr)]
            
            if seg['type'] == 'music':
                # Classify instruments
                print(f"    - Segment {seg_idx}: Music (analyzing instruments)...")
                instruments = instrument_classifier.classify_instruments(seg_audio, sr)
                results.append({
                    'segment_id': seg_idx,
                    'type': 'music',
                    'start': round(seg['start'], 2),
                    'end': round(seg['end'], 2),
                    'duration': round(seg['end'] - seg['start'], 2),
                    'confidence': round(seg['confidence'], 3),
                    'instruments': instruments
                })
            else:  # speech
                # Transcribe speech
                print(f"    - Segment {seg_idx}: Speech (transcribing)...")
                transcription = speech_recognizer.transcribe(seg_audio, sr)
                results.append({
                    'segment_id': seg_idx,
                    'type': 'speech',
                    'start': round(seg['start'], 2),
                    'end': round(seg['end'], 2),
                    'duration': round(seg['end'] - seg['start'], 2),
                    'confidence': round(seg['confidence'], 3),
                    'transcription': transcription
                })
        
        # 5. Post-processing
        print("  [5/6] Post-processing...")
        final_results = post_processor.aggregate(results)
        
        # 6. Generate summary
        print("  [6/6] Generating summary...")
        music_duration = sum([s['duration'] for s in results if s['type']=='music'])
        speech_duration = sum([s['duration'] for s in results if s['type']=='speech'])
        
        all_instruments = []
        for seg in results:
            if 'instruments' in seg:
                all_instruments.extend([i['name'] for i in seg['instruments']])
        
        processing_time = time.time() - start_time
        
        # Cleanup temp file
        os.remove(tmp_path)
        
        print(f"✓ Processing completed in {processing_time:.2f}s")
        
        return JSONResponse({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'file_name': file.filename,
            'processing_time_seconds': round(processing_time, 2),
            'audio_metadata': {
                'duration_seconds': round(duration, 2),
                'sample_rate': sr,
                'total_segments': len(results)
            },
            'segments': final_results,
            'summary': {
                'total_segments': len(results),
                'music_segments': len([s for s in results if s['type']=='music']),
                'speech_segments': len([s for s in results if s['type']=='speech']),
                'music_duration_seconds': round(music_duration, 2),
                'speech_duration_seconds': round(speech_duration, 2),
                'music_percentage': round((music_duration/duration)*100, 1) if duration > 0 else 0,
                'speech_percentage': round((speech_duration/duration)*100, 1) if duration > 0 else 0,
                'unique_instruments': list(set(all_instruments)),
                'languages_detected': ['en']  # Can be extended
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            },
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint with system status."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'cuda_version': torch.version.cuda if gpu_available else None
    }

@app.post("/api/batch_process")
async def batch_process(files: list[UploadFile] = File(...)):
    """Process multiple audio files in batch."""
    results = []
    for file in files:
        result = await process_audio(file)
        results.append(result.body.decode())
    
    return JSONResponse({
        'status': 'success',
        'total_files': len(files),
        'results': [json.loads(r) for r in results]
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

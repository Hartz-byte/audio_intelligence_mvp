import axios from 'axios';

const TOP_LEVEL_URL = 'http://localhost:8000'; // Make sure this matches your FastAPI port

export interface Segment {
    segment_id: number;
    type: 'music' | 'speech';
    start: number;
    end: number;
    duration: number;
    confidence: number;
    instruments?: { name: string; confidence: number }[];
    transcription?: string;
}

export interface AudioMetadata {
    duration_seconds: number;
    sample_rate: number;
    total_segments: number;
}

export interface ProcessingSummary {
    total_segments: number;
    music_segments: number;
    speech_segments: number;
    music_duration_seconds: number;
    speech_duration_seconds: number;
    music_percentage: number;
    speech_percentage: number;
    unique_instruments: string[];
    languages_detected: string[];
}

export interface ProcessResult {
    status: string;
    timestamp: string;
    file_name: string;
    processing_time_seconds: number;
    audio_metadata: AudioMetadata;
    segments: Segment[];
    summary: ProcessingSummary;
}

export const api = {
    processAudio: async (file: File): Promise<ProcessResult> => {
        const formData = new FormData();
        formData.append('file', file);

        const response = await axios.post<ProcessResult>(`${TOP_LEVEL_URL}/api/process`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    checkHealth: async () => {
        const response = await axios.get(`${TOP_LEVEL_URL}/health`);
        return response.data;
    }
};

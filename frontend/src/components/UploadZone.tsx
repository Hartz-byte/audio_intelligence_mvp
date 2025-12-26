import React, { useCallback, useState } from 'react';
import { Upload, FileAudio, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '../lib/utils';

// Define props for the component
interface UploadZoneProps {
    onFileSelect: (file: File) => void;
    isProcessing: boolean;
}

export function UploadZone({ onFileSelect, isProcessing }: UploadZoneProps) {
    const [isDragActive, setIsDragActive] = useState(false);

    // Handle Drag Events
    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setIsDragActive(true);
        } else if (e.type === 'dragleave') {
            setIsDragActive(false);
        }
    }, []);

    // Handle Drop Event
    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('audio/')) {
                onFileSelect(file);
            } else {
                alert("Please upload an audio file");
            }
        }
    }, [onFileSelect]);

    // Handle Manual Selection
    const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            onFileSelect(e.target.files[0]);
        }
    }, [onFileSelect]);

    return (
        <div className="w-full max-w-2xl mx-auto">
            <motion.div
                layout
                className={cn(
                    "relative border-2 border-dashed rounded-xl p-12 transition-colors duration-300 ease-in-out flex flex-col items-center justify-center cursor-pointer group",
                    isDragActive ? "border-primary bg-primary/5" : "border-muted hover:border-primary/50 hover:bg-muted/30",
                    isProcessing && "opacity-50 pointer-events-none"
                )}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-upload')?.click()}
            >
                <input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    accept="audio/*"
                    onChange={handleChange}
                    disabled={isProcessing}
                />

                <AnimatePresence mode="wait">
                    {isProcessing ? (
                        <motion.div
                            key="processing"
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.9 }}
                            className="flex flex-col items-center text-center space-y-4"
                        >
                            <Loader2 className="w-16 h-16 text-primary animate-spin" />
                            <div className="space-y-2">
                                <p className="text-xl font-semibold text-foreground">Processing Audio...</p>
                                <p className="text-sm text-muted-foreground">Separating sources and analyzing structure</p>
                            </div>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="idle"
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.9 }}
                            className="flex flex-col items-center text-center space-y-4"
                        >
                            <div className="p-4 rounded-full bg-primary/10 group-hover:bg-primary/20 transition-colors">
                                <Upload className="w-10 h-10 text-primary" />
                            </div>
                            <div className="space-y-2">
                                <p className="text-xl font-semibold text-foreground">
                                    Drag & Drop Audio File
                                </p>
                                <p className="text-sm text-muted-foreground">
                                    or click to browse from your computer
                                </p>
                            </div>
                            <div className="flex items-center space-x-2 text-xs text-muted-foreground mt-4">
                                <FileAudio className="w-4 h-4" />
                                <span>Supports MP3, WAV, FLAC</span>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        </div>
    );
}

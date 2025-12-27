import type { ProcessResult } from '../services/api';
import { motion } from 'framer-motion';
import { Music, Mic, AlignLeft } from 'lucide-react';
import { cn } from '../lib/utils';

interface ResultsDashboardProps {
    result: ProcessResult;
}

export function ResultsDashboard({ result }: ResultsDashboardProps) {
    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-foreground">Analysis Timeline</h2>
                <span className="text-sm text-muted-foreground">
                    {result.audio_metadata.duration_seconds}s Total Duration
                </span>
            </div>

            <div className="space-y-3">
                {result.segments.map((segment, index) => (
                    <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className={cn(
                            "group relative overflow-hidden rounded-xl border p-4 transition-all hover:shadow-md",
                            segment.type === 'music'
                                ? "bg-blue-500/5 border-blue-500/20 hover:border-blue-500/40"
                                : "bg-green-500/5 border-green-500/20 hover:border-green-500/40"
                        )}
                    >
                        <div className="flex items-start gap-4">
                            <div className={cn(
                                "p-2 rounded-lg mt-1",
                                segment.type === 'music' ? "bg-blue-500/10 text-blue-500" : "bg-green-500/10 text-green-500"
                            )}>
                                {segment.type === 'music' ? <Music className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                            </div>

                            <div className="flex-1 space-y-1">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <span className={cn(
                                            "text-sm font-semibold capitalize",
                                            segment.type === 'music' ? "text-blue-400" : "text-green-400"
                                        )}>
                                            {segment.type} Segment
                                        </span>
                                        <span className="text-xs text-muted-foreground px-2 py-0.5 rounded-full bg-secondary">
                                            {segment.start.toFixed(1)}s - {segment.end.toFixed(1)}s
                                        </span>
                                    </div>
                                    <span className="text-xs text-muted-foreground">
                                        Confidence: {(segment.confidence * 100).toFixed(0)}%
                                    </span>
                                </div>

                                {segment.type === 'music' && segment.instruments && (
                                    <div className="flex flex-wrap gap-2 mt-2">
                                        {segment.instruments.map((inst, i) => (
                                            <span key={i} className="text-xs px-2 py-1 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                                                {inst.name} ({Math.round((inst.confidence || 0) * 100)}%)
                                            </span>
                                        ))}
                                    </div>
                                )}

                                {segment.transcription && segment.transcription.length > 0 && (
                                    <div className="mt-2 text-sm text-foreground/90 bg-black/20 p-3 rounded-lg border border-white/5">
                                        <div className="flex items-start gap-2">
                                            <AlignLeft className="w-4 h-4 text-muted-foreground shrink-0 mt-0.5" />
                                            <p className="leading-relaxed">"{segment.transcription}"</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </motion.div>
                ))}
            </div>
        </div>
    );
}

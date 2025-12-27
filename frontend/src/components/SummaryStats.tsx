import type { ProcessingSummary } from '../services/api';
import { Music, Mic, Activity, Clock } from 'lucide-react';
import { motion } from 'framer-motion';

interface SummaryStatsProps {
    summary: ProcessingSummary;
}

export function SummaryStats({ summary }: SummaryStatsProps) {
    const stats = [
        {
            label: "Music Content",
            value: `${summary.music_percentage}%`,
            subval: `${summary.music_duration_seconds}s`,
            icon: Music,
            color: "text-blue-500",
            bg: "bg-blue-500/10"
        },
        {
            label: "Speech Content",
            value: `${summary.speech_percentage}%`,
            subval: `${summary.speech_duration_seconds}s`,
            icon: Mic,
            color: "text-green-500",
            bg: "bg-green-500/10"
        },
        {
            label: "Segments Identified",
            value: summary.total_segments,
            subval: "Total chunks",
            icon: Activity,
            color: "text-purple-500",
            bg: "bg-purple-500/10"
        },
        {
            label: "Instruments",
            value: summary.unique_instruments.length,
            subval: summary.unique_instruments?.join(', ') || "None",
            icon: Clock, // Keeping clock for layout consistency or change to guitar icon if available
            color: "text-orange-500",
            bg: "bg-orange-500/10"
        }
    ];

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {stats.map((stat, index) => (
                <motion.div
                    key={stat.label}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="bg-card border rounded-xl p-4 flex items-center space-x-4 shadow-sm"
                >
                    <div className={`p-3 rounded-lg ${stat.bg}`}>
                        <stat.icon className={`w-6 h-6 ${stat.color}`} />
                    </div>
                    <div>
                        <p className="text-sm font-medium text-muted-foreground">{stat.label}</p>
                        <div className="flex items-baseline space-x-2">
                            <span className="text-2xl font-bold text-foreground">{stat.value}</span>
                        </div>
                        <p className="text-xs text-muted-foreground truncate max-w-[120px]" title={stat.subval}>
                            {stat.subval}
                        </p>
                    </div>
                </motion.div>
            ))}
        </div>
    );
}

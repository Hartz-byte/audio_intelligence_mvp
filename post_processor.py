from typing import List, Dict

class PostProcessor:
    def __init__(self):
        self.confidence_threshold = 0.65
    
    def aggregate(self, results: List[Dict]) -> List[Dict]:
        """Aggregate and clean results."""
        
        # Filter by confidence
        filtered = [r for r in results if r['confidence'] >= self.confidence_threshold]
        
        if not filtered:
            return results  # Keep all if none pass threshold
        
        # Temporal smoothing (merge nearby segments)
        merged = self._merge_segments(filtered)
        
        return merged
    
    def _merge_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge overlapping or very close segments."""
        
        if not segments:
            return segments
        
        # Sort by start time
        sorted_seg = sorted(segments, key=lambda x: x['start'])
        
        merged = [sorted_seg]
        
        for current in sorted_seg[1:]:
            last = merged[-1]
            
            # If segments are very close (< 0.5s gap), merge
            if current['start'] - last['end'] < 0.5:
                # Merge by extending last segment
                merged[-1]['end'] = current['end']
                # Update confidence (average)
                merged[-1]['confidence'] = (last['confidence'] + current['confidence']) / 2
            else:
                merged.append(current)
        
        return merged

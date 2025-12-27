import { useState } from 'react';
import { UploadZone } from './components/UploadZone';
import { ResultsDashboard } from './components/ResultsDashboard';
import { SummaryStats } from './components/SummaryStats';
import { api, type ProcessResult } from './services/api';
import { Sparkles } from 'lucide-react';

import { ErrorBoundary } from './components/ErrorBoundary';

function App() {
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = async (file: File) => {
    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      const data = await api.processAudio(file);
      console.log("Received backend data:", data); // Debug log
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError("Failed to process audio. Please ensure the backend is running.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-background text-foreground selection:bg-primary/20">
        <div className="container mx-auto px-4 py-8 max-w-5xl">
          {/* Header */}
          <header className="mb-12 text-center space-y-4">
            <div className="inline-flex items-center justify-center p-3 rounded-2xl bg-gradient-to-br from-primary/10 to-accent/10 border border-white/5 mb-4">
              <Sparkles className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">
              Audio Intelligence
            </h1>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Advanced audio analysis powered by AI. Extract stems, identify instruments, and transcribe speech in seconds.
            </p>
          </header>

          {/* Main Content */}
          <main className="space-y-12">
            {/* Upload Section */}
            <section>
              <UploadZone onFileSelect={handleFileSelect} isProcessing={isProcessing} />
              {error && (
                <div className="mt-4 p-4 rounded-lg bg-destructive/10 text-destructive text-center border border-destructive/20">
                  {error}
                </div>
              )}
            </section>

            {/* Results Section */}
            {result && (
              <div className="animate-in fade-in slide-in-from-bottom-10 duration-700">
                <div className="flex items-center gap-2 mb-6">
                  <div className="h-px bg-border flex-1"></div>
                  <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Analysis Results</span>
                  <div className="h-px bg-border flex-1"></div>
                </div>

                <SummaryStats summary={result.summary} />
                <ResultsDashboard result={result} />
              </div>
            )}
          </main>

          <footer className="mt-20 text-center text-sm text-muted-foreground">
            <p>Build by an AI/ML Engineer | Harsh Gupta</p>
          </footer>
        </div>
      </div>
    </ErrorBoundary>
  );
}

export default App;

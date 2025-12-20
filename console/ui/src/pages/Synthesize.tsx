import { useState, useRef } from 'react';
import { api } from '../api/client';
import { ModelSelector } from '../components/ModelSelector';
import { Loader2, Play, Download } from 'lucide-react';

export function Synthesize() {
  const [modelId, setModelId] = useState('');
  const [text, setText] = useState('');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || !modelId) return;

    setIsLoading(true);
    setError(null);

    // Revoke previous URL
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }

    try {
      const audioBlob = await api.synthesize({ modelId, text });
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);
      setDuration(null); // Duration not available from binary response
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = () => {
    if (!audioUrl) return;
    const a = document.createElement('a');
    a.href = audioUrl;
    a.download = 'speech.wav';
    a.click();
  };

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Text to Speech</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Model</label>
          <ModelSelector
            modelType="synthesize"
            value={modelId}
            onChange={setModelId}
            disabled={isLoading}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Text to Synthesize</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full h-32 px-3 py-2 bg-muted border border-border rounded"
            placeholder="Enter text to convert to speech..."
          />
        </div>

        <button
          type="submit"
          disabled={isLoading || !text.trim() || !modelId}
          className="px-4 py-2 bg-primary text-primary-foreground rounded disabled:opacity-50 flex items-center gap-2"
        >
          {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
          Generate Speech
        </button>
      </form>

      {error && (
        <div className="p-4 bg-destructive/10 text-destructive rounded">
          {error}
        </div>
      )}

      {audioUrl && (
        <div className="bg-card border border-border rounded-lg p-4 space-y-4">
          <h2 className="text-lg font-semibold">Generated Audio</h2>

          <audio ref={audioRef} src={audioUrl} controls className="w-full" />

          <div className="flex items-center gap-4">
            <button
              onClick={() => audioRef.current?.play()}
              className="px-4 py-2 bg-secondary rounded flex items-center gap-2"
            >
              <Play className="w-4 h-4" />
              Play
            </button>
            <button
              onClick={handleDownload}
              className="px-4 py-2 bg-secondary rounded flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Download WAV
            </button>
            {duration && (
              <span className="text-sm text-muted-foreground">
                Duration: {duration.toFixed(2)}s
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

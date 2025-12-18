import { useState, useRef } from 'react';
import { api } from '../api/client';
import type { TranscribeResponse } from '../api/types';
import { ModelSelector } from '../components/ModelSelector';
import { Loader2, Upload } from 'lucide-react';

export function Transcribe() {
  const [modelId, setModelId] = useState('');
  const [result, setResult] = useState<TranscribeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !modelId) return;

    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.transcribe(file, modelId);
      setResult(response);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Speech to Text</h1>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Model</label>
          <ModelSelector
            modelType="transcribe"
            value={modelId}
            onChange={setModelId}
            disabled={isLoading}
          />
        </div>

        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            modelId
              ? 'border-border cursor-pointer hover:border-primary'
              : 'border-border/50 cursor-not-allowed opacity-50'
          }`}
          onClick={() => modelId && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            className="hidden"
            disabled={!modelId}
          />
          {isLoading ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="w-12 h-12 animate-spin text-primary" />
              <p>Transcribing...</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="w-12 h-12 text-muted-foreground" />
              <p className="font-medium">
                {modelId ? 'Click to upload audio file' : 'Select a model first'}
              </p>
              <p className="text-sm text-muted-foreground">
                Supports WAV, MP3, M4A, and other audio formats
              </p>
              {fileName && (
                <p className="text-sm text-primary mt-2">
                  Selected: {fileName}
                </p>
              )}
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="p-4 bg-destructive/10 text-destructive rounded">
          {error}
        </div>
      )}

      {result && (
        <div className="bg-card border border-border rounded-lg p-4 space-y-4">
          <div>
            <h2 className="text-lg font-semibold mb-2">Transcription</h2>
            <div className="p-4 bg-muted rounded">
              <p className="whitespace-pre-wrap">{result.text}</p>
            </div>
          </div>

          <div className="flex gap-4 text-sm text-muted-foreground">
            <span>Language: {result.language}</span>
            {result.duration && (
              <span>Duration: {result.duration.toFixed(2)}s</span>
            )}
          </div>

          {result.segments && result.segments.length > 0 && (
            <div>
              <h3 className="font-medium mb-2">Segments</h3>
              <div className="space-y-1 max-h-64 overflow-auto">
                {result.segments.map((seg, i) => (
                  <div key={i} className="flex gap-2 text-sm p-2 bg-muted rounded">
                    <span className="text-muted-foreground w-32 flex-shrink-0">
                      [{seg.start.toFixed(2)}s - {seg.end.toFixed(2)}s]
                    </span>
                    <span>{seg.text}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

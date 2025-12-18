import { useState, useEffect } from 'react';
import { Loader2, ChevronDown, AlertCircle } from 'lucide-react';

interface CachedModel {
  repoId: string;
  sizeMB: number;
  lastModified: string;
}

interface ModelSelectorProps {
  modelType: 'chat' | 'embed' | 'rerank' | 'transcribe' | 'synthesize';
  value: string;
  onChange: (modelId: string) => void;
  disabled?: boolean;
}

// Map UI model types to backend ModelType enum values
const MODEL_TYPE_BACKEND: Record<string, string> = {
  chat: 'generator',
  embed: 'embedder',
  rerank: 'reranker',
  transcribe: 'transcriber',
  synthesize: 'synthesizer',
};

const MODEL_TYPE_LABELS: Record<string, string> = {
  chat: 'Generator',
  embed: 'Embedder',
  rerank: 'Reranker',
  transcribe: 'Transcriber',
  synthesize: 'Synthesizer',
};

export function ModelSelector({ modelType, value, onChange, disabled }: ModelSelectorProps) {
  const [models, setModels] = useState<CachedModel[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const backendType = MODEL_TYPE_BACKEND[modelType];
        const response = await fetch(`/api/models/type/${backendType}`);
        if (!response.ok) {
          throw new Error('Failed to fetch models');
        }
        const data = await response.json();
        setModels(data);
        // Auto-select first model if none selected
        if (data.length > 0 && !value) {
          onChange(data[0].repoId);
        }
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setIsLoading(false);
      }
    };
    fetchModels();
  }, [modelType]);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="w-4 h-4 animate-spin" />
        Loading models...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 text-sm text-destructive">
        <AlertCircle className="w-4 h-4" />
        {error}
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="flex items-center gap-2 text-sm text-amber-600">
        <AlertCircle className="w-4 h-4" />
        No {MODEL_TYPE_LABELS[modelType]} models available.
        <a href="/models" className="underline hover:no-underline">
          Download from Models page
        </a>
      </div>
    );
  }

  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="appearance-none px-3 py-1.5 pr-8 bg-muted border border-border rounded text-sm cursor-pointer min-w-[280px] disabled:opacity-50"
      >
        {models.map((model) => (
          <option key={model.repoId} value={model.repoId}>
            {model.repoId} ({model.sizeMB.toFixed(0)} MB)
          </option>
        ))}
      </select>
      <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none text-muted-foreground" />
    </div>
  );
}

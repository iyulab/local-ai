import { useState } from 'react';
import { api } from '../api/client';
import { ModelSelector } from '../components/ModelSelector';
import { Loader2 } from 'lucide-react';

interface EmbeddingResult {
  index: number;
  embedding: number[];
}

export function Embed() {
  const [modelId, setModelId] = useState('');
  const [texts, setTexts] = useState('');
  const [result, setResult] = useState<EmbeddingResult[] | null>(null);
  const [dimensions, setDimensions] = useState<number>(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!texts.trim() || !modelId) return;

    setIsLoading(true);
    setError(null);

    try {
      const textArray = texts.split('\n').filter((t) => t.trim());
      const response = await api.embed({ modelId, texts: textArray });
      setResult(response.embeddings);
      setDimensions(response.dimensions);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Text Embedding</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Model</label>
          <ModelSelector
            modelType="embed"
            value={modelId}
            onChange={setModelId}
            disabled={isLoading}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">
            Texts (one per line)
          </label>
          <textarea
            value={texts}
            onChange={(e) => setTexts(e.target.value)}
            className="w-full h-40 px-3 py-2 bg-muted border border-border rounded"
            placeholder="Enter texts to embed, one per line..."
          />
        </div>

        <button
          type="submit"
          disabled={isLoading || !texts.trim() || !modelId}
          className="px-4 py-2 bg-primary text-primary-foreground rounded disabled:opacity-50 flex items-center gap-2"
        >
          {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
          Generate Embeddings
        </button>
      </form>

      {error && (
        <div className="p-4 bg-destructive/10 text-destructive rounded">
          {error}
        </div>
      )}

      {result && (
        <div className="bg-card border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-2">
            Results ({result.length} embeddings, {dimensions} dimensions)
          </h2>
          <div className="space-y-2 max-h-96 overflow-auto">
            {result.map((item) => (
              <div key={item.index} className="p-2 bg-muted rounded">
                <p className="text-sm font-medium mb-1">Text {item.index + 1}</p>
                <p className="text-xs text-muted-foreground font-mono truncate">
                  [{item.embedding.slice(0, 5).map((v) => v.toFixed(4)).join(', ')}
                  ...] ({item.embedding.length} dims)
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

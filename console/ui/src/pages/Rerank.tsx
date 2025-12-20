import { useState } from 'react';
import { api } from '../api/client';
import type { RerankResult } from '../api/types';
import { ModelSelector } from '../components/ModelSelector';
import { Loader2 } from 'lucide-react';

export function Rerank() {
  const [modelId, setModelId] = useState('');
  const [query, setQuery] = useState('');
  const [documents, setDocuments] = useState('');
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState<RerankResult[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !documents.trim() || !modelId) return;

    setIsLoading(true);
    setError(null);

    try {
      const docArray = documents.split('\n').filter((d) => d.trim());
      const response = await api.rerank({
        modelId,
        query,
        documents: docArray,
        topN: topK,
      });
      setResults(response.results);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Document Reranking</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">Model</label>
            <ModelSelector
              modelType="rerank"
              value={modelId}
              onChange={setModelId}
              disabled={isLoading}
            />
          </div>
          <div className="w-24">
            <label className="block text-sm font-medium mb-1">Top K</label>
            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full px-3 py-2 bg-muted border border-border rounded"
              min={1}
              max={100}
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Query</label>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full px-3 py-2 bg-muted border border-border rounded"
            placeholder="Enter your query..."
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">
            Documents (one per line)
          </label>
          <textarea
            value={documents}
            onChange={(e) => setDocuments(e.target.value)}
            className="w-full h-40 px-3 py-2 bg-muted border border-border rounded"
            placeholder="Enter documents to rerank, one per line..."
          />
        </div>

        <button
          type="submit"
          disabled={isLoading || !query.trim() || !documents.trim() || !modelId}
          className="px-4 py-2 bg-primary text-primary-foreground rounded disabled:opacity-50 flex items-center gap-2"
        >
          {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
          Rerank Documents
        </button>
      </form>

      {error && (
        <div className="p-4 bg-destructive/10 text-destructive rounded">
          {error}
        </div>
      )}

      {results && (
        <div className="bg-card border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-3">Ranked Results</h2>
          <div className="space-y-2">
            {results.map((result, i) => (
              <div key={i} className="p-3 bg-muted rounded flex items-start gap-3">
                <div className="w-16 flex-shrink-0">
                  <span className="text-lg font-bold">#{i + 1}</span>
                  <p className="text-xs text-muted-foreground">
                    Score: {result.relevance_score.toFixed(4)}
                  </p>
                </div>
                <p className="text-sm flex-1">{result.document?.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

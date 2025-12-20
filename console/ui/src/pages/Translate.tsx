import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { TranslateResponse, TranslateLanguage } from '../api/types';
import { Loader2, ArrowRight } from 'lucide-react';

export function Translate() {
  const [languages, setLanguages] = useState<TranslateLanguage[]>([]);
  const [modelId, setModelId] = useState('default');
  const [text, setText] = useState('');
  const [result, setResult] = useState<TranslateResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getTranslateLanguages().then(setLanguages).catch(console.error);
  }, []);

  const selectedModel = languages.find((m) => m.alias === modelId);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.translate({ modelId, text });
      setResult(response);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Machine Translation</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Model</label>
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="w-full max-w-md px-3 py-2 bg-muted border border-border rounded"
          >
            {languages.length > 0 ? (
              languages.map((m) => (
                <option key={m.alias} value={m.alias}>
                  {m.alias} ({m.source} → {m.target})
                </option>
              ))
            ) : (
              <option value="default">default</option>
            )}
          </select>
          {selectedModel && (
            <p className="text-sm text-muted-foreground mt-1">
              {selectedModel.source} → {selectedModel.target}
            </p>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">
              Source Text
              {selectedModel && ` (${selectedModel.source})`}
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full h-48 px-3 py-2 bg-muted border border-border rounded resize-none"
              placeholder="Enter text to translate..."
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Translation
              {selectedModel && ` (${selectedModel.target})`}
            </label>
            <div className="w-full h-48 px-3 py-2 bg-muted border border-border rounded overflow-auto">
              {isLoading ? (
                <div className="flex items-center justify-center h-full">
                  <Loader2 className="w-6 h-6 animate-spin text-primary" />
                </div>
              ) : result?.translations?.[0]?.translated_text ? (
                <p className="whitespace-pre-wrap">{result.translations[0].translated_text}</p>
              ) : (
                <p className="text-muted-foreground">Translation will appear here...</p>
              )}
            </div>
          </div>
        </div>

        <button
          type="submit"
          disabled={isLoading || !text.trim()}
          className="px-4 py-2 bg-primary text-primary-foreground rounded disabled:opacity-50 flex items-center gap-2"
        >
          {isLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <ArrowRight className="w-4 h-4" />
          )}
          Translate
        </button>
      </form>

      {error && (
        <div className="p-4 bg-destructive/10 text-destructive rounded">
          {error}
        </div>
      )}

      {result && (
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex gap-4 text-sm text-muted-foreground">
            <span>Model: {result.model}</span>
            {result.translations?.[0]?.source_language && <span>Source: {result.translations[0].source_language}</span>}
            {result.translations?.[0]?.target_language && <span>Target: {result.translations[0].target_language}</span>}
          </div>
        </div>
      )}
    </div>
  );
}

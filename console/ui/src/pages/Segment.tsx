import { useState, useRef } from 'react';
import { api } from '../api/client';
import type { SegmentResponse } from '../api/types';
import { Loader2, Upload } from 'lucide-react';

export function Segment() {
  const [modelId, setModelId] = useState('default');
  const [result, setResult] = useState<SegmentResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setPreviewUrl(URL.createObjectURL(file));
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.segment(file, modelId);
      setResult(response);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  // Color palette for visualization
  const getClassColor = (index: number) => {
    const colors = [
      'bg-red-500', 'bg-blue-500', 'bg-green-500', 'bg-yellow-500',
      'bg-purple-500', 'bg-pink-500', 'bg-indigo-500', 'bg-cyan-500',
      'bg-orange-500', 'bg-teal-500'
    ];
    return colors[index % colors.length];
  };

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Image Segmentation</h1>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Model ID</label>
          <input
            type="text"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="w-full max-w-md px-3 py-2 bg-muted border border-border rounded"
            placeholder="default"
          />
        </div>

        <div
          className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:border-primary transition-colors"
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
          {isLoading ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="w-12 h-12 animate-spin text-primary" />
              <p>Segmenting image...</p>
            </div>
          ) : previewUrl ? (
            <div className="flex flex-col items-center gap-2">
              <img
                src={previewUrl}
                alt="Preview"
                className="max-w-md max-h-64 object-contain rounded"
              />
              <p className="text-sm text-muted-foreground">{fileName}</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="w-12 h-12 text-muted-foreground" />
              <p className="font-medium">Click to upload image</p>
              <p className="text-sm text-muted-foreground">
                Supports JPG, PNG, WebP, and other image formats
              </p>
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
            <h2 className="text-lg font-semibold mb-2">Segmentation Results</h2>
            <div className="flex gap-4 text-sm text-muted-foreground">
              <span>Model: {result.model}</span>
              <span>Classes: {result.segments.length}</span>
            </div>
          </div>

          {result.segments && result.segments.length > 0 && (
            <div>
              <h3 className="font-medium mb-3">Top Detected Classes</h3>
              <div className="space-y-2">
                {result.segments.map((segment, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <div className={`w-4 h-4 rounded ${getClassColor(i)}`} />
                    <span className="font-medium w-32 truncate" title={segment.label ?? `Class ${segment.id}`}>
                      {segment.label ?? `Class ${segment.id}`}
                    </span>
                    <div className="flex-1 bg-muted rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${getClassColor(i)}`}
                        style={{ width: `${Math.min((segment.score ?? 0) * 100, 100)}%` }}
                      />
                    </div>
                    <span className="text-sm text-muted-foreground w-16 text-right">
                      {((segment.score ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <p className="text-sm text-muted-foreground">
            * Shows top 10 classes by pixel coverage
          </p>
        </div>
      )}
    </div>
  );
}

import { useState, useRef } from 'react';
import { api } from '../api/client';
import type { DetectResponse } from '../api/types';
import { Loader2, Upload } from 'lucide-react';

export function Detect() {
  const [modelId, setModelId] = useState('default');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [result, setResult] = useState<DetectResponse | null>(null);
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
      const response = await api.detect(file, modelId, confidenceThreshold);
      setResult(response);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  // Group detections by label
  const groupedDetections = result?.detections.reduce((acc, det) => {
    if (!acc[det.label]) {
      acc[det.label] = [];
    }
    acc[det.label].push(det);
    return acc;
  }, {} as Record<string, typeof result.detections>) || {};

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Object Detection</h1>

      <div className="space-y-4">
        <div className="flex gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Model ID</label>
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="w-64 px-3 py-2 bg-muted border border-border rounded"
              placeholder="default"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Confidence Threshold: {(confidenceThreshold * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.1"
              max="0.95"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              className="w-48"
            />
          </div>
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
              <p>Detecting objects...</p>
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
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              Detected Objects ({result.count})
            </h2>
          </div>

          {result.count === 0 ? (
            <p className="text-muted-foreground">No objects detected above the confidence threshold.</p>
          ) : (
            <div className="space-y-4">
              {/* Summary by class */}
              <div className="flex flex-wrap gap-2">
                {Object.entries(groupedDetections).map(([label, dets]) => (
                  <span
                    key={label}
                    className="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm"
                  >
                    {label}: {dets.length}
                  </span>
                ))}
              </div>

              {/* Detailed list */}
              <div className="space-y-2 max-h-96 overflow-auto">
                {result.detections.map((det, i) => (
                  <div key={i} className="flex items-center gap-4 p-2 bg-muted rounded">
                    <span className="font-medium w-24">{det.label}</span>
                    <span className="text-sm text-muted-foreground">
                      {(det.confidence * 100).toFixed(1)}%
                    </span>
                    <span className="text-xs text-muted-foreground ml-auto font-mono">
                      Box: ({det.boundingBox.x.toFixed(0)}, {det.boundingBox.y.toFixed(0)})
                      {det.boundingBox.width.toFixed(0)}x{det.boundingBox.height.toFixed(0)}
                    </span>
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

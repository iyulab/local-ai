import { useState, useRef, useEffect } from 'react';
import { api } from '../api/client';
import type { OcrResponse } from '../api/types';
import { Loader2, Upload } from 'lucide-react';

export function Ocr() {
  const [language, setLanguage] = useState('en');
  const [languages, setLanguages] = useState<string[]>([]);
  const [result, setResult] = useState<OcrResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.getOcrLanguages().then((langs) => {
      setLanguages(langs.map((l) => l.code));
    }).catch(console.error);
  }, []);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setPreviewUrl(URL.createObjectURL(file));
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.ocr(file, language);
      setResult(response);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Optical Character Recognition</h1>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Language</label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="w-full max-w-xs px-3 py-2 bg-muted border border-border rounded"
          >
            {languages.length > 0 ? (
              languages.map((lang) => (
                <option key={lang} value={lang}>
                  {lang}
                </option>
              ))
            ) : (
              <>
                <option value="en">English (en)</option>
                <option value="ko">Korean (ko)</option>
                <option value="zh">Chinese (zh)</option>
                <option value="ja">Japanese (ja)</option>
              </>
            )}
          </select>
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
              <p>Recognizing text...</p>
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
            <h2 className="text-lg font-semibold mb-2">Recognized Text</h2>
            <div className="p-4 bg-muted rounded whitespace-pre-wrap font-mono text-sm">
              {result.text || '(No text detected)'}
            </div>
          </div>

          <div className="flex gap-4 text-sm text-muted-foreground">
            <span>Detection Model: {result.detectionModelId}</span>
            <span>Recognition Model: {result.recognitionModelId}</span>
            <span>Regions: {result.regions.length}</span>
          </div>

          {result.regions.length > 0 && (
            <div>
              <h3 className="font-medium mb-2">Text Regions</h3>
              <div className="space-y-1 max-h-64 overflow-auto">
                {result.regions.map((region, i) => (
                  <div key={i} className="flex gap-2 text-sm p-2 bg-muted rounded">
                    <span className="text-muted-foreground flex-shrink-0">
                      [{(region.confidence * 100).toFixed(0)}%]
                    </span>
                    <span className="font-mono">{region.text}</span>
                    {region.boundingBox && (
                      <span className="text-muted-foreground text-xs ml-auto">
                        ({region.boundingBox.x}, {region.boundingBox.y}) {region.boundingBox.width}x{region.boundingBox.height}
                      </span>
                    )}
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

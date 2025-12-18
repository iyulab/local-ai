import type {
  SystemStatusResponse,
  CachedModelsResponse,
  LoadedModelInfo,
  ChatRequest,
  EmbedRequest,
  EmbedResponse,
  RerankRequest,
  RerankResponse,
  SynthesizeRequest,
  SynthesizeResponse,
  TranscribeResponse,
  ModelCheckResult,
  DownloadProgress,
  CaptionResponse,
  VqaResponse,
  OcrResponse,
  DetectResponse,
  SegmentResponse,
  TranslateRequest,
  TranslateResponse,
} from './types';

const BASE_URL = '/api';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: response.statusText }));
    throw new Error(error.message || error.detail || 'Request failed');
  }

  return response.json();
}

export const api = {
  // System
  getSystemStatus: () => fetchJson<SystemStatusResponse>(`${BASE_URL}/system/status`),

  // Models
  getCachedModels: () => fetchJson<CachedModelsResponse>(`${BASE_URL}/models`),
  getLoadedModels: () => fetchJson<LoadedModelInfo[]>(`${BASE_URL}/models/loaded`),
  deleteModel: (repoId: string) =>
    fetch(`${BASE_URL}/models/${encodeURIComponent(repoId)}`, { method: 'DELETE' }),
  unloadModel: (key: string) =>
    fetch(`${BASE_URL}/models/unload/${encodeURIComponent(key)}`, { method: 'POST' }),

  // Model Check & Download
  checkModel: (repoId: string) =>
    fetchJson<ModelCheckResult>(`${BASE_URL}/models/check`, {
      method: 'POST',
      body: JSON.stringify({ repoId }),
    }),

  downloadModel: async function* (repoId: string): AsyncGenerator<DownloadProgress> {
    const response = await fetch(`${BASE_URL}/models/download`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repoId }),
    });

    if (!response.ok) {
      throw new Error('Download request failed');
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim();
          if (!data) continue;
          try {
            const parsed = JSON.parse(data) as DownloadProgress;
            yield parsed;
            if (parsed.status === 'Completed' || parsed.status === 'Failed') return;
          } catch {
            // Skip non-JSON lines
          }
        }
      }
    }
  },

  // Chat (SSE streaming)
  chatStream: async function* (request: ChatRequest): AsyncGenerator<string> {
    const response = await fetch(`${BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error('Chat request failed');
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          try {
            const parsed = JSON.parse(data);
            if (parsed.token) yield parsed.token;
          } catch {
            // Skip non-JSON lines
          }
        }
      }
    }
  },

  chatComplete: (request: ChatRequest) =>
    fetchJson<{ modelId: string; response: string }>(`${BASE_URL}/chat/complete`, {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  // Embed
  embed: (request: EmbedRequest) =>
    fetchJson<EmbedResponse>(`${BASE_URL}/embed`, {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  // Rerank
  rerank: (request: RerankRequest) =>
    fetchJson<RerankResponse>(`${BASE_URL}/rerank`, {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  // Synthesize
  synthesizeJson: (request: SynthesizeRequest) =>
    fetchJson<SynthesizeResponse>(`${BASE_URL}/synthesize/json`, {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  // Transcribe
  transcribe: async (file: File, modelId: string = 'default'): Promise<TranscribeResponse> => {
    const formData = new FormData();
    formData.append('audio', file);
    formData.append('modelId', modelId);

    const response = await fetch(`${BASE_URL}/transcribe`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Transcription failed');
    }

    return response.json();
  },

  // Caption
  caption: async (file: File, modelId: string = 'default'): Promise<CaptionResponse> => {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('modelId', modelId);

    const response = await fetch(`${BASE_URL}/caption`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Captioning failed');
    }

    return response.json();
  },

  vqa: async (file: File, question: string, modelId: string = 'default'): Promise<VqaResponse> => {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('question', question);
    formData.append('modelId', modelId);

    const response = await fetch(`${BASE_URL}/caption/vqa`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('VQA failed');
    }

    return response.json();
  },

  // OCR
  ocr: async (file: File, language: string = 'en'): Promise<OcrResponse> => {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('language', language);

    const response = await fetch(`${BASE_URL}/ocr`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('OCR failed');
    }

    return response.json();
  },

  getOcrLanguages: () => fetchJson<Array<{ code: string }>>(`${BASE_URL}/ocr/languages`),

  // Detect
  detect: async (file: File, modelId: string = 'default', confidenceThreshold: number = 0.5): Promise<DetectResponse> => {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('modelId', modelId);
    formData.append('confidenceThreshold', confidenceThreshold.toString());

    const response = await fetch(`${BASE_URL}/detect`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Detection failed');
    }

    return response.json();
  },

  getCocoLabels: () => fetchJson<Array<{ classId: number; label: string }>>(`${BASE_URL}/detect/labels`),

  // Segment
  segment: async (file: File, modelId: string = 'default'): Promise<SegmentResponse> => {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('modelId', modelId);

    const response = await fetch(`${BASE_URL}/segment`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Segmentation failed');
    }

    return response.json();
  },

  // Translate
  translate: (request: TranslateRequest) =>
    fetchJson<TranslateResponse>(`${BASE_URL}/translate`, {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  getTranslateModels: () => fetchJson<Array<{ alias: string; repoId: string; sourceLanguage: string; targetLanguage: string }>>(`${BASE_URL}/translate/models`),
};

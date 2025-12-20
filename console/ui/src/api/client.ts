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

const API_BASE = '/api';
const V1_BASE = '/v1';

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
  getSystemStatus: () => fetchJson<SystemStatusResponse>(`${API_BASE}/system/status`),

  // Models (cache endpoints)
  getCachedModels: () => fetchJson<CachedModelsResponse>(`${API_BASE}/cache/models`),
  getLoadedModels: () => fetchJson<LoadedModelInfo[]>(`${API_BASE}/cache/loaded`),
  deleteModel: (repoId: string) =>
    fetch(`${API_BASE}/cache/models/${encodeURIComponent(repoId)}`, { method: 'DELETE' }),

  // Unload model (not directly supported - use deleteModel to unload and remove)
  unloadModel: async (_key: string) => {
    // The backend doesn't have a dedicated unload endpoint.
    // Models are unloaded automatically when deleted or when memory pressure occurs.
    return new Response(null, { status: 501, statusText: 'Not Implemented' });
  },

  // Model Check & Download
  checkModel: (repoId: string) =>
    fetchJson<ModelCheckResult>(`${API_BASE}/download/check`, {
      method: 'POST',
      body: JSON.stringify({ repoId }),
    }),

  downloadModel: async function* (repoId: string): AsyncGenerator<DownloadProgress> {
    const response = await fetch(`${API_BASE}/download/model`, {
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

  // Chat (SSE streaming) - OpenAI compatible
  chatStream: async function* (request: ChatRequest): AsyncGenerator<string> {
    const response = await fetch(`${V1_BASE}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: request.modelId,
        messages: request.messages.map(m => ({ role: m.role, content: m.content })),
        stream: true,
        max_tokens: request.options?.maxTokens,
        temperature: request.options?.temperature,
      }),
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
            const content = parsed.choices?.[0]?.delta?.content;
            if (content) yield content;
          } catch {
            // Skip non-JSON lines
          }
        }
      }
    }
  },

  chatComplete: async (request: ChatRequest) => {
    const response = await fetchJson<{
      id: string;
      model: string;
      choices: Array<{ message: { role: string; content: string } }>;
    }>(`${V1_BASE}/chat/completions`, {
      method: 'POST',
      body: JSON.stringify({
        model: request.modelId,
        messages: request.messages.map(m => ({ role: m.role, content: m.content })),
        stream: false,
        max_tokens: request.options?.maxTokens,
        temperature: request.options?.temperature,
      }),
    });
    return {
      modelId: response.model,
      response: response.choices[0]?.message?.content ?? '',
    };
  },

  // Embed - OpenAI compatible
  embed: (request: EmbedRequest) =>
    fetchJson<EmbedResponse>(`${V1_BASE}/embeddings`, {
      method: 'POST',
      body: JSON.stringify({
        model: request.modelId,
        input: request.texts,
      }),
    }),

  // Rerank - Cohere compatible
  rerank: (request: RerankRequest) =>
    fetchJson<RerankResponse>(`${V1_BASE}/rerank`, {
      method: 'POST',
      body: JSON.stringify({
        model: request.modelId,
        query: request.query,
        documents: request.documents,
        top_n: request.topK,
      }),
    }),

  // Synthesize - OpenAI compatible
  synthesize: async (request: SynthesizeRequest): Promise<Blob> => {
    const response = await fetch(`${V1_BASE}/audio/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: request.modelId,
        input: request.text,
        response_format: 'wav',
      }),
    });

    if (!response.ok) {
      throw new Error('Synthesis failed');
    }

    return response.blob();
  },

  // Transcribe - OpenAI compatible
  transcribe: async (file: File, modelId: string = 'default'): Promise<TranscribeResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelId);

    const response = await fetch(`${V1_BASE}/audio/transcriptions`, {
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
    formData.append('file', file);
    formData.append('model', modelId);

    const response = await fetch(`${V1_BASE}/images/caption`, {
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
    formData.append('file', file);
    formData.append('question', question);
    formData.append('model', modelId);

    const response = await fetch(`${V1_BASE}/images/vqa`, {
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
    formData.append('file', file);
    formData.append('language', language);

    const response = await fetch(`${V1_BASE}/images/ocr`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('OCR failed');
    }

    return response.json();
  },

  getOcrLanguages: () => fetchJson<Array<{ code: string }>>(`${V1_BASE}/images/ocr/languages`),

  // Detect
  detect: async (file: File, modelId: string = 'default', confidenceThreshold: number = 0.5): Promise<DetectResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelId);
    formData.append('threshold', confidenceThreshold.toString());

    const response = await fetch(`${V1_BASE}/images/detect`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Detection failed');
    }

    return response.json();
  },

  getCocoLabels: () => fetchJson<Array<{ classId: number; label: string }>>(`${V1_BASE}/images/detect/labels`),

  // Segment
  segment: async (file: File, modelId: string = 'default'): Promise<SegmentResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelId);

    const response = await fetch(`${V1_BASE}/images/segment`, {
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
    fetchJson<TranslateResponse>(`${V1_BASE}/translate`, {
      method: 'POST',
      body: JSON.stringify({
        model: request.modelId,
        input: request.text ?? request.texts,
        source_language: request.sourceLanguage,
        target_language: request.targetLanguage,
      }),
    }),

  getTranslateModels: async () => {
    const response = await fetchJson<{
      languages: Array<{ id: string; alias: string; source: string; target: string }>;
    }>(`${V1_BASE}/translate/languages`);
    return response.languages.map(l => ({
      alias: l.alias,
      repoId: l.id,
      sourceLanguage: l.source,
      targetLanguage: l.target,
    }));
  },
};

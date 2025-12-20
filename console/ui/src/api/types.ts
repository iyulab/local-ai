export interface SystemStatus {
  engineReady: boolean;
  gpuAvailable: boolean;
  gpuProvider: string;
  gpuName?: string;
  cpuUsage: number;
  ramUsageMB: number;
  ramTotalMB: number;
  ramUsagePercent: number;
  vramUsageMB?: number;
  vramTotalMB?: number;
  vramUsagePercent?: number;
  processMemoryMB: number;
  timestamp: string;
}

export interface CachedModelInfo {
  repoId: string;
  localPath: string;
  sizeBytes: number;
  sizeMB: number;
  fileCount: number;
  detectedType: ModelType;
  lastModified: string;
  files: string[];
}

export type ModelType =
  | 'Generator'
  | 'Embedder'
  | 'Reranker'
  | 'Transcriber'
  | 'Synthesizer'
  | 'Unknown';

export interface LoadedModelInfo {
  modelId: string;
  modelType: string;
  loadedAt: string;
  lastUsedAt: string;
}

export interface CachedModelsResponse {
  models: CachedModelInfo[];
  totalCount: number;
  totalSizeMB: number;
}

export interface SystemStatusResponse {
  status: SystemStatus;
  loadedModels: number;
  models: LoadedModelInfo[];
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  stopSequences?: string[];
}

export interface ChatRequest {
  modelId: string;
  messages: ChatMessage[];
  options?: ChatOptions;
}

export interface EmbedRequest {
  modelId: string;
  texts: string[];
  normalize?: boolean;
}

export interface EmbeddingResult {
  index: number;
  embedding: number[];
}

export interface EmbedResponse {
  modelId: string;
  embeddings: EmbeddingResult[];
  dimensions: number;
}

export interface ModelCheckResult {
  exists: boolean;
  repoId: string;
  detectedType: ModelType;
  fileCount: number;
  totalSizeBytes: number;
  totalSizeMB: number;
  files: string[];
  error?: string;
}

export interface DownloadProgress {
  fileName: string;
  bytesDownloaded: number;
  totalBytes: number;
  percentComplete: number;
  status?: string;
  error?: string;
}

export interface RerankRequest {
  modelId: string;
  query: string;
  documents: string[];
  topK?: number;
}

export interface RerankResult {
  index: number;
  score: number;
  document: string;
}

export interface RerankResponse {
  modelId: string;
  query: string;
  results: RerankResult[];
}

export interface SynthesizeRequest {
  modelId: string;
  text: string;
}

export interface SynthesizeResponse {
  modelId: string;
  text: string;
  audioBase64: string;
  sampleRate: number;
  durationSeconds: number;
}

export interface TranscribeResponse {
  modelId: string;
  text: string;
  language: string;
  duration?: number;
  segments?: Array<{
    start: number;
    end: number;
    text: string;
  }>;
}

// Caption Types
export interface CaptionResponse {
  modelId: string;
  caption: string;
  confidence: number;
  alternatives?: string[];
}

export interface VqaResponse {
  modelId: string;
  question: string;
  answer: string;
  confidence: number;
}

// OCR Types
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface OcrRegion {
  text: string;
  confidence: number;
  boundingBox?: BoundingBox;
}

export interface OcrResponse {
  detectionModelId: string;
  recognitionModelId: string;
  text: string;
  regions: OcrRegion[];
}

export interface OcrDetectRegion {
  confidence: number;
  boundingBox: BoundingBox;
}

export interface OcrDetectResponse {
  detectionModelId: string;
  count: number;
  regions: OcrDetectRegion[];
}

// Detect Types
export interface Detection {
  classId: number;
  label: string;
  confidence: number;
  boundingBox: BoundingBox;
}

export interface DetectResponse {
  modelId: string;
  count: number;
  detections: Detection[];
}

// Segment Types
export interface SegmentClass {
  classId: number;
  label: string;
  pixelCount: number;
  percentage: number;
}

export interface SegmentResponse {
  modelId: string;
  width: number;
  height: number;
  numClasses: number;
  topClasses: SegmentClass[];
}

// Translate Types
export interface TranslateRequest {
  modelId: string;
  text?: string;
  texts?: string[];
  sourceLanguage?: string;
  targetLanguage?: string;
}

export interface TranslationResult {
  index?: number;
  sourceText: string;
  translatedText: string;
  sourceLanguage: string;
  targetLanguage: string;
}

export interface TranslateResponse {
  modelId: string;
  sourceText?: string;
  translatedText?: string;
  sourceLanguage?: string;
  targetLanguage?: string;
  translations?: TranslationResult[];
}

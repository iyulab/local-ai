# LMSupply

**Local Model Supply for .NET — on-demand AI inference**

[![CI](https://github.com/iyulab/lm-supply/actions/workflows/ci.yml/badge.svg)](https://github.com/iyulab/lm-supply/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="images/1.png" width="49%" alt="LMSupply Console"/>
  <img src="images/2.png" width="49%" alt="LMSupply Console"/>
</p>
<p align="center">
  <img src="images/3.png" width="49%" alt="LMSupply Console"/>
  <img src="images/4.png" width="49%" alt="LMSupply Console"/>
</p>

> Start small. Download what you need. Run locally.

```csharp
// This is all you need. No setup. No configuration. No API keys.
await using var model = await LocalEmbedder.LoadAsync("auto");  // Hardware-optimized selection
float[] embedding = await model.EmbedAsync("Hello, world!");
```

LMSupply is designed around three core principles:

### 🪶 Minimal Footprint
Your application ships with **zero bundled models**. The base package is tiny. Models, tokenizers, and runtime components are downloaded **only when first requested** and cached for reuse.

### ⚡ Lazy Everything
```
First run:  LoadAsync("default") → Downloads model → Caches → Runs inference
Next runs:  LoadAsync("default") → Uses cached model → Runs inference instantly
```
No pre-download scripts. No model management. Just use it.

### 🎯 Zero Boilerplate
Traditional approach:
```csharp
// ❌ Without LMSupply: 50+ lines of setup
var tokenizer = LoadTokenizer(modelPath);
var session = new InferenceSession(modelPath, sessionOptions);
var inputIds = tokenizer.Encode(text);
var attentionMask = CreateAttentionMask(inputIds);
var inputs = new List<NamedOnnxValue> { ... };
var outputs = session.Run(inputs);
var embeddings = PostProcess(outputs);
// ... error handling, pooling, normalization, cleanup ...
```

```csharp
// ✅ With LMSupply: 2 lines
await using var model = await LocalEmbedder.LoadAsync("default");
float[] embedding = await model.EmbedAsync("Hello, world!");
```

---

## Packages

| Package | Description | Status |
|---------|-------------|--------|
| [LMSupply.Embedder](docs/embedder.md) | Text → Vector embeddings (ONNX + GGUF) | [![NuGet](https://img.shields.io/nuget/v/LMSupply.Embedder.svg)](https://www.nuget.org/packages/LMSupply.Embedder) |
| [LMSupply.Reranker](docs/reranker.md) | Semantic reranking for search | [![NuGet](https://img.shields.io/nuget/v/LMSupply.Reranker.svg)](https://www.nuget.org/packages/LMSupply.Reranker) |
| [LMSupply.Generator](docs/generator.md) | Text generation & chat (ONNX + GGUF) | [![NuGet](https://img.shields.io/nuget/v/LMSupply.Generator.svg)](https://www.nuget.org/packages/LMSupply.Generator) |
| [LMSupply.Captioner](docs/captioner.md) | Image → Text captioning | [![NuGet](https://img.shields.io/nuget/v/LMSupply.Captioner.svg)](https://www.nuget.org/packages/LMSupply.Captioner) |
| [LMSupply.Ocr](docs/ocr.md) | Document OCR | [![NuGet](https://img.shields.io/nuget/v/LMSupply.Ocr.svg)](https://www.nuget.org/packages/LMSupply.Ocr) |
| [LMSupply.Detector](docs/detector.md) | Object detection | [![NuGet](https://img.shields.io/nuget/v/LMSupply.Detector.svg)](https://www.nuget.org/packages/LMSupply.Detector) |
| [LMSupply.Segmenter](docs/segmenter.md) | Image segmentation | [![NuGet](https://img.shields.io/nuget/v/LMSupply.Segmenter.svg)](https://www.nuget.org/packages/LMSupply.Segmenter) |
| [LMSupply.Translator](docs/translator.md) | Neural machine translation | [![NuGet](https://img.shields.io/nuget/v/LMSupply.Translator.svg)](https://www.nuget.org/packages/LMSupply.Translator) |
| [LMSupply.Transcriber](docs/transcriber.md) | Speech → Text (Whisper) | [![NuGet](https://img.shields.io/nuget/v/LMSupply.Transcriber.svg)](https://www.nuget.org/packages/LMSupply.Transcriber) |
| [LMSupply.Synthesizer](docs/synthesizer.md) | Text → Speech (Piper) | [![NuGet](https://img.shields.io/nuget/v/LMSupply.Synthesizer.svg)](https://www.nuget.org/packages/LMSupply.Synthesizer) |
| [LMSupply.Llama](docs/llama.md) | Shared llama.cpp runtime for GGUF | [![NuGet](https://img.shields.io/nuget/v/LMSupply.Llama.svg)](https://www.nuget.org/packages/LMSupply.Llama) |

---

## Quick Start

### Text Embeddings

```csharp
using LMSupply.Embedder;

// Use "auto" for hardware-optimized model selection
await using var model = await LocalEmbedder.LoadAsync("auto");

// Single text
float[] embedding = await model.EmbedAsync("Hello, world!");

// Batch processing
float[][] embeddings = await model.EmbedAsync(new[]
{
    "First document",
    "Second document",
    "Third document"
});

// Similarity
float similarity = LocalEmbedder.CosineSimilarity(embeddings[0], embeddings[1]);

// GGUF models (via LLamaSharp) - Auto-detected by repo name pattern
await using var ggufModel = await LocalEmbedder.LoadAsync("nomic-ai/nomic-embed-text-v1.5-GGUF");
float[] ggufEmbedding = await ggufModel.EmbedAsync("Hello from GGUF!");
```

### Semantic Reranking

```csharp
using LMSupply.Reranker;

await using var reranker = await LocalReranker.LoadAsync("default");

var results = await reranker.RerankAsync(
    query: "What is machine learning?",
    documents: new[]
    {
        "Machine learning is a subset of artificial intelligence...",
        "The weather today is sunny and warm...",
        "Deep learning uses neural networks..."
    },
    topK: 2
);

foreach (var result in results)
{
    Console.WriteLine($"[{result.Score:F4}] {result.Document}");
}
```

### Text Generation

```csharp
using LMSupply.Generator;

// ONNX models (via GenAI)
var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()  // Phi-4-mini-instruct
    .BuildAsync();

string response = await generator.GenerateCompleteAsync("What is machine learning?");
Console.WriteLine(response);

// GGUF models (via LLamaSharp) - Access to thousands of quantized models
await using var model = await LocalGenerator.LoadAsync("gguf:default");  // Llama 3.2 3B

await foreach (var token in model.GenerateAsync("Hello, my name is"))
{
    Console.Write(token);
}

// Chat format with GGUF
var messages = new[]
{
    ChatMessage.System("You are a helpful assistant."),
    ChatMessage.User("Explain quantum computing simply.")
};

await foreach (var token in model.GenerateChatAsync(messages))
{
    Console.Write(token);
}
```

### Translation

```csharp
using LMSupply.Translator;

await using var translator = await LocalTranslator.LoadAsync("ko-en");

// Translate Korean to English
string english = await translator.TranslateAsync("안녕하세요, 세계!");
Console.WriteLine(english); // "Hello, world!"

// Batch translation
string[] translations = await translator.TranslateBatchAsync(new[]
{
    "첫 번째 문장입니다.",
    "두 번째 문장입니다."
});
```

### Speech Recognition (Transcriber)

```csharp
using LMSupply.Transcriber;

await using var transcriber = await LocalTranscriber.LoadAsync("default");

// Transcribe audio file
var result = await transcriber.TranscribeAsync("audio.wav");
Console.WriteLine(result.Text);
Console.WriteLine($"Language: {result.Language}");

// Streaming transcription
await foreach (var segment in transcriber.TranscribeStreamingAsync("audio.wav"))
{
    Console.WriteLine($"[{segment.Start:F2}s] {segment.Text}");
}
```

### Text-to-Speech (Synthesizer)

```csharp
using LMSupply.Synthesizer;

await using var synthesizer = await LocalSynthesizer.LoadAsync("default");

// Synthesize and save to file
await synthesizer.SynthesizeToFileAsync("Hello, world!", "output.wav");

// Get audio samples
var result = await synthesizer.SynthesizeAsync("Hello!");
Console.WriteLine($"Duration: {result.DurationSeconds:F2}s");
Console.WriteLine($"Real-time factor: {result.RealTimeFactor:F1}x");
```

---

## Available Models

*Updated: 2025-12 based on MTEB leaderboard and community benchmarks*

### Embedder (ONNX)

| Alias | Model | Dims | Params | Context | Best For |
|-------|-------|------|--------|---------|----------|
| `default` | bge-small-en-v1.5 | 384 | 33M | 512 | Balanced speed/quality |
| `fast` | all-MiniLM-L6-v2 | 384 | 22M | 256 | Ultra-low latency |
| `quality` | bge-base-en-v1.5 | 768 | 110M | 512 | Higher accuracy |
| `large` | nomic-embed-text-v1.5 | 768 | 137M | 8192 | Long context RAG |
| `multilingual` | multilingual-e5-base | 768 | 278M | 512 | 100+ languages |

### Embedder (GGUF via LLamaSharp)

GGUF models are auto-detected by `-GGUF` or `_gguf` in repo name, or `.gguf` file extension.

| Model Repository | Dims | Context | Best For |
|------------------|------|---------|----------|
| `nomic-ai/nomic-embed-text-v1.5-GGUF` | 768 | 8K | Long context, matryoshka |
| `BAAI/bge-small-en-v1.5-GGUF` | 384 | 512 | Compact and fast |
| `BAAI/bge-base-en-v1.5-GGUF` | 768 | 512 | Quality balance |
| Any HuggingFace GGUF embedding repo | varies | varies | Custom models |

### Reranker

| Alias | Model | Params | Context | Best For |
|-------|-------|--------|---------|----------|
| `default` | ms-marco-MiniLM-L-6-v2 | 22M | 512 | Balanced speed/quality |
| `fast` | ms-marco-TinyBERT-L-2-v2 | 4.4M | 512 | Ultra-low latency |
| `quality` | bge-reranker-base | 278M | 512 | Higher accuracy |
| `large` | bge-reranker-large | 560M | 512 | Best accuracy |
| `multilingual` | bge-reranker-v2-m3 | 568M | 8192 | Long docs, 100+ languages |

### Generator (ONNX)

| Alias | Model | Params | Context | License | Best For |
|-------|-------|--------|---------|---------|----------|
| `default` | Phi-4-mini-instruct | 3.8B | 16K | MIT | Balanced reasoning |
| `fast` | Llama-3.2-1B-Instruct | 1B | 8K | Llama 3.2 | Ultra-fast inference |
| `quality` | phi-4 | 14B | 16K | MIT | Best reasoning |
| `medium` | Phi-3.5-mini-instruct | 3.8B | 128K | MIT | Long context |
| `multilingual` | gemma-2-2b-it | 2B | 8K | Gemma ToU | Multi-language |

### Generator (GGUF via LLamaSharp)

| Alias | Model | Params | Context | Best For |
|-------|-------|--------|---------|----------|
| `gguf:default` | Llama 3.2 3B Instruct | 3B | 8K | Balanced quality/speed |
| `gguf:fast` | Llama 3.2 1B Instruct | 1B | 8K | Quick responses |
| `gguf:quality` | Qwen 2.5 7B Instruct | 7B | 32K | Higher quality |
| `gguf:large` | Qwen 2.5 14B Instruct | 14B | 32K | Best quality |
| `gguf:korean` | EXAONE 3.5 7.8B | 7.8B | 32K | Korean language |
| `gguf:code` | Qwen 2.5 Coder 7B | 7B | 32K | Coding tasks |
| `gguf:reasoning` | DeepSeek R1 Distill 8B | 8B | 32K | Complex reasoning |

### Translator

| Alias | Direction | Model | Best For |
|-------|-----------|-------|----------|
| `ko-en` | Korean → English | OPUS-MT | Korean translation |
| `en-ko` | English → Korean | OPUS-MT | Korean translation |
| `ja-en` | Japanese → English | OPUS-MT | Japanese translation |
| `zh-en` | Chinese → English | OPUS-MT | Chinese translation |
| `multilingual` | Many → English | mBART/M2M100 | 100+ languages |

### Transcriber (Whisper)

| Alias | Model | Params | Size | WER | Best For |
|-------|-------|--------|------|-----|----------|
| `fast` | Whisper Tiny | 39M | ~150MB | 7.6% | Ultra-fast transcription |
| `default` | Whisper Base | 74M | ~290MB | 5.0% | Balanced speed/quality |
| `quality` | Whisper Small | 244M | ~970MB | 3.4% | Higher accuracy |
| `large` | Whisper Large V3 | 1.5B | ~6GB | 2.5% | Best accuracy |
| `english` | Whisper Base.en | 74M | ~290MB | 4.3% | English-optimized |

### Synthesizer (Piper TTS)

| Alias | Voice | Language | Sample Rate | Best For |
|-------|-------|----------|-------------|----------|
| `default` | Lessac | en-US | 22050 Hz | Balanced quality |
| `fast` | Ryan | en-US | 16000 Hz | Ultra-fast synthesis |
| `quality` | Amy | en-US | 22050 Hz | High quality |
| `british` | Semaine | en-GB | 22050 Hz | British English |
| `korean` | KSS | ko-KR | 22050 Hz | Korean |
| `japanese` | JSUT | ja-JP | 22050 Hz | Japanese |
| `chinese` | Huayan | zh-CN | 22050 Hz | Mandarin Chinese |

---

## Adaptive Model Selection ("auto" mode)

Use `"auto"` to let LMSupply select the optimal model based on your hardware:

```csharp
// Hardware-optimized model selection
await using var embedder = await LocalEmbedder.LoadAsync("auto");
await using var generator = await LocalGenerator.LoadAsync("auto");
await using var reranker = await LocalReranker.LoadAsync("auto");
```

LMSupply detects your hardware and selects models accordingly:

| Performance Tier | Hardware | Embedder | Generator | Reranker |
|------------------|----------|----------|-----------|----------|
| **Low** | CPU only or GPU <4GB | bge-small (33M) | Llama-3.2-1B | MiniLM-L6 (22M) |
| **Medium** | GPU 4-8GB | bge-base (110M) | Phi-3.5-mini | bge-reranker-base |
| **High** | GPU 8-16GB | gte-large (434M) | Phi-4-mini | bge-reranker-large |
| **Ultra** | GPU 16GB+ | gte-large (434M) | Phi-4 (14B) | bge-reranker-large |

**Key benefits:**
- **Zero configuration** - Just use `"auto"`, no hardware research needed
- **Optimal performance** - Larger models on capable hardware
- **Graceful degradation** - Smaller models on limited hardware
- **Backward compatible** - Existing aliases (`"default"`, `"fast"`, `"quality"`) still work

---

## GPU Acceleration

GPU acceleration is **automatic** — LMSupply detects your hardware and downloads appropriate runtime binaries on first use:

```
Detection priority: CUDA → DirectML → CoreML → CPU
```

```csharp
// Auto-detect (default) - uses GPU if available, falls back to CPU
var options = new EmbedderOptions { Provider = ExecutionProvider.Auto };

// Force specific provider
var options = new EmbedderOptions { Provider = ExecutionProvider.Cuda };     // NVIDIA
var options = new EmbedderOptions { Provider = ExecutionProvider.DirectML }; // Windows GPU
var options = new EmbedderOptions { Provider = ExecutionProvider.CoreML };   // macOS
```

### Verify GPU Detection

```csharp
using LMSupply.Runtime;

// Quick summary (returns formatted string)
Console.WriteLine(EnvironmentDetector.GetEnvironmentSummary());

// Or access individual properties
var gpu = EnvironmentDetector.DetectGpu();
var provider = EnvironmentDetector.GetRecommendedProvider();

Console.WriteLine($"Provider: {provider}");
Console.WriteLine($"CUDA Available: {gpu.Vendor == GpuVendor.Nvidia && gpu.CudaDriverVersionMajor >= 11}");
Console.WriteLine($"DirectML Available: {gpu.DirectMLSupported}");
```

### Troubleshooting GPU Issues

**Do NOT install ONNX Runtime packages manually.** LMSupply handles runtime binary management automatically via lazy downloading.

If you have conflicting packages installed, remove them:

```bash
dotnet remove package Microsoft.ML.OnnxRuntime
dotnet remove package Microsoft.ML.OnnxRuntime.Gpu
dotnet remove package Microsoft.ML.OnnxRuntime.DirectML
```

For NVIDIA CUDA support, ensure you have:
- NVIDIA GPU drivers installed
- CUDA 11.x or 12.x runtime (LMSupply auto-selects the appropriate version)

---

## Thread Safety & Batch Processing

All LMSupply models are **thread-safe** for concurrent inference. ONNX Runtime's `InferenceSession.Run()` is thread-safe by design.

```csharp
// Safe: Concurrent inference on the same model instance
await using var embedder = await LocalEmbedder.LoadAsync("default");

await Parallel.ForEachAsync(documents, async (doc, ct) =>
{
    var embedding = await embedder.EmbedAsync(doc, ct);
    // Process embedding...
});

// Or with Task.WhenAll
var tasks = documents.Select(d => embedder.EmbedAsync(d));
var embeddings = await Task.WhenAll(tasks);
```

**Performance tips:**
- GPU inference: 2-4 concurrent operations typically optimal
- CPU inference: Match `MaxDegreeOfParallelism` to core count
- Use `EmbedBatchAsync()` when available for better throughput

---

## Custom Models

LMSupply supports any HuggingFace repository with ONNX models through automatic file discovery:

```csharp
// Use any HuggingFace ONNX model repository
await using var embedder = await LocalEmbedder.LoadAsync("my-org/my-custom-embedder");
await using var captioner = await LocalCaptioner.LoadAsync("my-org/my-vision-model");
var generator = await LocalGenerator.LoadAsync("my-org/my-llm-onnx");
```

The system automatically:
- Discovers ONNX files via HuggingFace API
- Detects subfolder structure (`onnx/`, `cpu/`, `cuda/`)
- Selects appropriate quantization variants
- Downloads required tokenizer and config files

For private repositories, set the `HF_TOKEN` environment variable.

---

## Model Caching

Models are cached following HuggingFace Hub conventions:

- **Default**: `~/.cache/huggingface/hub`
- **Environment variables**: `HF_HUB_CACHE`, `HF_HOME`, or `XDG_CACHE_HOME`
- **Manual override**: `new EmbedderOptions { CacheDirectory = "/path/to/cache" }`

---

## Requirements

### Software
- .NET 10.0+
- Windows 10+, Linux, or macOS 11+

### Hardware (Recommended)

| Use Case | RAM | GPU VRAM | Notes |
|----------|-----|----------|-------|
| **Embeddings** | 4GB+ | Optional | CPU works fine for small models |
| **Reranking** | 8GB+ | 4GB+ | GPU recommended for large models |
| **Text Generation** | 16GB+ | 8GB+ | VRAM strongly recommended |
| **Speech (Whisper)** | 8GB+ | 4GB+ | GPU significantly faster |
| **Vision (Detection/Captioning)** | 8GB+ | 4GB+ | GPU recommended |

**Minimum for "auto" mode:**
- Any modern CPU with 8GB RAM
- For best experience: NVIDIA GPU with 8GB+ VRAM

---

## Documentation

### Getting Started
- [Model Lifecycle](docs/MODEL_LIFECYCLE.md) - Loading, using, and disposing models
- [GPU Providers](docs/GPU_PROVIDERS.md) - GPU acceleration and provider selection
- [Memory Requirements](docs/MEMORY_REQUIREMENTS.md) - Model memory requirements and OOM prevention
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

### Text & Language
- [Embedder Guide](docs/embedder.md) - Text → Vector embeddings
- [Reranker Guide](docs/reranker.md) - Semantic reranking
- [Generator Guide](docs/generator.md) - Text generation & chat
- [Translator Guide](docs/translator.md) - Neural machine translation

### Vision
- [Captioner Guide](docs/captioner.md) - Image → Text captioning
- [OCR Guide](docs/ocr.md) - Document text recognition
- [Detector Guide](docs/detector.md) - Object detection
- [Segmenter Guide](docs/segmenter.md) - Image segmentation

### Audio
- [Transcriber Guide](docs/transcriber.md) - Speech → Text (Whisper)
- [Synthesizer Guide](docs/synthesizer.md) - Text → Speech (Piper)

---

## License

MIT License - see [LICENSE](LICENSE) for details.
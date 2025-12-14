# LocalAI

[![CI](https://github.com/iyulab/local-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/iyulab/local-ai/actions/workflows/ci.yml)
[![Embedder](https://img.shields.io/nuget/v/LocalAI.Embedder.svg)](https://www.nuget.org/packages/LocalAI.Embedder)
[![Reranker](https://img.shields.io/nuget/v/LocalAI.Reranker.svg)](https://www.nuget.org/packages/LocalAI.Reranker)
[![Generator](https://img.shields.io/nuget/v/LocalAI.Generator.svg)](https://www.nuget.org/packages/LocalAI.Generator)
[![Captioner](https://img.shields.io/nuget/v/LocalAI.Captioner.svg)](https://www.nuget.org/packages/LocalAI.Captioner)
[![Ocr](https://img.shields.io/nuget/v/LocalAI.Ocr.svg)](https://www.nuget.org/packages/LocalAI.Ocr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Philosophy

**Start small. Download what you need. Run locally.**

```csharp
// This is all you need. No setup. No configuration. No API keys.
await using var model = await LocalEmbedder.LoadAsync("default");
float[] embedding = await model.EmbedAsync("Hello, world!");
```

LocalAI is designed around three core principles:

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
// ❌ Without LocalAI: 50+ lines of setup
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
// ✅ With LocalAI: 2 lines
await using var model = await LocalEmbedder.LoadAsync("default");
float[] embedding = await model.EmbedAsync("Hello, world!");
```

---

## Packages

| Package | Description | Status |
|---------|-------------|--------|
| [LocalAI.Embedder](docs/embedder.md) | Text → Vector embeddings | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Embedder.svg)](https://www.nuget.org/packages/LocalAI.Embedder) |
| [LocalAI.Reranker](docs/reranker.md) | Semantic reranking for search | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Reranker.svg)](https://www.nuget.org/packages/LocalAI.Reranker) |
| [LocalAI.Generator](docs/generator.md) | Text generation & chat | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Generator.svg)](https://www.nuget.org/packages/LocalAI.Generator) |
| [LocalAI.Captioner](docs/captioner.md) | Image → Text captioning | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Captioner.svg)](https://www.nuget.org/packages/LocalAI.Captioner) |
| [LocalAI.Ocr](docs/ocr.md) | Document OCR | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Ocr.svg)](https://www.nuget.org/packages/LocalAI.Ocr) |
| LocalAI.Detector | Object detection | 📋 Planned |
| LocalAI.Segmenter | Image segmentation | 📋 Planned |
| LocalAI.Translator | Neural machine translation | 📋 Planned |
| LocalAI.Transcriber | Speech → Text (Whisper) | 📋 Planned |
| LocalAI.Synthesizer | Text → Speech | 📋 Planned |

---

## Quick Start

### Text Embeddings

```csharp
using LocalAI.Embedder;

await using var model = await LocalEmbedder.LoadAsync("default");

// Single text
float[] embedding = await model.EmbedAsync("Hello, world!");

// Batch processing
float[][] embeddings = await model.EmbedBatchAsync(new[]
{
    "First document",
    "Second document",
    "Third document"
});

// Similarity
float similarity = model.CosineSimilarity(embeddings[0], embeddings[1]);
```

### Semantic Reranking

```csharp
using LocalAI.Reranker;

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
using LocalAI.Generator;

// Simple generation
var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()  // Phi-3.5 Mini
    .BuildAsync();

string response = await generator.GenerateCompleteAsync("What is machine learning?");
Console.WriteLine(response);

// Chat format
var messages = new[]
{
    new ChatMessage(ChatRole.System, "You are a helpful assistant."),
    new ChatMessage(ChatRole.User, "Explain quantum computing simply.")
};

string chatResponse = await generator.GenerateChatCompleteAsync(messages);

// Streaming
await foreach (var token in generator.GenerateAsync("Write a story:"))
{
    Console.Write(token);
}
```

---

## Available Models

*Updated: 2025-01 based on MTEB leaderboard and community benchmarks*

### Embedder

| Alias | Model | Dims | Params | Context | Best For |
|-------|-------|------|--------|---------|----------|
| `default` | bge-small-en-v1.5 | 384 | 33M | 512 | Balanced speed/quality |
| `fast` | all-MiniLM-L6-v2 | 384 | 22M | 256 | Ultra-low latency |
| `quality` | bge-base-en-v1.5 | 768 | 110M | 512 | Higher accuracy |
| `large` | nomic-embed-text-v1.5 | 768 | 137M | 8192 | Long context RAG |
| `multilingual` | multilingual-e5-base | 768 | 278M | 512 | 100+ languages |

### Reranker

| Alias | Model | Params | Context | Best For |
|-------|-------|--------|---------|----------|
| `default` | ms-marco-MiniLM-L-6-v2 | 22M | 512 | Balanced speed/quality |
| `fast` | ms-marco-TinyBERT-L-2-v2 | 4.4M | 512 | Ultra-low latency |
| `quality` | bge-reranker-base | 278M | 512 | Higher accuracy |
| `large` | bge-reranker-large | 560M | 512 | Best accuracy |
| `multilingual` | bge-reranker-v2-m3 | 568M | 8192 | Long docs, 100+ languages |

### Generator

| Alias | Model | Params | Context | License | Best For |
|-------|-------|--------|---------|---------|----------|
| `default` | Phi-4-mini-instruct | 3.8B | 16K | MIT | Balanced reasoning |
| `fast` | Llama-3.2-1B-Instruct | 1B | 8K | Llama 3.2 | Ultra-fast inference |
| `quality` | phi-4 | 14B | 16K | MIT | Best reasoning |
| `medium` | Phi-3.5-mini-instruct | 3.8B | 128K | MIT | Long context |
| `multilingual` | gemma-2-2b-it | 2B | 8K | Gemma ToU | Multi-language |

---

## GPU Acceleration

GPU acceleration is automatic when detected:

```csharp
// Auto-detect (default) - uses GPU if available, falls back to CPU
var options = new EmbedderOptions { Provider = ExecutionProvider.Auto };

// Force specific provider
var options = new EmbedderOptions { Provider = ExecutionProvider.Cuda };     // NVIDIA
var options = new EmbedderOptions { Provider = ExecutionProvider.DirectML }; // Windows GPU
var options = new EmbedderOptions { Provider = ExecutionProvider.CoreML };   // macOS
```

Install the appropriate ONNX Runtime package for GPU support:

```bash
dotnet add package Microsoft.ML.OnnxRuntime.Gpu       # NVIDIA CUDA
dotnet add package Microsoft.ML.OnnxRuntime.DirectML  # Windows (AMD, Intel, NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.CoreML    # macOS
```

---

## Model Caching

Models are cached following HuggingFace Hub conventions:

- **Default**: `~/.cache/huggingface/hub`
- **Environment variables**: `HF_HUB_CACHE`, `HF_HOME`, or `XDG_CACHE_HOME`
- **Manual override**: `new EmbedderOptions { CacheDirectory = "/path/to/cache" }`

---

## Requirements

- .NET 10.0+
- Windows, Linux, or macOS

---

## Documentation

- [Embedder Guide](docs/embedder.md)
- [Reranker Guide](docs/reranker.md)
- [Generator Guide](docs/generator.md)
- [Captioner Guide](docs/captioner.md)
- [OCR Guide](docs/ocr.md)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Release Process

Releases are automated via GitHub Actions when `Directory.Build.props` is updated:

1. Update the `<Version>` in `Directory.Build.props`
2. Commit and push to main
3. CI automatically publishes all packages to NuGet and creates a GitHub release

Requires `NUGET_API_KEY` secret configured in GitHub repository settings.

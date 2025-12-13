# LocalAI

[![CI](https://github.com/iyulab/local-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/iyulab/local-ai/actions/workflows/ci.yml)
[![NuGet](https://img.shields.io/nuget/v/LocalAI.Embedder.svg)](https://www.nuget.org/packages/LocalAI.Embedder)
[![NuGet](https://img.shields.io/nuget/v/LocalAI.Reranker.svg)](https://www.nuget.org/packages/LocalAI.Reranker)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

.NET libraries for on-device AI inference with zero external API dependencies. Run embeddings, reranking, and more entirely on your local machine with automatic GPU acceleration.

## Features

- **Zero Configuration**: Works out of the box with sensible defaults
- **Automatic GPU Acceleration**: Detects and uses CUDA, DirectML, or CoreML automatically
- **HuggingFace Compatible**: Downloads models from HuggingFace Hub with standard caching
- **Cross-Platform**: Windows, Linux, macOS support
- **Production Ready**: Thread-safe, async-first, IAsyncDisposable support

## Packages

| Package | Description | NuGet |
|---------|-------------|-------|
| [LocalAI.Embedder](docs/embedder.md) | Text embeddings with sentence-transformers models | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Embedder.svg)](https://www.nuget.org/packages/LocalAI.Embedder) |
| [LocalAI.Reranker](docs/reranker.md) | Semantic reranking with cross-encoder models | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Reranker.svg)](https://www.nuget.org/packages/LocalAI.Reranker) |

## Quick Start

### Text Embeddings

```csharp
using LocalAI.Embedder;

// Load a model (downloads automatically on first use)
await using var model = await LocalEmbedder.LoadAsync("default");

// Generate embeddings
float[] embedding = await model.EmbedAsync("Hello, world!");

// Batch processing
float[][] embeddings = await model.EmbedBatchAsync(new[]
{
    "First document",
    "Second document",
    "Third document"
});

// Calculate similarity
float similarity = model.CosineSimilarity(embeddings[0], embeddings[1]);
```

### Semantic Reranking

```csharp
using LocalAI.Reranker;

// Load a reranker model
await using var reranker = await LocalReranker.LoadAsync("default");

// Rerank documents by relevance to a query
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

## Available Models

### Embedder Models

| Alias | Model | Dimensions | Size |
|-------|-------|------------|------|
| `default` | all-MiniLM-L6-v2 | 384 | ~90MB |
| `large` | all-mpnet-base-v2 | 768 | ~420MB |
| `multilingual` | paraphrase-multilingual-MiniLM-L12-v2 | 384 | ~470MB |

### Reranker Models

| Alias | Model | Max Tokens | Size |
|-------|-------|------------|------|
| `default` | ms-marco-MiniLM-L-6-v2 | 512 | ~90MB |
| `quality` | ms-marco-MiniLM-L-12-v2 | 512 | ~134MB |
| `fast` | ms-marco-TinyBERT-L-2-v2 | 512 | ~18MB |
| `multilingual` | bge-reranker-v2-m3 | 8192 | ~1.1GB |

## GPU Acceleration

GPU acceleration is automatic when available:

```csharp
// Auto-detect best provider (default)
var options = new EmbedderOptions { Provider = ExecutionProvider.Auto };

// Force specific provider
var options = new EmbedderOptions { Provider = ExecutionProvider.Cuda };
var options = new EmbedderOptions { Provider = ExecutionProvider.DirectML }; // Windows
var options = new EmbedderOptions { Provider = ExecutionProvider.CoreML };   // macOS
var options = new EmbedderOptions { Provider = ExecutionProvider.Cpu };
```

For GPU support, install the appropriate ONNX Runtime package:

```bash
# NVIDIA CUDA
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows DirectML (AMD, Intel, NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML

# macOS CoreML
dotnet add package Microsoft.ML.OnnxRuntime.CoreML
```

## Model Caching

Models are cached following HuggingFace Hub standard:

- Default: `~/.cache/huggingface/hub`
- Override with `HF_HUB_CACHE`, `HF_HOME`, or `XDG_CACHE_HOME` environment variables
- Or specify directly: `new EmbedderOptions { CacheDirectory = "/path/to/cache" }`

## Requirements

- .NET 8.0 or later
- Windows, Linux, or macOS

## Documentation

- [Embedder Guide](docs/embedder.md)
- [Reranker Guide](docs/reranker.md)
- [API Reference](docs/api-reference.md)
- [Performance Tips](docs/performance.md)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Release Process

Releases are automated via GitHub Actions:

```bash
# Release all packages with same version
git tag v0.5.0
git push origin v0.5.0

# Release individual packages
git tag core-v0.1.1
git tag embedder-v0.4.1
git tag reranker-v0.2.1
git push origin --tags
```

Requires `NUGET_API_KEY` secret configured in GitHub repository settings.

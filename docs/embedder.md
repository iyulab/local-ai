# LMSupply.Embedder

A lightweight, zero-configuration text embedding library for .NET with automatic GPU acceleration.

Supports both **ONNX** models (sentence-transformers) and **GGUF** models (via LLamaSharp).

## Installation

```bash
dotnet add package LMSupply.Embedder
```

GPU acceleration is **automatic** — LMSupply detects your hardware and downloads appropriate runtime binaries on first use. No additional packages required.

## Basic Usage

```csharp
using LMSupply.Embedder;

// Load the default model (ONNX)
await using var model = await LocalEmbedder.LoadAsync("default");

// Generate a single embedding
float[] embedding = await model.EmbedAsync("Hello, world!");
Console.WriteLine($"Embedding dimensions: {embedding.Length}");

// Generate multiple embeddings
float[][] embeddings = await model.EmbedAsync(new[]
{
    "First document",
    "Second document"
});
```

## Available Models (ONNX)

| Alias | Model | Dimensions | Params | Context | Best For |
|-------|-------|------------|--------|---------|----------|
| `default` | bge-small-en-v1.5 | 384 | 33M | 512 | Balanced speed/quality |
| `fast` | all-MiniLM-L6-v2 | 384 | 22M | 256 | Ultra-low latency |
| `quality` | bge-base-en-v1.5 | 768 | 110M | 512 | Higher accuracy |
| `large` | nomic-embed-text-v1.5 | 768 | 137M | 8192 | Long context RAG |
| `multilingual` | multilingual-e5-base | 768 | 278M | 512 | 100+ languages |

You can also use any HuggingFace ONNX embedding model by its full ID:

```csharp
var model = await LocalEmbedder.LoadAsync("sentence-transformers/all-MiniLM-L12-v2");
```

## GGUF Models (via LLamaSharp)

GGUF embedding models are auto-detected by repo name patterns (`-GGUF`, `_gguf`) or `.gguf` extension.

```csharp
using LMSupply.Embedder;

// Load GGUF model (auto-detected by "-GGUF" in repo name)
await using var model = await LocalEmbedder.LoadAsync("nomic-ai/nomic-embed-text-v1.5-GGUF");

// Usage is identical to ONNX models
float[] embedding = await model.EmbedAsync("Hello from GGUF!");

// Batch processing
float[][] embeddings = await model.EmbedAsync(new[]
{
    "First document",
    "Second document"
});
```

### Available GGUF Embedding Models

| Model Repository | Dims | Context | Best For |
|------------------|------|---------|----------|
| `nomic-ai/nomic-embed-text-v1.5-GGUF` | 768 | 8K | Long context, matryoshka |
| `BAAI/bge-small-en-v1.5-GGUF` | 384 | 512 | Compact and fast |
| `BAAI/bge-base-en-v1.5-GGUF` | 768 | 512 | Quality balance |

You can also use local GGUF files:

```csharp
await using var model = await LocalEmbedder.LoadAsync("/path/to/embedding-model.gguf");
```

## Multilingual Support

For non-English text, use the `multilingual` model which supports 100+ languages:

```csharp
// Load multilingual model
await using var model = await LocalEmbedder.LoadAsync("multilingual");

// Korean text embedding
float[] koreanEmbedding = await model.EmbedAsync("안녕하세요, 세계!");

// Japanese text embedding
float[] japaneseEmbedding = await model.EmbedAsync("こんにちは、世界！");

// Chinese text embedding
float[] chineseEmbedding = await model.EmbedAsync("你好，世界！");

// Cross-lingual similarity works!
float similarity = LocalEmbedder.CosineSimilarity(koreanEmbedding, japaneseEmbedding);
```

## Configuration Options

```csharp
var options = new EmbedderOptions
{
    // GPU/CPU execution provider
    Provider = ExecutionProvider.Auto,  // Auto, Cpu, Cuda, DirectML, CoreML

    // Maximum sequence length (tokens)
    MaxSequenceLength = 512,

    // Normalize embeddings to unit length
    NormalizeEmbeddings = true,

    // Pooling strategy (ONNX models only)
    PoolingMode = PoolingMode.Mean,  // Mean, Cls, Max

    // Lowercase input text (for uncased models)
    DoLowerCase = true,

    // Custom cache directory
    CacheDirectory = null  // Uses ~/.cache/huggingface/hub by default
};

var model = await LocalEmbedder.LoadAsync("default", options);
```

## Similarity Calculation

```csharp
// Cosine similarity (for normalized embeddings)
float similarity = LocalEmbedder.CosineSimilarity(embedding1, embedding2);

// Dot product
float dotProduct = LocalEmbedder.DotProduct(embedding1, embedding2);

// Euclidean distance
float distance = LocalEmbedder.EuclideanDistance(embedding1, embedding2);
```

## Semantic Search Example

```csharp
using LMSupply.Embedder;

await using var model = await LocalEmbedder.LoadAsync("default");

// Index documents
var documents = new[]
{
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language",
    "Neural networks are inspired by biological neurons"
};

float[][] docEmbeddings = await model.EmbedAsync(documents);

// Search
string query = "What is AI?";
float[] queryEmbedding = await model.EmbedAsync(query);

// Find most similar documents
var results = documents
    .Select((doc, i) => new
    {
        Document = doc,
        Score = LocalEmbedder.CosineSimilarity(queryEmbedding, docEmbeddings[i])
    })
    .OrderByDescending(x => x.Score)
    .Take(3);

foreach (var result in results)
{
    Console.WriteLine($"[{result.Score:F4}] {result.Document}");
}
```

## Clustering Example

```csharp
using LMSupply.Embedder;

await using var model = await LocalEmbedder.LoadAsync("default");

var texts = new[]
{
    // Technology
    "Artificial intelligence is changing industries.",
    "Machine learning models improve with more data.",
    "Cloud computing enables scalable applications.",
    // Nature
    "The forest is home to many species of birds.",
    "Rivers flow from mountains to the sea.",
    // Food
    "Italian pasta is often served with tomato sauce.",
    "Sushi is a traditional Japanese dish."
};

var embeddings = await model.EmbedAsync(texts);

// Find texts with similarity > 0.5
for (int i = 0; i < texts.Length; i++)
{
    for (int j = i + 1; j < texts.Length; j++)
    {
        var sim = LocalEmbedder.CosineSimilarity(embeddings[i], embeddings[j]);
        if (sim > 0.5)
        {
            Console.WriteLine($"Similar: \"{texts[i]}\" <-> \"{texts[j]}\" ({sim:F3})");
        }
    }
}
```

## Pooling Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `Mean` | Average of all token embeddings | General purpose (default) |
| `Cls` | Use [CLS] token embedding | Classification tasks |
| `Max` | Max pooling across tokens | Capturing key features |

```csharp
var meanOptions = new EmbedderOptions { PoolingMode = PoolingMode.Mean };
var clsOptions = new EmbedderOptions { PoolingMode = PoolingMode.Cls };
var maxOptions = new EmbedderOptions { PoolingMode = PoolingMode.Max };
```

## Thread Safety

The `IEmbeddingModel` instance is thread-safe and can be shared across multiple threads:

```csharp
await using var model = await LocalEmbedder.LoadAsync("default");

// Safe to use concurrently
await Parallel.ForEachAsync(documents, async (doc, ct) =>
{
    var embedding = await model.EmbedAsync(doc, ct);
    // Process embedding...
});
```

## Warmup

To avoid cold-start latency on first inference:

```csharp
await using var model = await LocalEmbedder.LoadAsync("default");
await model.WarmupAsync();  // Pre-loads the model
```

## Model Information

```csharp
var info = model.GetModelInfo();
if (info != null)
{
    Console.WriteLine($"Model: {info.RepoId}");
    Console.WriteLine($"Dimensions: {info.Dimensions}");
    Console.WriteLine($"Max Tokens: {info.MaxSequenceLength}");
}
```

## Local Models

You can use locally stored ONNX models:

```csharp
var model = await LocalEmbedder.LoadAsync("/path/to/model.onnx");
```

The directory should contain:
- `model.onnx` - The ONNX model file
- `vocab.txt` - Vocabulary file

## Performance Tips

1. **Reuse the model instance** - Creating a new instance loads the model from disk
2. **Use batch processing** - `EmbedAsync(string[])` is more efficient than multiple `EmbedAsync` calls
3. **Enable GPU acceleration** - LMSupply automatically uses GPU when available
4. **Warmup before production** - Call `WarmupAsync()` to avoid cold-start latency
5. **Choose the right model** - Use `fast` for latency-sensitive, `quality` for accuracy

## GGUF vs ONNX

| Feature | ONNX | GGUF |
|---------|------|------|
| Format | ONNX Runtime | llama.cpp |
| GPU Support | CUDA, DirectML, CoreML | CUDA, Metal, Vulkan |
| Quantization | FP32/FP16/INT8 | Q4/Q5/Q8/F16 |
| Model Sources | HuggingFace ONNX repos | HuggingFace GGUF repos |
| Best For | Standard transformers | Long context, quantized models |

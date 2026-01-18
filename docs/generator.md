# LMSupply.Generator

Local text generation and chat with ONNX Runtime GenAI and GGUF (LLamaSharp) support.

## Installation

```bash
dotnet add package LMSupply.Generator
```

## Quick Start

### Simple Text Generation

```csharp
using LMSupply.Generator;

// Using the builder pattern
var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()        // Uses Phi-3.5 Mini
    .BuildAsync();

// Generate text
string response = await generator.GenerateCompleteAsync("What is machine learning?");
Console.WriteLine(response);

await generator.DisposeAsync();
```

### Chat Completion

```csharp
using LMSupply.Generator;
using LMSupply.Generator.Models;

var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()
    .BuildAsync();

// Chat format
var messages = new[]
{
    new ChatMessage(ChatRole.System, "You are a helpful assistant."),
    new ChatMessage(ChatRole.User, "Explain quantum computing in simple terms.")
};

string response = await generator.GenerateChatCompleteAsync(messages);
Console.WriteLine(response);
```

### Streaming Generation

```csharp
await foreach (var token in generator.GenerateAsync("Write a short story about a robot:"))
{
    Console.Write(token);
}
```

## Model Selection

### Preset Models

```csharp
// Default: Phi-3.5 Mini (balanced, MIT license)
.WithDefaultModel()

// Or use presets
.WithModel(GeneratorModelPreset.Default)   // Phi-3.5 Mini
.WithModel(GeneratorModelPreset.Fast)      // Llama 3.2 1B
.WithModel(GeneratorModelPreset.Quality)   // Phi-4
.WithModel(GeneratorModelPreset.Small)     // Llama 3.2 1B
```

### HuggingFace Models

```csharp
// Use any ONNX model from HuggingFace
.WithHuggingFaceModel("microsoft/Phi-3.5-mini-instruct-onnx")
.WithHuggingFaceModel("onnx-community/Llama-3.2-1B-Instruct-ONNX")
```

### Local Models

```csharp
// Use a local model directory
.WithModelPath("C:/models/my-model-onnx")
```

## Configuration Options

### Execution Provider

```csharp
var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()
    .WithProvider(ExecutionProvider.Auto)      // Auto-detect best provider
    .WithProvider(ExecutionProvider.Cuda)      // NVIDIA GPU
    .WithProvider(ExecutionProvider.DirectML)  // Windows GPU (AMD, Intel, NVIDIA)
    .WithProvider(ExecutionProvider.CoreML)    // macOS Apple Silicon
    .WithProvider(ExecutionProvider.Cpu)       // CPU only
    .BuildAsync();
```

### Generation Options

```csharp
var options = new GeneratorOptions
{
    MaxTokens = 512,              // Maximum tokens to generate
    Temperature = 0.7f,           // Randomness (0.0 = deterministic)
    TopP = 0.9f,                  // Nucleus sampling
    TopK = 50,                    // Top-K sampling
    RepetitionPenalty = 1.1f,     // Discourage repetition
    DoSample = true               // Enable sampling (vs greedy)
};

string response = await generator.GenerateCompleteAsync(prompt, options);

// Or use presets
string creative = await generator.GenerateCompleteAsync(prompt, GeneratorOptions.Creative);
string precise = await generator.GenerateCompleteAsync(prompt, GeneratorOptions.Precise);
```

### Memory Management

```csharp
// Limit memory usage
var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()
    .WithMemoryLimit(8.0)    // 8GB limit
    .BuildAsync();

// Or with detailed options
var memoryOptions = new MemoryAwareOptions
{
    MaxMemoryBytes = 8L * 1024 * 1024 * 1024,  // 8GB
    WarningThreshold = 0.80,                    // GC at 80%
    CriticalThreshold = 0.95,                   // Fail at 95%
    AutoGcOnWarning = true
};

var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()
    .WithMemoryManagement(memoryOptions)
    .BuildAsync();
```

## Hardware Detection

```csharp
using LMSupply.Generator;

// Get hardware recommendations
var recommendation = HardwareDetector.GetRecommendation();

Console.WriteLine(recommendation.GetSummary());
// Output:
// Hardware: NVIDIA RTX 4090 (24.0GB)
// System Memory: 64.0GB
// Provider: Cuda
// Quantization: FP16
// Max Context: 16384
// Recommended Models: microsoft/Phi-3.5-mini-instruct-onnx, microsoft/phi-4-onnx

// Auto-select best provider
var provider = HardwareDetector.GetBestProvider();
```

## Speculative Decoding

Speed up generation by using a smaller draft model:

```csharp
using LMSupply.Generator;

// Create draft (small/fast) and target (large/accurate) models
var draftModel = await TextGeneratorBuilder.Create()
    .WithModel(GeneratorModelPreset.Fast)
    .BuildAsync();

var targetModel = await TextGeneratorBuilder.Create()
    .WithModel(GeneratorModelPreset.Quality)
    .BuildAsync();

// Create speculative decoder
var decoder = SpeculativeDecoderBuilder.Create()
    .WithDraftModel(draftModel)
    .WithTargetModel(targetModel)
    .WithSpeculationLength(5)
    .WithAdaptiveSpeculation(true)
    .Build();

// Generate with speculative decoding
var result = await decoder.GenerateCompleteAsync("Explain neural networks:");

Console.WriteLine(result.Text);
Console.WriteLine(result.Stats.GetSummary());
// Output:
// Total Tokens: 256
// Draft/Accepted: 200/180 (90.0%)
// Target Tokens: 76
// Throughput: 45.2 tok/s
// Time: 5667ms
```

## Model Factory

For advanced scenarios with multiple models:

```csharp
using LMSupply.Generator;

using var factory = new OnnxGeneratorModelFactory();

// Check if model is available locally
if (!factory.IsModelAvailable("microsoft/Phi-3.5-mini-instruct-onnx"))
{
    // Download model
    await factory.DownloadModelAsync(
        "microsoft/Phi-3.5-mini-instruct-onnx",
        progress: new Progress<double>(p => Console.WriteLine($"Downloading: {p:P0}"))
    );
}

// Create model instance
var model = await factory.CreateAsync("microsoft/Phi-3.5-mini-instruct-onnx");

// List available models
foreach (var modelId in factory.GetAvailableModels())
{
    Console.WriteLine(modelId);
}
```

## Well-Known Models

| Alias | Model | Parameters | License |
|-------|-------|------------|---------|
| Default | Phi-3.5-mini-instruct | 3.8B | MIT |
| Fast | Llama-3.2-1B-Instruct | 1B | Llama 3.2 |
| Quality | phi-4 | 14B | MIT |
| Small | Llama-3.2-1B-Instruct | 1B | Llama 3.2 |

## Chat Formats

The library automatically detects chat format based on model ID:

| Format | Models |
|--------|--------|
| Phi-3 | Phi-3, Phi-3.5, Phi-4 |
| Llama 3 | Llama-3, Llama-3.1, Llama-3.2 |
| ChatML | Most other models |
| Gemma | Gemma, Gemma-2 |

Or specify explicitly:

```csharp
var generator = await TextGeneratorBuilder.Create()
    .WithHuggingFaceModel("my-model")
    .WithChatFormat("phi3")   // phi3, llama3, chatml, gemma
    .BuildAsync();
```

## GPU Support

Install the appropriate ONNX Runtime package:

```bash
# NVIDIA CUDA
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows (AMD, Intel, NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML

# macOS Apple Silicon
dotnet add package Microsoft.ML.OnnxRuntime.CoreML
```

## GGUF Model Support

GGUF models are loaded via [LLamaSharp](https://github.com/SciSharp/LLamaSharp), providing access to the vast ecosystem of quantized models on HuggingFace.

### Quick Start with GGUF

```csharp
using LMSupply.Generator;

// Load a GGUF model using the "gguf:" prefix
await using var model = await LocalGenerator.LoadAsync("gguf:default");

// Generate text
await foreach (var token in model.GenerateAsync("Hello, my name is"))
{
    Console.Write(token);
}
```

### GGUF Model Aliases

| Alias | Model | Parameters | Use Case |
|-------|-------|------------|----------|
| `gguf:default` | Llama 3.2 3B Instruct | 3B | Balanced quality/speed |
| `gguf:fast` | Llama 3.2 1B Instruct | 1B | Quick responses |
| `gguf:quality` | Qwen 2.5 7B Instruct | 7B | Higher quality |
| `gguf:large` | Qwen 2.5 14B Instruct | 14B | Best quality |
| `gguf:multilingual` | Gemma 2 9B | 9B | Non-English tasks |
| `gguf:korean` | EXAONE 3.5 7.8B | 7.8B | Korean language |
| `gguf:code` | Qwen 2.5 Coder 7B | 7B | Coding tasks |
| `gguf:reasoning` | DeepSeek R1 Distill 8B | 8B | Complex reasoning |

### Using HuggingFace GGUF Repositories

```csharp
// Load from any GGUF repository
await using var model = await LocalGenerator.LoadAsync(
    "bartowski/Llama-3.2-3B-Instruct-GGUF");

// Specify a particular quantization
await using var model = await LocalGenerator.LoadAsync(
    "bartowski/Qwen2.5-7B-Instruct-GGUF",
    new GeneratorOptions { GgufFileName = "Qwen2.5-7B-Instruct-Q4_K_M.gguf" });
```

### GGUF Configuration Options

```csharp
var options = new GeneratorOptions
{
    // Context length (default: from model metadata)
    MaxContextLength = 4096,

    // GPU layers (0 = CPU only, -1 = all layers on GPU)
    GpuLayerCount = -1,

    // Batch size for prompt processing
    BatchSize = 512,

    // Number of threads for CPU inference
    ThreadCount = 8
};

await using var model = await LocalGenerator.LoadAsync("gguf:default", options);
```

### Chat Generation with GGUF

```csharp
using LMSupply.Generator;
using LMSupply.Generator.Models;

await using var model = await LocalGenerator.LoadAsync("gguf:default");

var messages = new[]
{
    ChatMessage.System("You are a helpful assistant."),
    ChatMessage.User("What is the capital of France?")
};

await foreach (var token in model.GenerateChatAsync(messages))
{
    Console.Write(token);
}
```

### Generation Options

```csharp
var genOptions = new GenerationOptions
{
    MaxTokens = 256,          // Maximum tokens to generate
    Temperature = 0.7f,        // Randomness (0.0 = deterministic)
    TopP = 0.9f,              // Nucleus sampling
    TopK = 40                  // Top-K sampling
};

await foreach (var token in model.GenerateAsync(prompt, genOptions))
{
    Console.Write(token);
}
```

### Supported Chat Formats

The library auto-detects chat format from model filenames:

| Format | Models |
|--------|--------|
| Llama 3 | Llama-3, Llama-3.1, Llama-3.2, CodeLlama |
| ChatML | Qwen, Yi, InternLM, OpenChat |
| Gemma | Gemma, Gemma-2 |
| Phi-3 | Phi-3, Phi-3.5, Phi-4 |
| Mistral | Mistral, Mixtral |
| EXAONE | EXAONE |
| DeepSeek | DeepSeek, DeepSeek-R1 |
| Vicuna | Vicuna |
| Zephyr | Zephyr |

### Model Information

```csharp
await using var model = await LocalGenerator.LoadAsync("gguf:default");

var info = model.GetModelInfo();

Console.WriteLine($"Model: {info.ModelId}");
Console.WriteLine($"Path: {info.ModelPath}");
Console.WriteLine($"Context: {info.MaxContextLength}");
Console.WriteLine($"Format: {info.ChatFormat}");
Console.WriteLine($"Provider: {info.ExecutionProvider}");  // "LLamaSharp"
```

### GGUF vs ONNX

| Feature | GGUF (LLamaSharp) | ONNX (GenAI) |
|---------|-------------------|--------------|
| Model availability | Extensive | Limited |
| Quantization options | Many (Q2-Q8) | FP16, INT4 |
| Setup complexity | Simple | Simple |
| GPU support | CUDA, Metal | CUDA, DirectML, CoreML |
| Memory efficiency | Good | Good |
| Inference speed | Fast | Fast |

## Requirements

- .NET 10.0+
- ONNX Runtime GenAI 0.7+ (for ONNX models)
- LLamaSharp 0.25+ (for GGUF models)
- Windows, Linux, or macOS

## License

MIT License

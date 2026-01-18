# LMSupply.Llama

Shared llama.cpp runtime management for GGUF model support in LMSupply.

## Overview

`LMSupply.Llama` provides centralized management of LLamaSharp/llama.cpp native binaries, enabling GGUF model support across LMSupply packages (Embedder, Generator, and future packages).

This package follows LMSupply's on-demand philosophy: native binaries are downloaded only when first needed, with automatic GPU backend detection and fallback.

## Features

- **On-demand binary download**: Native llama.cpp binaries downloaded from NuGet on first use
- **Automatic backend selection**: Detects hardware and selects optimal backend
- **Fallback chain**: Automatically falls back to CPU if GPU backends fail
- **Cross-platform**: Windows, Linux, macOS (including Apple Silicon)
- **Shared runtime**: Single runtime instance shared across all GGUF models

## Supported Backends

| Backend | Platform | Hardware |
|---------|----------|----------|
| `Cuda13` | Windows/Linux | NVIDIA GPU with CUDA 13.x |
| `Cuda12` | Windows/Linux | NVIDIA GPU with CUDA 12.x |
| `Vulkan` | Windows/Linux | AMD/Intel discrete GPU |
| `Metal` | macOS | Apple Silicon (M1/M2/M3) |
| `Cpu` | All | CPU with AVX2/AVX512 optimization |

## Backend Selection

The runtime manager automatically detects your hardware and selects the best backend:

```
macOS ARM64:  Metal → CPU
NVIDIA GPU:   CUDA 13 → CUDA 12 → CPU
AMD GPU:      Vulkan → CPU
Intel GPU:    Vulkan → CPU
No GPU:       CPU
```

## Usage

This package is typically used indirectly through `LMSupply.Embedder` or `LMSupply.Generator`. Direct usage is also supported:

```csharp
using LMSupply.Llama;
using LMSupply.Runtime;

// Get the singleton instance
var manager = LlamaRuntimeManager.Instance;

// Initialize with auto-detection
await manager.EnsureInitializedAsync(ExecutionProvider.Auto);

// Check active backend
Console.WriteLine($"Backend: {manager.ActiveBackend}");
Console.WriteLine($"Binary Path: {manager.BinaryPath}");

// Get environment summary
Console.WriteLine(manager.GetEnvironmentSummary());
```

## GPU Layer Calculation

The runtime manager can recommend GPU layer counts based on available VRAM:

```csharp
var manager = LlamaRuntimeManager.Instance;
await manager.EnsureInitializedAsync();

// Estimate layers for a 7B model (~14GB)
long modelSize = 14L * 1024 * 1024 * 1024;
int gpuLayers = manager.GetRecommendedGpuLayers(modelSize);

Console.WriteLine($"Recommended GPU layers: {gpuLayers}");
```

## Manual Backend Override

Force a specific backend:

```csharp
// Force CUDA
await manager.EnsureInitializedAsync(ExecutionProvider.Cuda);

// Force CPU only
await manager.EnsureInitializedAsync(ExecutionProvider.Cpu);
```

## Architecture

```
LMSupply.Llama
├── LlamaRuntimeManager.cs   - Singleton runtime initialization
├── LlamaNuGetDownloader.cs  - Downloads backend NuGet packages
└── LlamaBackend.cs          - Backend enum definition

Dependencies:
├── LMSupply.Core            - GPU detection, environment info
└── LLamaSharp               - C# bindings for llama.cpp
```

## Consumer Packages

The following packages use LMSupply.Llama for GGUF support:

- **LMSupply.Embedder** - GGUF embedding models (e.g., nomic-embed-text)
- **LMSupply.Generator** - GGUF language models (e.g., Llama 3.2, Qwen 2.5)

## Troubleshooting

### Backend Initialization Failed

If you see errors about missing native libraries:

1. Ensure you have network access for the first-run download
2. Check cache directory permissions (`~/.cache/lmsupply` or `%LOCALAPPDATA%\LMSupply`)
3. For CUDA, verify NVIDIA drivers are installed

### Force CPU Backend

If GPU backends are causing issues:

```csharp
var options = new EmbedderOptions { Provider = ExecutionProvider.Cpu };
await using var model = await LocalEmbedder.LoadAsync("nomic-ai/nomic-embed-text-v1.5-GGUF", options);
```

### Clear Cache

Delete cached binaries to force re-download:

```bash
# Windows
del /s /q %LOCALAPPDATA%\LMSupply\cache\runtimes\llamasharp

# Linux/macOS
rm -rf ~/.local/share/LMSupply/cache/runtimes/llamasharp
```

## Version Compatibility

LMSupply.Llama automatically uses the LLamaSharp version specified in the project's dependency tree, ensuring backend package compatibility.

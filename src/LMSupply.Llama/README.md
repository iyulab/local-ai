# LMSupply.Llama

Shared LLamaSharp/llama.cpp runtime management for LMSupply libraries.

## Features

- **On-demand binary downloading**: Native binaries are downloaded from NuGet on first use
- **GPU auto-detection**: Automatic selection of CUDA, Vulkan, Metal, or CPU backend
- **Fallback chain**: CUDA 13 → CUDA 12 → Vulkan → CPU (platform-dependent)
- **Caching**: Downloaded binaries are cached locally for reuse

## Supported Backends

| Backend | Platform | GPU |
|---------|----------|-----|
| CUDA 12/13 | Windows, Linux | NVIDIA |
| Vulkan | Windows, Linux | AMD, Intel |
| Metal | macOS | Apple Silicon |
| CPU | All | AVX2/AVX512 optimized |

## Usage

This package is used internally by:
- `LMSupply.Generator` - GGUF text generation
- `LMSupply.Embedder` - GGUF embedding

## Architecture

```
LMSupply.Llama
├── LlamaBackend.cs          # Backend enum (Cpu, Cuda12, Vulkan, Metal, etc.)
├── LlamaRuntimeManager.cs   # Singleton manager with fallback chain
├── LlamaNuGetDownloader.cs  # Downloads LLamaSharp.Backend.* packages
└── DownloadProgress.cs      # Progress reporting
```

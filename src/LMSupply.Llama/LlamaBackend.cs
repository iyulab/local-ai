namespace LMSupply.Llama;

/// <summary>
/// Specifies the llama.cpp backend type for native binary selection.
/// </summary>
public enum LlamaBackend
{
    /// <summary>
    /// CPU-only backend (AVX2 optimized).
    /// </summary>
    Cpu,

    /// <summary>
    /// CUDA 12.x backend for NVIDIA GPUs.
    /// </summary>
    Cuda12,

    /// <summary>
    /// CUDA 13.x backend for NVIDIA GPUs.
    /// </summary>
    Cuda13,

    /// <summary>
    /// Vulkan backend for cross-platform GPU acceleration.
    /// </summary>
    Vulkan,

    /// <summary>
    /// Metal backend for Apple Silicon (macOS/iOS).
    /// </summary>
    Metal,

    /// <summary>
    /// ROCm/HIP backend for AMD GPUs.
    /// </summary>
    Rocm
}

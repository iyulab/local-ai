using System.Diagnostics;
using System.Runtime.InteropServices;
using LLama.Native;
using LMSupply.Runtime;

namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// Manages llama.cpp native binary download and configuration.
/// Follows LMSupply's on-demand philosophy: binaries are downloaded only when first needed.
/// </summary>
public sealed class LlamaRuntimeManager
{
    private static readonly Lazy<LlamaRuntimeManager> _instance = new(() => new());

    /// <summary>
    /// Gets the singleton instance of the Llama runtime manager.
    /// </summary>
    public static LlamaRuntimeManager Instance => _instance.Value;

    private readonly SemaphoreSlim _initLock = new(1, 1);
    private bool _initialized;
    private LlamaBackend _activeBackend;
    private string? _binaryPath;

    /// <summary>
    /// Gets whether the runtime has been initialized.
    /// </summary>
    public bool IsInitialized => _initialized;

    /// <summary>
    /// Gets the currently active backend.
    /// </summary>
    public LlamaBackend ActiveBackend => _activeBackend;

    /// <summary>
    /// Gets the path to the loaded native binaries.
    /// </summary>
    public string? BinaryPath => _binaryPath;

    /// <summary>
    /// Ensures the llama.cpp runtime is initialized with the specified backend.
    /// Downloads native binaries on first use with automatic fallback.
    /// </summary>
    /// <param name="provider">The execution provider to use. Auto will use fallback chain.</param>
    /// <param name="progress">Optional progress reporter for download operations.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task EnsureInitializedAsync(
        ExecutionProvider provider = ExecutionProvider.Auto,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (_initialized)
            return;

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (_initialized)
                return;

            // 1. Detect platform and GPU
            var platform = EnvironmentDetector.DetectPlatform();
            var gpu = EnvironmentDetector.DetectGpu();

            // 2. Get backend fallback chain
            var chain = provider == ExecutionProvider.Auto
                ? GetBackendFallbackChain(platform, gpu)
                : [DetermineBackend(provider, platform, gpu)];

            Exception? lastException = null;

            // 3. Try each backend in the fallback chain
            foreach (var backend in chain)
            {
                try
                {
                    Debug.WriteLine($"[LlamaRuntimeManager] Trying backend: {backend}");

                    // Download native binaries from NuGet (if not cached)
                    var binaryPath = await DownloadNativeBinaryFromNuGetAsync(
                        backend, platform, progress, cancellationToken);

                    // Configure LLamaSharp to use downloaded binaries
                    ConfigureNativeLibrary(binaryPath, backend);

                    _activeBackend = backend;
                    _binaryPath = binaryPath;
                    _initialized = true;

                    Debug.WriteLine($"[LlamaRuntimeManager] Successfully initialized with backend: {backend}");
                    return;
                }
                catch (OperationCanceledException)
                {
                    throw; // Don't catch cancellation
                }
                catch (Exception ex) when (backend != LlamaBackend.Cpu)
                {
                    // Log and continue to next backend in chain
                    Debug.WriteLine($"[LlamaRuntimeManager] Backend '{backend}' failed: {ex.Message}. Trying next...");
                    lastException = ex;
                }
            }

            // Should not reach here since CPU is always in chain
            throw lastException ?? new InvalidOperationException("No backend available for LLamaSharp");
        }
        finally
        {
            _initLock.Release();
        }
    }

    /// <summary>
    /// Gets a prioritized list of backends to try based on detected hardware.
    /// Fallback chain: CUDA → Vulkan → Metal → CPU
    /// </summary>
    public static IReadOnlyList<LlamaBackend> GetBackendFallbackChain(PlatformInfo platform, GpuInfo gpu)
    {
        var chain = new List<LlamaBackend>();
        var isArm64 = platform.Architecture == Architecture.Arm64;

        // macOS ARM64: Metal first
        if (platform.IsMacOS && isArm64)
        {
            chain.Add(LlamaBackend.Metal);
        }

        // NVIDIA GPU: CUDA (try newest first, fallback to older)
        if (gpu.Vendor == GpuVendor.Nvidia)
        {
            if (gpu.CudaDriverVersionMajor >= 13)
                chain.Add(LlamaBackend.Cuda13);
            if (gpu.CudaDriverVersionMajor >= 12)
                chain.Add(LlamaBackend.Cuda12);
        }

        // AMD/Intel discrete GPU on Windows/Linux: Vulkan
        if ((gpu.Vendor == GpuVendor.Amd || gpu.Vendor == GpuVendor.Intel) && !platform.IsMacOS)
        {
            // Vulkan can be unstable, but include it in fallback chain
            chain.Add(LlamaBackend.Vulkan);
        }

        // CPU always as final fallback (well-optimized with AVX2/AVX512)
        chain.Add(LlamaBackend.Cpu);

        return chain;
    }

    /// <summary>
    /// Determines the best backend based on provider and detected hardware.
    /// </summary>
    private static LlamaBackend DetermineBackend(
        ExecutionProvider provider,
        PlatformInfo platform,
        GpuInfo gpu)
    {
        // Explicit provider selection
        if (provider != ExecutionProvider.Auto)
        {
            return provider switch
            {
                ExecutionProvider.Cuda when gpu.CudaDriverVersionMajor >= 13 => LlamaBackend.Cuda13,
                ExecutionProvider.Cuda when gpu.CudaDriverVersionMajor >= 12 => LlamaBackend.Cuda12,
                ExecutionProvider.Cuda => LlamaBackend.Cuda12, // fallback to CUDA 12
                ExecutionProvider.DirectML => LlamaBackend.Vulkan, // Use Vulkan as DirectML alternative
                ExecutionProvider.CoreML when platform.IsMacOS => LlamaBackend.Metal,
                _ => LlamaBackend.Cpu
            };
        }

        // Auto selection based on hardware
        var isArm64 = platform.Architecture == Architecture.Arm64;

        if (platform.IsMacOS && isArm64)
        {
            // Apple Silicon - always use Metal
            return LlamaBackend.Metal;
        }

        if (gpu.Vendor == GpuVendor.Nvidia)
        {
            // NVIDIA GPU - use CUDA (most reliable for llama.cpp)
            if (gpu.CudaDriverVersionMajor >= 13)
                return LlamaBackend.Cuda13;
            if (gpu.CudaDriverVersionMajor >= 12)
                return LlamaBackend.Cuda12;
        }

        if (gpu.Vendor == GpuVendor.Amd && !platform.IsMacOS)
        {
            // AMD discrete GPU - try Vulkan (but may be unstable)
            // Only use if explicitly requested via ExecutionProvider
            // For auto mode, prefer CPU for stability
        }

        // Intel integrated GPUs and other cases - use CPU for stability
        // Intel Iris Xe, Intel UHD, etc. often have issues with Vulkan backend
        // CPU backend is well-optimized with AVX2/AVX512 support
        return LlamaBackend.Cpu;
    }

    /// <summary>
    /// Downloads the native binary from NuGet LLamaSharp.Backend.* packages.
    /// </summary>
    private async Task<string> DownloadNativeBinaryFromNuGetAsync(
        LlamaBackend backend,
        PlatformInfo platform,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        using var downloader = new LlamaNuGetDownloader();
        return await downloader.DownloadAsync(
            backend,
            platform,
            progress: progress,
            cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Configures LLamaSharp to use the downloaded native binaries.
    /// </summary>
    private static void ConfigureNativeLibrary(string? binaryPath, LlamaBackend backend)
    {
        // Configure LLamaSharp to search in our download directory (if available)
        if (!string.IsNullOrEmpty(binaryPath) && Directory.Exists(binaryPath))
        {
            // Check if files are in variant subdirectories (avx, avx2, avx512, noavx)
            // Note: Using avx2 first because avx512 may have compatibility issues on some CPUs
            var variantDirs = new[] { "avx2", "avx", "noavx", "avx512" };
            string? selectedPath = null;

            foreach (var variant in variantDirs)
            {
                var variantPath = Path.Combine(binaryPath, variant);
                if (Directory.Exists(variantPath) && Directory.GetFiles(variantPath, "llama.*").Length > 0)
                {
                    selectedPath = variantPath;
                    break;
                }
            }

            // If no variant subdirectories, use the base path
            selectedPath ??= binaryPath;

            // Try to find llama library file and specify it directly
            var llamaLib = Directory.GetFiles(selectedPath, "llama.*").FirstOrDefault();
            if (llamaLib != null)
            {
                NativeLibraryConfig.All.WithLibrary(llamaLib, null);
            }
            else
            {
                NativeLibraryConfig.All.WithSearchDirectory(selectedPath);
            }
        }

        // Enable auto-fallback to search system paths and NuGet package paths
        // This allows using LLamaSharp.Backend.* packages as fallback
        NativeLibraryConfig.All.WithAutoFallback(true);

        // Configure backend-specific settings
        switch (backend)
        {
            case LlamaBackend.Cuda12:
            case LlamaBackend.Cuda13:
                NativeLibraryConfig.All.WithCuda();
                break;

            case LlamaBackend.Vulkan:
                NativeLibraryConfig.All.WithVulkan();
                break;

            case LlamaBackend.Metal:
                // Metal is automatically used on macOS arm64
                break;

            case LlamaBackend.Cpu:
            default:
                // CPU is the default
                break;
        }

        // Set logging level (use WithLogCallback for custom logging)
        NativeLibraryConfig.All.WithLogCallback((level, message) =>
        {
            if (level >= LLamaLogLevel.Warning)
            {
                System.Diagnostics.Debug.WriteLine($"[LLamaSharp:{level}] {message}");
            }
        });
    }

    /// <summary>
    /// Gets the recommended GPU layer count based on available VRAM.
    /// </summary>
    /// <param name="modelSizeBytes">The model size in bytes.</param>
    /// <returns>Recommended number of layers to offload to GPU.</returns>
    public int GetRecommendedGpuLayers(long modelSizeBytes)
    {
        if (_activeBackend == LlamaBackend.Cpu)
            return 0;

        var gpu = EnvironmentDetector.DetectGpu();
        var vramMB = gpu.TotalMemoryMB ?? 0;
        if (vramMB <= 0)
            return 0;

        var vramBytes = vramMB * 1024L * 1024L;

        // Reserve ~2GB for system overhead
        var availableVram = vramBytes - (2L * 1024 * 1024 * 1024);
        if (availableVram <= 0)
            return 0;

        // Estimate layers that can fit in VRAM
        // Typical 7B model: ~32 layers, ~14GB total
        // Each layer is roughly modelSize / numLayers
        const int typicalLayerCount = 32;
        var bytesPerLayer = modelSizeBytes / typicalLayerCount;

        if (bytesPerLayer <= 0)
            return typicalLayerCount; // Assume full offload for small models

        var layersInVram = (int)(availableVram / bytesPerLayer);
        return Math.Clamp(layersInVram, 0, 999); // LLamaSharp uses 999 as "all layers"
    }

    /// <summary>
    /// Gets environment summary for diagnostics.
    /// </summary>
    public string GetEnvironmentSummary()
    {
        if (!_initialized)
            return "LlamaRuntimeManager not initialized";

        return $"""
            Llama Backend: {_activeBackend}
            Binary Path: {_binaryPath}
            Initialized: {_initialized}
            """;
    }
}

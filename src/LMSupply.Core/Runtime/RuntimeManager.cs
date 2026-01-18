using System.Diagnostics;
using LMSupply.Download;

namespace LMSupply.Runtime;

/// <summary>
/// Orchestrates runtime binary management including detection, download, caching, and loading.
/// Downloads native binaries on-demand from NuGet.org - no pre-built manifest required.
/// </summary>
public sealed class RuntimeManager : IAsyncDisposable
{
    private readonly OnnxNuGetDownloader _nugetDownloader;
    private readonly RuntimeManagerOptions _options;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    private bool _initialized;
    private PlatformInfo? _platform;
    private GpuInfo? _gpu;

    /// <summary>
    /// Gets the singleton instance of the runtime manager.
    /// </summary>
    public static RuntimeManager Instance { get; } = new();

    /// <summary>
    /// Creates a new runtime manager with default options.
    /// </summary>
    public RuntimeManager() : this(new RuntimeManagerOptions())
    {
    }

    /// <summary>
    /// Creates a new runtime manager with custom options.
    /// </summary>
    public RuntimeManager(RuntimeManagerOptions options)
    {
        _options = options;
        _nugetDownloader = new OnnxNuGetDownloader(options.CacheDirectory);
    }

    /// <summary>
    /// Gets the detected platform information.
    /// </summary>
    public PlatformInfo Platform => _platform ?? throw new InvalidOperationException("Runtime manager not initialized");

    /// <summary>
    /// Gets the detected GPU information.
    /// </summary>
    public GpuInfo Gpu => _gpu ?? throw new InvalidOperationException("Runtime manager not initialized");

    /// <summary>
    /// Gets the recommended execution provider based on detected hardware.
    /// </summary>
    public ExecutionProvider RecommendedProvider => _gpu?.RecommendedProvider ?? ExecutionProvider.Cpu;

    /// <summary>
    /// Initializes the runtime manager by detecting hardware.
    /// </summary>
    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        if (_initialized)
            return;

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (_initialized)
                return;

            // Detect platform and GPU
            _platform = EnvironmentDetector.DetectPlatform();
            _gpu = EnvironmentDetector.DetectGpu();

            // Setup CUDA/cuDNN DLL search paths for Windows
            // This must be done before any ONNX session creation
            SetupCudaDllSearchPaths();

            _initialized = true;
        }
        finally
        {
            _initLock.Release();
        }
    }

    /// <summary>
    /// Sets up CUDA and cuDNN DLL search paths for Windows.
    /// This enables ONNX Runtime's CUDA provider to find native dependencies.
    /// Uses both AddDllDirectory (for LoadLibraryEx) and PATH modification (for LoadLibrary).
    /// </summary>
    private void SetupCudaDllSearchPaths()
    {
        if (!OperatingSystem.IsWindows())
            return;

        // Initialize CUDA environment detection
        var cudaEnv = CudaEnvironment.Instance;
        cudaEnv.Initialize();

        // Determine target CUDA version from detected GPU
        var cudaMajorVersion = _gpu?.CudaDriverVersionMajor ?? 12;

        // Get all DLL search paths from CudaEnvironment
        var pathsToAdd = cudaEnv.GetDllSearchPaths(cudaMajorVersion).ToList();

        // Register paths with NativeLoader
        foreach (var path in pathsToAdd)
        {
            NativeLoader.Instance.AddToWindowsDllSearchPath(path);
            Debug.WriteLine($"[RuntimeManager] Added to DLL search path: {path}");
        }

        // Also modify PATH environment variable for current process
        // This ensures ONNX Runtime can find DLLs even when using standard LoadLibrary
        if (pathsToAdd.Count > 0)
        {
            var currentPath = Environment.GetEnvironmentVariable("PATH") ?? "";
            var newPaths = pathsToAdd.Where(p => !currentPath.Contains(p, StringComparison.OrdinalIgnoreCase));
            if (newPaths.Any())
            {
                var pathToAdd = string.Join(Path.PathSeparator.ToString(), newPaths);
                Environment.SetEnvironmentVariable("PATH", pathToAdd + Path.PathSeparator + currentPath);
                Debug.WriteLine($"[RuntimeManager] Added to PATH: {pathToAdd}");
            }
        }

        // Log diagnostics in debug mode
        Debug.WriteLine(cudaEnv.GetDiagnostics());
    }

    /// <summary>
    /// Ensures a runtime binary is available, downloading from NuGet if necessary.
    /// When provider is null (Auto mode), uses the fallback chain: CUDA → DirectML → CoreML → CPU.
    /// </summary>
    /// <param name="package">The package name (e.g., "onnxruntime").</param>
    /// <param name="version">Optional version. If null, auto-detects from assembly.</param>
    /// <param name="provider">Optional provider. If null, uses fallback chain for best available.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Path to the binary directory.</returns>
    public async Task<string> EnsureRuntimeAsync(
        string package,
        string? version = null,
        string? provider = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        await InitializeAsync(cancellationToken);

        // If provider is explicitly specified, download for that provider
        if (!string.IsNullOrEmpty(provider))
        {
            return await DownloadRuntimeForProviderAsync(provider, version, progress, cancellationToken);
        }

        // Auto mode: try providers in fallback chain order
        var chain = GetProviderFallbackChain();
        Exception? lastException = null;

        foreach (var providerToTry in chain)
        {
            try
            {
                return await DownloadRuntimeForProviderAsync(providerToTry, version, progress, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                throw; // Don't catch cancellation
            }
            catch (Exception ex) when (providerToTry != "cpu")
            {
                // Log and continue to next provider in chain
                Debug.WriteLine(
                    $"[RuntimeManager] Provider '{providerToTry}' failed: {ex.Message}. Trying next provider...");
                lastException = ex;
            }
        }

        // Should not reach here since CPU is always in chain
        throw lastException ?? new InvalidOperationException("No provider available for ONNX Runtime");
    }

    /// <summary>
    /// Downloads runtime for a specific provider from NuGet.
    /// </summary>
    private async Task<string> DownloadRuntimeForProviderAsync(
        string provider,
        string? version,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var binaryPath = await _nugetDownloader.DownloadAsync(
            provider,
            _platform!,
            version,
            progress,
            cancellationToken);

        // Register with NativeLoader for DLL resolution
        NativeLoader.Instance.RegisterDirectory(binaryPath, preload: true, primaryLibrary: "onnxruntime");

        return binaryPath;
    }

    /// <summary>
    /// Gets the best provider string for the current hardware.
    /// </summary>
    public string GetDefaultProvider()
    {
        if (_gpu is null)
            return "cpu";

        return _gpu.Vendor switch
        {
            GpuVendor.Nvidia when _gpu.CudaDriverVersionMajor >= 12 => "cuda12",
            GpuVendor.Nvidia when _gpu.CudaDriverVersionMajor >= 11 => "cuda11",
            _ when _gpu.DirectMLSupported => "directml",
            _ when _gpu.CoreMLSupported => "coreml",
            _ => "cpu"
        };
    }

    /// <summary>
    /// Gets a prioritized list of providers to try based on detected hardware.
    /// The fallback chain ensures zero-configuration GPU acceleration:
    /// CUDA (cuda12/cuda11) → DirectML → CoreML → CPU
    /// </summary>
    public IReadOnlyList<string> GetProviderFallbackChain()
    {
        var chain = new List<string>();

        if (_gpu is not null)
        {
            // CUDA first (if NVIDIA GPU with sufficient driver)
            if (_gpu.Vendor == GpuVendor.Nvidia)
            {
                if (_gpu.CudaDriverVersionMajor >= 12)
                    chain.Add("cuda12");
                else if (_gpu.CudaDriverVersionMajor >= 11)
                    chain.Add("cuda11");
            }

            // DirectML (Windows with D3D12 support - works with AMD, Intel, NVIDIA)
            if (_gpu.DirectMLSupported)
                chain.Add("directml");

            // CoreML (macOS/iOS)
            if (_gpu.CoreMLSupported)
                chain.Add("coreml");
        }

        // CPU always as final fallback
        chain.Add("cpu");

        return chain;
    }

    /// <summary>
    /// Gets the cache directory path.
    /// </summary>
    public string CacheDirectory => _options.CacheDirectory ?? GetDefaultCacheDirectory();

    /// <summary>
    /// Gets environment information summary.
    /// </summary>
    public string GetEnvironmentSummary()
    {
        if (!_initialized)
            return "Runtime manager not initialized";

        return $"""
            Platform: {_platform}
            GPU: {_gpu}
            Recommended Provider: {RecommendedProvider}
            Default Provider String: {GetDefaultProvider()}
            Cache Directory: {CacheDirectory}
            """;
    }

    private static string GetDefaultCacheDirectory()
    {
        var baseDir = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return Path.Combine(baseDir, "LMSupply", "cache", "runtimes");
    }

    public async ValueTask DisposeAsync()
    {
        _nugetDownloader.Dispose();
        _initLock.Dispose();
    }
}

/// <summary>
/// Options for the runtime manager.
/// </summary>
public sealed class RuntimeManagerOptions
{
    /// <summary>
    /// Gets or sets the cache directory.
    /// </summary>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Gets or sets the maximum cache size in bytes. Default is 10 GB.
    /// </summary>
    public long MaxCacheSize { get; set; } = 10L * 1024 * 1024 * 1024;

    /// <summary>
    /// Gets or sets the proxy URL.
    /// </summary>
    public string? ProxyUrl { get; set; }

    /// <summary>
    /// Gets or sets the proxy username.
    /// </summary>
    public string? ProxyUsername { get; set; }

    /// <summary>
    /// Gets or sets the proxy password.
    /// </summary>
    public string? ProxyPassword { get; set; }

    /// <summary>
    /// Gets or sets the maximum retry attempts for downloads.
    /// </summary>
    public int MaxRetries { get; set; } = 3;
}

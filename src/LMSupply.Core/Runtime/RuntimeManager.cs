using System.Diagnostics;
using LMSupply.Download;

namespace LMSupply.Runtime;

/// <summary>
/// Orchestrates runtime binary management including detection, download, caching, and loading.
/// This is the main entry point for the lazy binary distribution system.
/// </summary>
public sealed class RuntimeManager : IAsyncDisposable
{
    private readonly ManifestProvider _manifestProvider;
    private readonly BinaryDownloader _downloader;
    private readonly RuntimeCache _cache;
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
        _manifestProvider = new ManifestProvider();
        _downloader = new BinaryDownloader(new BinaryDownloaderOptions
        {
            ProxyUrl = options.ProxyUrl,
            ProxyUsername = options.ProxyUsername,
            ProxyPassword = options.ProxyPassword,
            MaxRetries = options.MaxRetries
        });
        _cache = new RuntimeCache(new RuntimeCacheOptions
        {
            CacheDirectory = options.CacheDirectory,
            MaxCacheSize = options.MaxCacheSize
        });
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
    /// Ensures a runtime binary is available, downloading if necessary.
    /// When provider is null (Auto mode), uses the fallback chain: CUDA → DirectML → CoreML → CPU.
    /// </summary>
    /// <param name="package">The package name (e.g., "onnxruntime").</param>
    /// <param name="version">Optional version. If null, uses latest available.</param>
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

        var rid = _platform!.RuntimeIdentifier;
        var manifest = await _manifestProvider.GetManifestAsync(cancellationToken: cancellationToken);

        // Resolve version if not specified
        var actualVersion = version;
        if (string.IsNullOrEmpty(actualVersion))
        {
            var pkg = manifest.GetPackage(package);
            actualVersion = pkg?.Versions.Keys
                .OrderByDescending(v => Version.TryParse(v, out var ver) ? ver : new Version(0, 0))
                .FirstOrDefault();

            if (actualVersion is null)
                throw new InvalidOperationException($"No versions available for package: {package}");
        }

        // If provider is explicitly specified, use single-provider logic with CPU fallback
        if (!string.IsNullOrEmpty(provider))
        {
            return await EnsureRuntimeForProviderAsync(
                package, actualVersion, rid, provider, progress, cancellationToken);
        }

        // Auto mode: try providers in fallback chain order
        var chain = GetProviderFallbackChain();
        Exception? lastException = null;

        foreach (var providerToTry in chain)
        {
            try
            {
                return await EnsureRuntimeForProviderAsync(
                    package, actualVersion, rid, providerToTry, progress, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                throw; // Don't catch cancellation
            }
            catch (Exception ex) when (providerToTry != "cpu")
            {
                // Log and continue to next provider in chain
                System.Diagnostics.Debug.WriteLine(
                    $"[RuntimeManager] Provider '{providerToTry}' failed for {package}: {ex.Message}. Trying next provider...");
                lastException = ex;
            }
        }

        // Should not reach here since CPU is always in chain, but just in case
        throw lastException ?? new InvalidOperationException($"No provider available for {package}");
    }

    /// <summary>
    /// Ensures runtime for a specific provider with CPU fallback.
    /// </summary>
    private async Task<string> EnsureRuntimeForProviderAsync(
        string package,
        string version,
        string rid,
        string provider,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Check cache first
        var cachedPath = await _cache.GetCachedPathAsync(package, version, rid, provider, cancellationToken);
        if (cachedPath is not null)
        {
            NativeLoader.Instance.RegisterDirectory(Path.GetDirectoryName(cachedPath)!, preload: true, primaryLibrary: package);
            return Path.GetDirectoryName(cachedPath)!;
        }

        // Get binary entry from manifest
        var entry = await _manifestProvider.GetBinaryAsync(package, version, rid, provider, cancellationToken);
        if (entry is null)
        {
            throw new InvalidOperationException(
                $"No binary available for {package} {version} on {rid} with provider {provider}");
        }

        // Download and cache
        var targetDirectory = _cache.GetCacheDirectory(package, version, rid, provider);
        var binaryPath = await _downloader.DownloadAsync(entry, targetDirectory, progress, cancellationToken);

        await _cache.RegisterAsync(entry, package, version, binaryPath, cancellationToken);
        NativeLoader.Instance.RegisterDirectory(targetDirectory, preload: true, primaryLibrary: package);

        return targetDirectory;
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
    /// Gets the runtime cache.
    /// </summary>
    public RuntimeCache Cache => _cache;

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
            Cache Directory: {_cache.GetCacheDirectory("", "", "", "")}
            Cache Size: {_cache.GetTotalCacheSize() / (1024.0 * 1024.0):F2} MB
            """;
    }

    public async ValueTask DisposeAsync()
    {
        _manifestProvider.Dispose();
        _downloader.Dispose();
        _cache.Dispose();
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

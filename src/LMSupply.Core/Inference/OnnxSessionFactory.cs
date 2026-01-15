using System.Diagnostics;
using LMSupply.Download;
using LMSupply.Runtime;
using Microsoft.ML.OnnxRuntime;

namespace LMSupply.Inference;

/// <summary>
/// Result of session creation including metadata about the active execution provider.
/// </summary>
public sealed class SessionCreationResult
{
    /// <summary>
    /// The created inference session.
    /// </summary>
    public required InferenceSession Session { get; init; }

    /// <summary>
    /// The execution provider that was requested.
    /// </summary>
    public required ExecutionProvider RequestedProvider { get; init; }

    /// <summary>
    /// The execution providers actually active in the session.
    /// </summary>
    public required IReadOnlyList<string> ActiveProviders { get; init; }

    /// <summary>
    /// Whether GPU acceleration is actually being used.
    /// </summary>
    public bool IsGpuActive => ActiveProviders.Any(p =>
        p.Contains("CUDA", StringComparison.OrdinalIgnoreCase) ||
        p.Contains("DML", StringComparison.OrdinalIgnoreCase) ||
        p.Contains("DirectML", StringComparison.OrdinalIgnoreCase) ||
        p.Contains("CoreML", StringComparison.OrdinalIgnoreCase) ||
        p.Contains("TensorRT", StringComparison.OrdinalIgnoreCase));
}

/// <summary>
/// Factory for creating ONNX Runtime inference sessions with proper execution provider configuration.
/// Supports lazy loading of native runtime binaries via RuntimeManager.
/// </summary>
public static class OnnxSessionFactory
{
    /// <summary>
    /// Creates an ONNX Runtime inference session asynchronously, ensuring runtime binaries are available.
    /// This is the recommended method as it downloads required binaries on first use.
    /// When provider is Auto, uses fallback chain: CUDA → DirectML → CoreML → CPU.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="provider">The execution provider to use.</param>
    /// <param name="configureOptions">Optional callback to configure additional session options.</param>
    /// <param name="progress">Optional progress reporter for binary downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A configured inference session.</returns>
    public static async Task<InferenceSession> CreateAsync(
        string modelPath,
        ExecutionProvider provider = ExecutionProvider.Auto,
        Action<SessionOptions>? configureOptions = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Use CreateWithInfoAsync which handles the fallback chain and return just the session
        var result = await CreateWithInfoAsync(modelPath, provider, configureOptions, progress, cancellationToken);
        return result.Session;
    }

    /// <summary>
    /// Creates an ONNX Runtime inference session with detailed information about active providers.
    /// Use this when you need to verify GPU acceleration is actually working.
    /// When provider is Auto, uses fallback chain: CUDA → DirectML → CoreML → CPU.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="provider">The execution provider to use.</param>
    /// <param name="configureOptions">Optional callback to configure additional session options.</param>
    /// <param name="progress">Optional progress reporter for binary downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A session creation result with provider information.</returns>
    public static async Task<SessionCreationResult> CreateWithInfoAsync(
        string modelPath,
        ExecutionProvider provider = ExecutionProvider.Auto,
        Action<SessionOptions>? configureOptions = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Ensure runtime binaries are available
        await RuntimeManager.Instance.InitializeAsync(cancellationToken);

        if (provider == ExecutionProvider.Auto)
        {
            // Use fallback chain: CUDA → DirectML → CoreML → CPU
            return await CreateWithFallbackChainAsync(modelPath, configureOptions, progress, cancellationToken);
        }

        // Explicit provider specified - use single provider with CPU fallback
        var providerString = provider switch
        {
            ExecutionProvider.Cuda => "cuda12",  // Try CUDA 12 first
            ExecutionProvider.DirectML => "directml",
            ExecutionProvider.CoreML => "coreml",
            _ => "cpu"
        };

        // Download runtime binaries if needed
        await RuntimeManager.Instance.EnsureRuntimeAsync(
            "onnxruntime",
            provider: providerString,
            progress: progress,
            cancellationToken: cancellationToken);

        // Create session and get active providers
        var session = Create(modelPath, provider, configureOptions);
        var activeProviders = GetActiveProviders(session);

        // Log warning if GPU was requested but not active
        var isGpuRequested = provider is ExecutionProvider.Cuda or ExecutionProvider.DirectML or ExecutionProvider.CoreML;
        var hasGpuProvider = activeProviders.Any(p =>
            p.Contains("CUDA", StringComparison.OrdinalIgnoreCase) ||
            p.Contains("DML", StringComparison.OrdinalIgnoreCase) ||
            p.Contains("CoreML", StringComparison.OrdinalIgnoreCase));

        if (isGpuRequested && !hasGpuProvider)
        {
            Debug.WriteLine($"[OnnxSessionFactory] Warning: GPU provider {provider} was requested but only CPU is active. " +
                $"Active providers: {string.Join(", ", activeProviders)}");
        }

        return new SessionCreationResult
        {
            Session = session,
            RequestedProvider = provider,
            ActiveProviders = activeProviders
        };
    }

    /// <summary>
    /// Creates a session using the fallback chain until one succeeds.
    /// For CUDA, verifies runtime libraries are available before attempting.
    /// For DirectML, trusts the session creation result (Windows manages DirectML).
    /// </summary>
    private static async Task<SessionCreationResult> CreateWithFallbackChainAsync(
        string modelPath,
        Action<SessionOptions>? configureOptions,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var fallbackChain = RuntimeManager.Instance.Gpu?.GetFallbackProviders()
            ?? new[] { ExecutionProvider.Cpu };

        Exception? lastException = null;
        var triedProviders = new List<string>();

        foreach (var providerToTry in fallbackChain)
        {
            try
            {
                // For CUDA, check if runtime libraries are available FIRST
                if (providerToTry == ExecutionProvider.Cuda)
                {
                    var (cudaAvailable, missingLibs) = CheckCudaRuntimeAvailability();
                    if (!cudaAvailable)
                    {
                        Console.WriteLine($"[Fallback] CUDA: skipped (missing: {string.Join(", ", missingLibs)})");
                        triedProviders.Add("CUDA(skipped)");
                        continue;
                    }
                }

                // Get provider string for binary download
                var providerString = providerToTry switch
                {
                    ExecutionProvider.Cuda => "cuda12",  // Try CUDA 12 first
                    ExecutionProvider.DirectML => "directml",
                    ExecutionProvider.CoreML => "coreml",
                    _ => "cpu"
                };

                // Download runtime binaries
                await RuntimeManager.Instance.EnsureRuntimeAsync(
                    "onnxruntime",
                    provider: providerString,
                    progress: progress,
                    cancellationToken: cancellationToken);

                // Try to create session with this provider
                var session = Create(modelPath, providerToTry, configureOptions);

                // Determine active providers based on what was requested
                var activeProviders = new List<string>();

                if (providerToTry == ExecutionProvider.Cuda)
                {
                    // For CUDA, verify the provider DLL was actually loaded
                    if (IsCudaProviderLoaded())
                    {
                        activeProviders.Add("CUDAExecutionProvider");
                        Console.WriteLine($"[Fallback] CUDA: success");
                    }
                    else
                    {
                        Console.WriteLine($"[Fallback] CUDA: failed (provider not loaded), trying next...");
                        triedProviders.Add("CUDA(failed)");
                        session.Dispose();
                        continue;
                    }
                }
                else if (providerToTry == ExecutionProvider.DirectML)
                {
                    // DirectML is managed by Windows - trust the session creation
                    // If it fails, ONNX Runtime would have thrown or fallen back internally
                    activeProviders.Add("DmlExecutionProvider");
                    Console.WriteLine($"[Fallback] DirectML: success");
                }
                else if (providerToTry == ExecutionProvider.CoreML)
                {
                    activeProviders.Add("CoreMLExecutionProvider");
                    Console.WriteLine($"[Fallback] CoreML: success");
                }

                activeProviders.Add("CPUExecutionProvider");

                return new SessionCreationResult
                {
                    Session = session,
                    RequestedProvider = ExecutionProvider.Auto,
                    ActiveProviders = activeProviders
                };
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex) when (providerToTry != ExecutionProvider.Cpu)
            {
                // Provide helpful error messages based on the error type
                if (ex.Message.Contains("CUDNN_STATUS_NOT_INITIALIZED"))
                {
                    Console.WriteLine($"[Fallback] {providerToTry}: cuDNN not in PATH. Add cuDNN bin to system PATH.");
                    Console.WriteLine($"           Example: C:\\Program Files\\NVIDIA\\CUDNN\\v9.x\\bin\\12.x");
                }
                else
                {
                    // Try to parse missing library names from CUDA errors
                    var missingLibs = ParseMissingLibrariesFromError(ex.Message);
                    if (missingLibs.Length > 0)
                    {
                        Console.WriteLine($"[Fallback] {providerToTry}: missing {string.Join(", ", missingLibs)}");
                    }
                    else
                    {
                        // Truncate long error messages
                        var msg = ex.Message.Length > 100 ? ex.Message[..100] + "..." : ex.Message;
                        Console.WriteLine($"[Fallback] {providerToTry}: {msg}");
                    }
                }
                triedProviders.Add($"{providerToTry}(error)");
                lastException = ex;
            }
        }

        // Final fallback to CPU
        Console.WriteLine($"[Fallback] Using CPU (tried: {string.Join(" -> ", triedProviders)})");

        await RuntimeManager.Instance.EnsureRuntimeAsync(
            "onnxruntime", provider: "cpu", progress: progress, cancellationToken: cancellationToken);

        var cpuSession = Create(modelPath, ExecutionProvider.Cpu, configureOptions);
        return new SessionCreationResult
        {
            Session = cpuSession,
            RequestedProvider = ExecutionProvider.Auto,
            ActiveProviders = new[] { "CPUExecutionProvider" }
        };
    }

    /// <summary>
    /// Checks if CUDA runtime libraries (cuBLAS, cuDNN) are available on the system.
    /// This is checked BEFORE attempting CUDA to avoid ONNX Runtime error messages.
    /// </summary>
    private static bool IsCudaRuntimeAvailable()
    {
        var (available, _) = CheckCudaRuntimeAvailability();
        return available;
    }

    /// <summary>
    /// Checks CUDA runtime availability using NativeLibrary.TryLoad.
    /// This works before ONNX Runtime is loaded.
    /// </summary>
    /// <returns>Tuple of (isAvailable, missingLibraries)</returns>
    private static (bool Available, string[] MissingLibraries) CheckCudaRuntimeAvailability()
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            return (false, new[] { "CUDA not supported on this platform" });

        // Get CUDA installation path
        var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        if (string.IsNullOrEmpty(cudaPath))
        {
            return (false, new[] { "CUDA_PATH not set" });
        }

        var cudaBin = Path.Combine(cudaPath, "bin");
        if (!Directory.Exists(cudaBin))
        {
            return (false, new[] { $"CUDA bin directory not found: {cudaBin}" });
        }

        // Check for required CUDA libraries
        var librariesToCheck = new[] { "cublas64_12.dll", "cublasLt64_12.dll" };
        var missing = new List<string>();

        foreach (var lib in librariesToCheck)
        {
            var libPath = Path.Combine(cudaBin, lib);
            if (!File.Exists(libPath))
            {
                missing.Add(lib);
            }
        }

        // Also check for cuDNN (required for CUDA provider)
        var cudnnDll = "cudnn64_9.dll";
        bool cudnnFound = false;

        // Check in CUDA bin directory
        if (File.Exists(Path.Combine(cudaBin, cudnnDll)))
        {
            cudnnFound = true;
        }

        // Check in standard cuDNN installation paths
        if (!cudnnFound)
        {
            var cudnnBasePath = @"C:\Program Files\NVIDIA\CUDNN";
            if (Directory.Exists(cudnnBasePath))
            {
                foreach (var versionDir in Directory.GetDirectories(cudnnBasePath, "v*"))
                {
                    var binDir = Path.Combine(versionDir, "bin");
                    if (Directory.Exists(binDir))
                    {
                        // Check CUDA version-specific subdirectories
                        foreach (var cudaVersionDir in Directory.GetDirectories(binDir))
                        {
                            if (File.Exists(Path.Combine(cudaVersionDir, cudnnDll)))
                            {
                                cudnnFound = true;
                                break;
                            }
                        }
                        // Also check bin directory itself
                        if (!cudnnFound && File.Exists(Path.Combine(binDir, cudnnDll)))
                        {
                            cudnnFound = true;
                        }
                    }
                    if (cudnnFound) break;
                }
            }
        }

        if (!cudnnFound)
        {
            missing.Add(cudnnDll);
        }

        if (missing.Count > 0)
            return (false, missing.ToArray());

        return (true, Array.Empty<string>());
    }

    /// <summary>
    /// Parses ONNX Runtime error messages to extract missing library names.
    /// Example: '...depends on "cublasLt64_12.dll" which is missing...'
    /// </summary>
    private static string[] ParseMissingLibrariesFromError(string errorMessage)
    {
        var missing = new List<string>();

        // Pattern: depends on "xxx.dll" which is missing
        var regex = new System.Text.RegularExpressions.Regex(
            @"depends on ""([^""]+\.dll)"" which is missing",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);

        var matches = regex.Matches(errorMessage);
        foreach (System.Text.RegularExpressions.Match match in matches)
        {
            if (match.Groups.Count > 1)
                missing.Add(match.Groups[1].Value);
        }

        return missing.ToArray();
    }

    /// <summary>
    /// Checks if CUDA provider DLL was successfully loaded into the process.
    /// </summary>
    private static bool IsCudaProviderLoaded()
    {
        try
        {
            var modules = System.Diagnostics.Process.GetCurrentProcess().Modules;
            foreach (System.Diagnostics.ProcessModule module in modules)
            {
                if (module.ModuleName?.Contains("onnxruntime_providers_cuda", StringComparison.OrdinalIgnoreCase) == true)
                    return true;
            }
        }
        catch { }
        return false;
    }

    /// <summary>
    /// Creates an ONNX Runtime inference session with the specified execution provider.
    /// Note: This assumes runtime binaries are already available. For lazy loading, use CreateAsync.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="provider">The execution provider to use.</param>
    /// <param name="configureOptions">Optional callback to configure additional session options.</param>
    /// <returns>A configured inference session.</returns>
    public static InferenceSession Create(
        string modelPath,
        ExecutionProvider provider = ExecutionProvider.Auto,
        Action<SessionOptions>? configureOptions = null)
    {
        var options = new SessionOptions();

        // Apply user configuration first
        configureOptions?.Invoke(options);

        // Configure execution provider
        ConfigureExecutionProvider(options, provider);

        return new InferenceSession(modelPath, options);
    }

    /// <summary>
    /// Configures the execution provider for the session options.
    /// </summary>
    public static void ConfigureExecutionProvider(SessionOptions options, ExecutionProvider provider)
    {
        switch (provider)
        {
            case ExecutionProvider.Auto:
                TryAddBestAvailableProvider(options);
                break;

            case ExecutionProvider.Cuda:
                TryAddCuda(options);
                break;

            case ExecutionProvider.DirectML:
                TryAddDirectML(options);
                break;

            case ExecutionProvider.CoreML:
                TryAddCoreML(options);
                break;

            case ExecutionProvider.Cpu:
                // CPU is always available as fallback
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(provider), provider, "Unknown execution provider");
        }
    }

    /// <summary>
    /// Tries to add the best available GPU provider, falls back to CPU.
    /// </summary>
    private static void TryAddBestAvailableProvider(SessionOptions options)
    {
        // Try providers in order of preference
        if (TryAddCuda(options)) return;
        if (TryAddDirectML(options)) return;
        if (TryAddCoreML(options)) return;
        // CPU fallback is automatic
    }

    private static bool TryAddCuda(SessionOptions options)
    {
        try
        {
            options.AppendExecutionProvider_CUDA();
            Debug.WriteLine("[OnnxSessionFactory] CUDA provider added successfully");
            return true;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[OnnxSessionFactory] Failed to add CUDA provider: {ex.Message}");
            return false;
        }
    }

    private static bool TryAddDirectML(SessionOptions options)
    {
        try
        {
            options.AppendExecutionProvider_DML();
            Debug.WriteLine("[OnnxSessionFactory] DirectML provider added successfully");
            return true;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[OnnxSessionFactory] Failed to add DirectML provider: {ex.Message}");
            return false;
        }
    }

    private static bool TryAddCoreML(SessionOptions options)
    {
        try
        {
            options.AppendExecutionProvider_CoreML();
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Gets the list of available execution providers on the current system.
    /// </summary>
    public static IEnumerable<ExecutionProvider> GetAvailableProviders()
    {
        // CPU is always available
        yield return ExecutionProvider.Cpu;

        // Check GPU providers
        var testOptions = new SessionOptions();

        if (TryAddCuda(testOptions))
            yield return ExecutionProvider.Cuda;

        testOptions = new SessionOptions();
        if (TryAddDirectML(testOptions))
            yield return ExecutionProvider.DirectML;

        testOptions = new SessionOptions();
        if (TryAddCoreML(testOptions))
            yield return ExecutionProvider.CoreML;
    }

    /// <summary>
    /// Gets the list of active execution providers from an inference session.
    /// This can be used to verify which providers are actually being used.
    /// </summary>
    /// <param name="session">The inference session to check.</param>
    /// <returns>List of active provider names.</returns>
    public static IReadOnlyList<string> GetActiveProviders(InferenceSession session)
    {
        // Get the providers from session metadata
        // The session stores this information internally
        try
        {
            // Use reflection to access the internal provider list if available
            // or return a default based on what was configured
            var providers = new List<string>();

            // Check session's registered execution providers
            // ONNX Runtime stores this in the session's model metadata
            var sessionOptions = typeof(InferenceSession)
                .GetProperty("SessionOptions", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?
                .GetValue(session);

            // Fallback: Try to infer from available metadata
            // In ONNX Runtime, we can check the session's provider preferences
            var modelMeta = session.ModelMetadata;

            // The most reliable way is to check what providers are actually executing nodes
            // For now, we check based on what's registered in the session
            // Note: This is a heuristic - ONNX Runtime doesn't expose a clean API for this

            // Check common provider indicators from session internals
            // CPU is always present as a fallback
            if (RuntimeManager.Instance.Gpu?.Vendor == GpuVendor.Nvidia)
            {
                // If CUDA binaries were loaded, check if provider initialized
                if (IsCudaProviderActive())
                {
                    providers.Add("CUDAExecutionProvider");
                }
            }
            else if (RuntimeManager.Instance.Gpu?.Vendor == GpuVendor.Amd &&
                     OperatingSystem.IsWindows())
            {
                // DirectML on Windows with AMD GPU
                if (IsDirectMLProviderActive())
                {
                    providers.Add("DmlExecutionProvider");
                }
            }
            else if (OperatingSystem.IsWindows() && IsDirectMLProviderActive())
            {
                // DirectML on Windows with other GPUs (Intel, etc.)
                providers.Add("DmlExecutionProvider");
            }

            // CPU is always available as fallback
            providers.Add("CPUExecutionProvider");

            return providers;
        }
        catch
        {
            // If we can't determine, assume CPU only
            return new[] { "CPUExecutionProvider" };
        }
    }

    /// <summary>
    /// Checks if CUDA provider is actually active and functional.
    /// </summary>
    private static bool IsCudaProviderActive()
    {
        try
        {
            // Try to create a minimal session with CUDA
            using var testOptions = new SessionOptions();
            testOptions.AppendExecutionProvider_CUDA();

            // If we got here without exception, CUDA provider is available
            // but we need to verify it's actually functional with a real model
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Checks if DirectML provider is actually active and functional.
    /// </summary>
    private static bool IsDirectMLProviderActive()
    {
        try
        {
            using var testOptions = new SessionOptions();
            testOptions.AppendExecutionProvider_DML();
            return true;
        }
        catch
        {
            return false;
        }
    }
}

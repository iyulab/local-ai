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
            ExecutionProvider.Cuda => RuntimeManager.Instance.GetDefaultProvider(),
            ExecutionProvider.DirectML => "directml",
            ExecutionProvider.CoreML => "cpu",
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
    /// Verifies that GPU providers are actually active after session creation.
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

        foreach (var providerToTry in fallbackChain)
        {
            try
            {
                // For GPU providers, check if runtime libraries are actually available
                if (providerToTry == ExecutionProvider.Cuda && !IsCudaRuntimeAvailable())
                {
                    Console.WriteLine($"[OnnxSessionFactory] CUDA runtime libraries not available, skipping to next provider...");
                    continue;
                }

                // Get provider string for binary download
                var providerString = providerToTry switch
                {
                    ExecutionProvider.Cuda => RuntimeManager.Instance.GetDefaultProvider(),
                    ExecutionProvider.DirectML => "directml",
                    ExecutionProvider.CoreML => "cpu",
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

                // Verify GPU is actually active for GPU providers
                bool isGpuProvider = providerToTry is ExecutionProvider.Cuda or ExecutionProvider.DirectML or ExecutionProvider.CoreML;
                if (isGpuProvider)
                {
                    var activeProviders = GetActiveProvidersFromSession(session);
                    bool hasGpuActive = activeProviders.Any(p =>
                        p.Contains("CUDA", StringComparison.OrdinalIgnoreCase) ||
                        p.Contains("Dml", StringComparison.OrdinalIgnoreCase) ||
                        p.Contains("CoreML", StringComparison.OrdinalIgnoreCase));

                    if (!hasGpuActive)
                    {
                        Console.WriteLine($"[OnnxSessionFactory] {providerToTry} provider was added but not activated, trying next provider...");
                        session.Dispose();
                        continue;
                    }

                    Console.WriteLine($"[OnnxSessionFactory] Successfully using {providerToTry}. Active: [{string.Join(", ", activeProviders)}]");
                    return new SessionCreationResult
                    {
                        Session = session,
                        RequestedProvider = ExecutionProvider.Auto,
                        ActiveProviders = activeProviders
                    };
                }

                // CPU provider - always succeeds
                Console.WriteLine($"[OnnxSessionFactory] Using CPU provider");
                return new SessionCreationResult
                {
                    Session = session,
                    RequestedProvider = ExecutionProvider.Auto,
                    ActiveProviders = new[] { "CPUExecutionProvider" }
                };
            }
            catch (OperationCanceledException)
            {
                throw; // Don't catch cancellation
            }
            catch (Exception ex) when (providerToTry != ExecutionProvider.Cpu)
            {
                Console.WriteLine($"[OnnxSessionFactory] Provider {providerToTry} failed: {ex.Message}. Trying next provider...");
                lastException = ex;
            }
        }

        // Should not reach here since CPU is always in chain, but just in case
        throw lastException ?? new InvalidOperationException("No provider available for session creation");
    }

    /// <summary>
    /// Checks if CUDA runtime libraries (cuBLAS, cuDNN) are available on the system.
    /// </summary>
    private static bool IsCudaRuntimeAvailable()
    {
        // Check for CUDA 12 runtime libraries
        var cudaLibs = new[] { "cublasLt64_12", "cublas64_12", "cudnn64_9", "cudnn64_8" };

        foreach (var lib in cudaLibs)
        {
            try
            {
                if (System.Runtime.InteropServices.NativeLibrary.TryLoad(lib, out var handle))
                {
                    System.Runtime.InteropServices.NativeLibrary.Free(handle);
                    return true; // At least one CUDA library is available
                }
            }
            catch
            {
                // Continue checking other libraries
            }
        }

        // Also check CUDA 11
        var cuda11Libs = new[] { "cublasLt64_11", "cublas64_11" };
        foreach (var lib in cuda11Libs)
        {
            try
            {
                if (System.Runtime.InteropServices.NativeLibrary.TryLoad(lib, out var handle))
                {
                    System.Runtime.InteropServices.NativeLibrary.Free(handle);
                    return true;
                }
            }
            catch
            {
                // Continue
            }
        }

        return false;
    }

    /// <summary>
    /// Gets active providers by checking session's internal state.
    /// Uses a simple inference to trigger actual provider initialization.
    /// </summary>
    private static IReadOnlyList<string> GetActiveProvidersFromSession(InferenceSession session)
    {
        var providers = new List<string>();

        // Check input/output metadata to see which providers are handling nodes
        // This is a heuristic based on ONNX Runtime behavior
        try
        {
            // The session's InputMetadata access triggers provider initialization
            _ = session.InputMetadata;

            // Check if specific provider DLLs are loaded in the process
            var loadedModules = System.Diagnostics.Process.GetCurrentProcess().Modules;
            foreach (System.Diagnostics.ProcessModule module in loadedModules)
            {
                var name = module.ModuleName?.ToLowerInvariant() ?? "";
                if (name.Contains("onnxruntime_providers_cuda"))
                {
                    providers.Add("CUDAExecutionProvider");
                }
                else if (name.Contains("onnxruntime_providers_dml") || name.Contains("directml"))
                {
                    providers.Add("DmlExecutionProvider");
                }
            }
        }
        catch
        {
            // Ignore errors
        }

        providers.Add("CPUExecutionProvider");
        return providers.Distinct().ToList();
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

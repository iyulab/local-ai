using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Runtime;

namespace LMSupply.Generator.Internal;

/// <summary>
/// Internal class for loading generator models.
/// </summary>
internal static class GeneratorModelLoader
{
    public static async Task<IGeneratorModel> LoadAsync(
        string modelId,
        GeneratorOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Ensure GenAI runtime binaries are available before loading the model
        await EnsureGenAiRuntimeAsync(options.Provider, progress, cancellationToken);

        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir);

        // Look up model in registry to get subfolder preference
        var modelInfo = ModelRegistry.GetModel(modelId);

        // Build preferences from registry info if available
        var preferences = modelInfo?.Subfolder != null
            ? new ModelPreferences { PreferredSubfolder = modelInfo.Subfolder }
            : ModelPreferences.Default;

        // Use discovery-based download for all models
        // This handles dynamic ONNX file names (e.g., phi-3.5-mini-instruct-*.onnx)
        var (basePath, discovery) = await downloader.DownloadWithDiscoveryAsync(
            modelId,
            preferences: preferences,
            progress: progress,
            cancellationToken: cancellationToken);

        // Build the actual model path including subfolder if present
        var modelPath = discovery.Subfolder != null
            ? Path.Combine(basePath, discovery.Subfolder.Replace('/', Path.DirectorySeparatorChar))
            : basePath;

        return await LoadFromPathAsync(modelPath, options, modelId);
    }

    public static async Task<IGeneratorModel> LoadFromPathAsync(
        string modelPath,
        GeneratorOptions options,
        string? modelId = null)
    {
        // Ensure GenAI runtime binaries are available before loading the model
        await EnsureGenAiRuntimeAsync(options.Provider, progress: null, CancellationToken.None);

        modelId ??= Path.GetFileName(modelPath);

        // Determine chat formatter
        var chatFormatter = options.ChatFormat != null
            ? ChatFormatterFactory.CreateByFormat(options.ChatFormat)
            : ChatFormatterFactory.Create(modelId);

        // Create and return the model
        var model = new OnnxGeneratorModel(
            modelId,
            modelPath,
            chatFormatter,
            options);

        return model;
    }

    /// <summary>
    /// Ensures GenAI runtime binaries (onnxruntime-genai) are downloaded for the specified provider.
    /// Also ensures the base onnxruntime binaries are available since genai depends on them.
    /// </summary>
    private static async Task EnsureGenAiRuntimeAsync(
        ExecutionProvider provider,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Initialize RuntimeManager to detect hardware
        await RuntimeManager.Instance.InitializeAsync(cancellationToken);

        // Resolve Auto to actual provider
        var actualProvider = provider == ExecutionProvider.Auto
            ? RuntimeManager.Instance.RecommendedProvider
            : provider;

        // Map provider to string for manifest lookup
        var providerString = actualProvider switch
        {
            ExecutionProvider.Cuda => RuntimeManager.Instance.GetDefaultProvider(), // cuda11 or cuda12
            ExecutionProvider.DirectML => "directml",
            ExecutionProvider.CoreML => "cpu", // CoreML uses CPU binaries
            _ => "cpu"
        };

        // Download base onnxruntime binaries first (genai depends on these)
        await RuntimeManager.Instance.EnsureRuntimeAsync(
            "onnxruntime",
            provider: providerString,
            progress: progress,
            cancellationToken: cancellationToken);

        // Download GenAI runtime binaries
        await RuntimeManager.Instance.EnsureRuntimeAsync(
            "onnxruntime-genai",
            provider: providerString,
            progress: progress,
            cancellationToken: cancellationToken);
    }
}

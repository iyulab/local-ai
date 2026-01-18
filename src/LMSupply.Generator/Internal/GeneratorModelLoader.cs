using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;
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
        // Detect model format from model ID
        var format = ModelFormatDetector.Detect(modelId);

        // Route to appropriate loader based on format
        return format switch
        {
            ModelFormat.Gguf => await LoadGgufAsync(modelId, options, progress, cancellationToken),
            ModelFormat.Onnx => await LoadOnnxAsync(modelId, options, progress, cancellationToken),
            ModelFormat.Unknown => await LoadOnnxAsync(modelId, options, progress, cancellationToken), // fallback
            _ => throw new NotSupportedException($"Unsupported model format: {format}")
        };
    }

    /// <summary>
    /// Loads an ONNX GenAI model from HuggingFace.
    /// </summary>
    private static async Task<IGeneratorModel> LoadOnnxAsync(
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

    /// <summary>
    /// Loads a GGUF model from HuggingFace.
    /// </summary>
    private static async Task<IGeneratorModel> LoadGgufAsync(
        string modelId,
        GeneratorOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();

        // Try to resolve as registry alias first
        var registryInfo = GgufModelRegistry.Resolve(modelId);
        string modelPath;
        string chatFormat;

        if (registryInfo != null)
        {
            // Download from registry
            using var downloader = new GgufModelDownloader(cacheDir);
            modelPath = await downloader.DownloadFromRegistryAsync(
                registryInfo,
                preferredQuantization: null,
                progress: progress,
                cancellationToken: cancellationToken);

            chatFormat = options.ChatFormat ?? registryInfo.ChatFormat;
        }
        else
        {
            // Assume it's a HuggingFace repo ID
            using var downloader = new GgufModelDownloader(cacheDir);
            modelPath = await downloader.DownloadAsync(
                modelId,
                filename: null,
                preferredQuantization: null,
                progress: progress,
                cancellationToken: cancellationToken);

            // Detect chat format from filename
            chatFormat = options.ChatFormat ?? GgufChatFormatDetector.DetectFromFilename(modelPath);
        }

        // Load the model from downloaded path
        var resolvedModelId = registryInfo?.DisplayName ?? modelId;
        var chatFormatter = ChatFormatterFactory.CreateByFormat(chatFormat);

        return await GgufGeneratorModel.LoadAsync(
            resolvedModelId,
            modelPath,
            chatFormatter,
            options,
            progress,
            cancellationToken);
    }

    public static async Task<IGeneratorModel> LoadFromPathAsync(
        string modelPath,
        GeneratorOptions options,
        string? modelId = null)
    {
        // Detect model format from path
        var format = ModelFormatDetector.Detect(modelPath);

        // Route to appropriate loader based on format
        return format switch
        {
            ModelFormat.Gguf => await LoadGgufFromPathAsync(modelPath, options, modelId),
            ModelFormat.Onnx => await LoadOnnxFromPathAsync(modelPath, options, modelId),
            ModelFormat.Unknown => await LoadOnnxFromPathAsync(modelPath, options, modelId), // fallback
            _ => throw new NotSupportedException($"Unsupported model format: {format}")
        };
    }

    /// <summary>
    /// Loads an ONNX GenAI model from a local path.
    /// </summary>
    private static async Task<IGeneratorModel> LoadOnnxFromPathAsync(
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
    /// Loads a GGUF model from a local path.
    /// </summary>
    private static async Task<IGeneratorModel> LoadGgufFromPathAsync(
        string modelPath,
        GeneratorOptions options,
        string? modelId = null)
    {
        modelId ??= Path.GetFileNameWithoutExtension(modelPath);

        // Detect chat format from filename or use provided
        var chatFormat = options.ChatFormat ?? GgufChatFormatDetector.DetectFromFilename(modelPath);
        var chatFormatter = ChatFormatterFactory.CreateByFormat(chatFormat);

        return await GgufGeneratorModel.LoadAsync(
            modelId,
            modelPath,
            chatFormatter,
            options,
            progress: null,
            CancellationToken.None);
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

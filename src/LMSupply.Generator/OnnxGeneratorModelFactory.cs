using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.ChatFormatters;

namespace LMSupply.Generator;

/// <summary>
/// Factory for creating ONNX-based generator models.
/// </summary>
public sealed class OnnxGeneratorModelFactory : IGeneratorModelFactory, IDisposable
{
    private readonly string _cacheDirectory;
    private readonly ExecutionProvider _defaultProvider;
    private readonly HuggingFaceDownloader _downloader;
    private bool _disposed;

    private const string DefaultRevision = "main";

    /// <summary>
    /// Files required for ONNX GenAI models.
    /// </summary>
    private static readonly string[] GenAiModelFiles =
    [
        "genai_config.json",
        "model.onnx",
        "model.onnx.data",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json"
    ];

    /// <summary>
    /// Creates a new factory with default settings.
    /// </summary>
    public OnnxGeneratorModelFactory()
        : this(GetDefaultCacheDirectory(), ExecutionProvider.Auto)
    {
    }

    /// <summary>
    /// Creates a new factory with specified settings.
    /// </summary>
    /// <param name="cacheDirectory">Directory for model cache.</param>
    /// <param name="defaultProvider">Default execution provider.</param>
    public OnnxGeneratorModelFactory(string cacheDirectory, ExecutionProvider defaultProvider)
    {
        _cacheDirectory = cacheDirectory ?? throw new ArgumentNullException(nameof(cacheDirectory));
        _defaultProvider = defaultProvider;
        _downloader = new HuggingFaceDownloader(cacheDirectory);
    }

    /// <inheritdoc />
    public async Task<IGeneratorModel> LoadAsync(
        string modelId,
        GeneratorOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new GeneratorOptions();

        var modelPath = await ResolveModelPathAsync(modelId, cancellationToken);
        var chatFormatter = ResolveChatFormatter(modelId, options.ChatFormat);

        // Merge default provider if not specified
        if (options.Provider == ExecutionProvider.Auto && _defaultProvider != ExecutionProvider.Auto)
        {
            options = new GeneratorOptions
            {
                CacheDirectory = options.CacheDirectory ?? _cacheDirectory,
                Provider = _defaultProvider,
                ChatFormat = options.ChatFormat,
                Verbose = options.Verbose,
                MaxContextLength = options.MaxContextLength,
                MaxConcurrentRequests = options.MaxConcurrentRequests
            };
        }

        return new Internal.OnnxGeneratorModel(modelId, modelPath, chatFormatter, options);
    }

    /// <inheritdoc />
    public bool IsModelAvailable(string modelId)
    {
        var snapshotPath = GetModelCachePath(modelId);

        // Check if model exists directly in snapshot
        if (IsValidModelDirectory(snapshotPath))
            return true;

        // Check for variant subdirectories
        return FindVariantSubfolder(snapshotPath) != null;
    }

    /// <inheritdoc />
    public async Task DownloadModelAsync(
        string modelId,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (IsModelAvailable(modelId))
        {
            progress?.Report(new ModelDownloadProgress(100, 100, null));
            return;
        }

        // Determine the best variant subfolder based on provider
        var subfolder = GetVariantSubfolder(modelId);

        // Create progress adapter
        IProgress<DownloadProgress>? downloadProgress = null;
        if (progress != null)
        {
            downloadProgress = new Progress<DownloadProgress>(p =>
            {
                progress.Report(new ModelDownloadProgress(
                    p.BytesDownloaded,
                    p.TotalBytes,
                    p.FileName));
            });
        }

        // Download using HuggingFace downloader
        await _downloader.DownloadModelAsync(
            modelId,
            files: GenAiModelFiles,
            revision: "main",
            subfolder: subfolder,
            progress: downloadProgress,
            cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Determines the variant subfolder based on model registry info and provider.
    /// </summary>
    private string? GetVariantSubfolder(string modelId)
    {
        // First, check ModelRegistry for explicit subfolder configuration
        var modelInfo = ModelRegistry.GetModel(modelId);
        if (modelInfo?.Subfolder != null)
        {
            // Registry may have provider-neutral subfolder (e.g., "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4")
            // Adapt based on provider if needed
            return AdaptSubfolderForProvider(modelInfo.Subfolder, _defaultProvider);
        }

        // Fallback for unregistered models: infer from model ID patterns
        if (modelId.Contains("phi", StringComparison.OrdinalIgnoreCase) && modelId.Contains("onnx", StringComparison.OrdinalIgnoreCase))
        {
            return _defaultProvider switch
            {
                ExecutionProvider.Cuda => "cuda/cuda-int4-rtn-block-32",
                ExecutionProvider.DirectML => "directml/directml-int4-awq-block-128",
                _ => "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
            };
        }

        return null;
    }

    /// <summary>
    /// Adapts a registry subfolder for the target execution provider.
    /// </summary>
    private static string AdaptSubfolderForProvider(string subfolder, ExecutionProvider provider)
    {
        // If the subfolder is already provider-specific, use as-is
        if (subfolder.Contains("cuda", StringComparison.OrdinalIgnoreCase) ||
            subfolder.Contains("directml", StringComparison.OrdinalIgnoreCase) ||
            subfolder.Contains("cpu", StringComparison.OrdinalIgnoreCase))
        {
            return provider switch
            {
                // For CUDA, try to find cuda variant
                ExecutionProvider.Cuda when subfolder.Contains("cpu", StringComparison.OrdinalIgnoreCase)
                    => subfolder.Replace("cpu_and_mobile", "cuda").Replace("cpu-", "cuda-"),
                // For DirectML, try to find directml variant
                ExecutionProvider.DirectML when subfolder.Contains("cpu", StringComparison.OrdinalIgnoreCase)
                    => subfolder.Replace("cpu_and_mobile", "directml").Replace("cpu-", "directml-"),
                _ => subfolder
            };
        }

        return subfolder;
    }

    /// <summary>
    /// Gets the cache path for a model, following HuggingFace cache structure.
    /// </summary>
    public string GetModelCachePath(string modelId)
    {
        // Use CacheManager for consistent path resolution
        // Structure: models--{org}--{name}/snapshots/{revision}
        return CacheManager.GetModelDirectory(_cacheDirectory, modelId, DefaultRevision);
    }

    /// <summary>
    /// Lists all locally available generator models.
    /// </summary>
    public IReadOnlyList<string> GetAvailableModels()
    {
        // Use CacheManager's type detection for consistent discovery
        return CacheManager.GetCachedModelsByType(_cacheDirectory, ModelType.Generator)
            .Select(m => m.RepoId)
            .ToList();
    }

    private async Task<string> ResolveModelPathAsync(string modelId, CancellationToken cancellationToken)
    {
        // Get proper HuggingFace cache path (models--{org}--{name}/snapshots/{revision})
        var snapshotPath = GetModelCachePath(modelId);

        // Check if model exists directly in snapshot
        if (IsValidModelDirectory(snapshotPath))
            return snapshotPath;

        // Check for variant subdirectories within snapshot (cpu-int4, cuda-int4, etc.)
        var foundPath = FindVariantSubfolder(snapshotPath);
        if (foundPath != null)
            return foundPath;

        // Model not found - attempt download
        await DownloadModelAsync(modelId, null, cancellationToken);

        // After download, check again
        if (IsValidModelDirectory(snapshotPath))
            return snapshotPath;

        // Check variants again after download
        foundPath = FindVariantSubfolder(snapshotPath);
        if (foundPath != null)
            return foundPath;

        throw new FileNotFoundException($"Model '{modelId}' not found at {snapshotPath}");
    }

    /// <summary>
    /// Finds a valid variant subfolder within the model directory.
    /// </summary>
    private string? FindVariantSubfolder(string basePath)
    {
        if (!Directory.Exists(basePath))
            return null;

        // Provider-specific variant prefixes in priority order
        var variantPatterns = _defaultProvider switch
        {
            ExecutionProvider.Cuda => new[] { "cuda", "gpu", "cpu" },
            ExecutionProvider.DirectML => new[] { "directml", "gpu", "cpu" },
            _ => new[] { "cpu", "gpu" }
        };

        foreach (var pattern in variantPatterns)
        {
            // Find subdirectories matching the pattern
            foreach (var subdir in Directory.GetDirectories(basePath))
            {
                var dirName = Path.GetFileName(subdir);
                if (dirName.StartsWith(pattern, StringComparison.OrdinalIgnoreCase) && IsValidModelDirectory(subdir))
                    return subdir;
            }
        }

        // Fallback: check any subdirectory that's a valid model
        foreach (var subdir in Directory.GetDirectories(basePath))
        {
            if (IsValidModelDirectory(subdir))
                return subdir;
        }

        return null;
    }

    private static bool IsValidModelDirectory(string path)
    {
        if (!Directory.Exists(path))
            return false;

        return File.Exists(Path.Combine(path, "genai_config.json"))
            || File.Exists(Path.Combine(path, "model.onnx"))
            || File.Exists(Path.Combine(path, "model.onnx.data"));
    }

    private static IChatFormatter ResolveChatFormatter(string modelId, string? explicitFormat)
    {
        if (!string.IsNullOrEmpty(explicitFormat))
            return ChatFormatterFactory.Create(explicitFormat);

        // Try to get from registry
        var modelInfo = ModelRegistry.GetModel(modelId);
        if (modelInfo != null)
            return ChatFormatterFactory.Create(modelInfo.ChatFormat);

        // Fall back to auto-detection from model name
        return ChatFormatterFactory.Create(modelId);
    }

    private static string GetDefaultCacheDirectory()
    {
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "huggingface", "hub");
    }

    /// <summary>
    /// Releases resources used by the factory.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _downloader.Dispose();
        _disposed = true;
    }
}

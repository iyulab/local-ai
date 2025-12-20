using LMSupply.Download;
using LMSupply.ImageGenerator.Models;

namespace LMSupply.ImageGenerator;

/// <summary>
/// Entry point for loading and creating local image generator models.
/// </summary>
public static class LocalImageGenerator
{
    /// <summary>
    /// Required files for LCM model loading.
    /// </summary>
    private static readonly string[] RequiredFiles =
    [
        // Text encoder
        "text_encoder/model.onnx",
        // UNet
        "unet/model.onnx",
        // VAE decoder
        "vae_decoder/model.onnx",
        // Tokenizer files
        "tokenizer/vocab.json",
        "tokenizer/merges.txt"
    ];

    /// <summary>
    /// Loads an image generator model.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Model identifier. Can be:
    /// - An alias: "default", "fast", "quality"
    /// - A HuggingFace repo ID: "TheyCallMeHex/LCM-Dreamshaper-V7-ONNX"
    /// - A local directory path containing ONNX model files
    /// </param>
    /// <param name="options">Optional model loading options.</param>
    /// <param name="progress">Optional progress reporter for model download.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Loaded image generator model.</returns>
    /// <example>
    /// <code>
    /// // Load default model
    /// await using var generator = await LocalImageGenerator.LoadAsync("default");
    ///
    /// // Generate an image
    /// var result = await generator.GenerateAsync("A sunset over mountains");
    /// await result.SaveAsync("output.png");
    /// </code>
    /// </example>
    public static async Task<IImageGeneratorModel> LoadAsync(
        string modelIdOrPath,
        ImageGeneratorOptions? options = null,
        IProgress<float>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);

        options ??= new ImageGeneratorOptions();

        // Resolve model alias to actual repo ID
        var modelDefinition = WellKnownImageModels.Resolve(modelIdOrPath);

        // Determine if this is a local path or needs download
        string modelPath;

        if (IsLocalPath(modelIdOrPath))
        {
            // Use local path directly
            modelPath = modelIdOrPath;
            if (!Directory.Exists(modelPath))
            {
                throw new DirectoryNotFoundException($"Model directory not found: {modelPath}");
            }
        }
        else
        {
            // Download from HuggingFace
            modelPath = await DownloadModelAsync(
                modelDefinition.RepoId,
                options,
                progress,
                cancellationToken);
        }

        // Load the model
        return await OnnxImageGeneratorModel.LoadAsync(
            modelDefinition,
            modelPath,
            options,
            cancellationToken);
    }

    /// <summary>
    /// Gets all available model aliases.
    /// </summary>
    /// <returns>Collection of valid model aliases.</returns>
    public static IReadOnlyCollection<string> GetAvailableAliases() =>
        WellKnownImageModels.GetAliases();

    /// <summary>
    /// Checks if the given identifier is a local file path.
    /// </summary>
    private static bool IsLocalPath(string modelIdOrPath)
    {
        // Check for absolute path or relative path indicators
        return Path.IsPathRooted(modelIdOrPath) ||
               modelIdOrPath.StartsWith("./", StringComparison.Ordinal) ||
               modelIdOrPath.StartsWith(".\\", StringComparison.Ordinal) ||
               modelIdOrPath.StartsWith("../", StringComparison.Ordinal) ||
               modelIdOrPath.StartsWith("..\\", StringComparison.Ordinal) ||
               Directory.Exists(modelIdOrPath);
    }

    /// <summary>
    /// Downloads a model from HuggingFace.
    /// </summary>
    private static async Task<string> DownloadModelAsync(
        string repoId,
        ImageGeneratorOptions options,
        IProgress<float>? progress,
        CancellationToken cancellationToken)
    {
        using var downloader = new HuggingFaceDownloader(options.CacheDirectory);

        // Track download progress
        var progressAdapter = progress != null
            ? new Progress<DownloadProgress>(p =>
            {
                if (p.TotalBytes > 0)
                {
                    progress.Report((float)p.BytesDownloaded / p.TotalBytes);
                }
            })
            : null;

        // Download model files
        var modelPath = await downloader.DownloadModelAsync(
            repoId,
            files: GetModelFiles(repoId),
            revision: "main",
            subfolder: null,
            progress: progressAdapter,
            cancellationToken: cancellationToken);

        return modelPath;
    }

    /// <summary>
    /// Gets the list of files to download for a model.
    /// </summary>
    private static IEnumerable<string> GetModelFiles(string repoId)
    {
        // Standard LCM ONNX model structure
        return
        [
            // ONNX models
            "text_encoder/model.onnx",
            "unet/model.onnx",
            "vae_decoder/model.onnx",
            // Tokenizer
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
            // Scheduler config
            "scheduler/scheduler_config.json",
            // Model config
            "model_index.json"
        ];
    }
}

using LMSupply.Generator.Abstractions;

namespace LMSupply.Generator;

/// <summary>
/// Factory class for creating local text generation models.
/// </summary>
public static class LocalGenerator
{
    /// <summary>
    /// Default model to use when no model is specified.
    /// Microsoft Phi-4 Mini (MIT license), 3.8B params, 16K context.
    /// </summary>
    public const string DefaultModel = "microsoft/Phi-4-mini-instruct-onnx";

    /// <summary>
    /// Loads a text generator from a HuggingFace model repository.
    /// Supports "auto" alias which selects optimal model based on hardware.
    /// </summary>
    /// <param name="modelId">The HuggingFace model identifier (e.g., "microsoft/Phi-3.5-mini-instruct-onnx") or "auto".</param>
    /// <param name="options">Model loading options.</param>
    /// <param name="progress">Progress callback for model downloading.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A text generator instance.</returns>
    /// <remarks>
    /// When "auto" is specified, the model is selected based on hardware:
    /// - Low tier (GPU &lt;4GB): Llama-3.2-1B (1B params)
    /// - Medium tier (GPU 4-8GB): Phi-3.5-mini (3.8B params)
    /// - High tier (GPU 8-16GB): Phi-4-mini (3.8B, 16K context)
    /// - Ultra tier (GPU 16GB+): Phi-4 (14B params)
    /// </remarks>
    public static Task<IGeneratorModel> LoadAsync(
        string modelId,
        GeneratorOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        options ??= new GeneratorOptions();

        // Handle "auto" alias - select optimal model based on hardware
        if (modelId.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            var autoModel = ModelRegistry.GetDefaultModel();
            modelId = autoModel.ModelId;
        }

        return Internal.GeneratorModelLoader.LoadAsync(modelId, options, progress, cancellationToken);
    }

    /// <summary>
    /// Loads a text generator from a local model directory.
    /// </summary>
    /// <param name="modelPath">The path to the local model directory.</param>
    /// <param name="options">Model loading options.</param>
    /// <returns>A text generator instance.</returns>
    public static Task<IGeneratorModel> LoadFromPathAsync(
        string modelPath,
        GeneratorOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        if (!Directory.Exists(modelPath))
        {
            throw new DirectoryNotFoundException($"Model directory not found: {modelPath}");
        }

        options ??= new GeneratorOptions();

        return Internal.GeneratorModelLoader.LoadFromPathAsync(modelPath, options);
    }

    /// <summary>
    /// Loads a text generator using the default model.
    /// </summary>
    /// <param name="options">Model loading options.</param>
    /// <param name="progress">Progress callback for model downloading.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A text generator instance.</returns>
    public static Task<IGeneratorModel> LoadDefaultAsync(
        GeneratorOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return LoadAsync(DefaultModel, options, progress, cancellationToken);
    }
}

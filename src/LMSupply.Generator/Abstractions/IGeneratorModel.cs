namespace LMSupply.Generator.Abstractions;

/// <summary>
/// Represents a loaded text generation model.
/// </summary>
public interface IGeneratorModel : ITextGenerator
{
    /// <summary>
    /// Gets the maximum context length supported by the model.
    /// </summary>
    int MaxContextLength { get; }

    /// <summary>
    /// Gets the chat formatter for this model.
    /// </summary>
    IChatFormatter ChatFormatter { get; }

    /// <summary>
    /// Gets whether GPU acceleration is being used for inference.
    /// </summary>
    bool IsGpuActive { get; }

    /// <summary>
    /// Gets the list of active execution providers.
    /// </summary>
    IReadOnlyList<string> ActiveProviders { get; }

    /// <summary>
    /// Gets the execution provider that was requested.
    /// </summary>
    ExecutionProvider RequestedProvider { get; }

    /// <summary>
    /// Gets the estimated memory usage of this model in bytes.
    /// Uses MemoryEstimator to calculate based on model parameters, context length, and quantization.
    /// </summary>
    long? EstimatedMemoryBytes { get; }

    /// <summary>
    /// Gets information about the model.
    /// </summary>
    GeneratorModelInfo GetModelInfo();
}

/// <summary>
/// Information about a generator model.
/// </summary>
/// <param name="ModelId">The model identifier.</param>
/// <param name="ModelPath">The local path to the model files.</param>
/// <param name="MaxContextLength">Maximum context length.</param>
/// <param name="ChatFormat">The chat format name.</param>
/// <param name="ExecutionProvider">The execution provider being used.</param>
public readonly record struct GeneratorModelInfo(
    string ModelId,
    string ModelPath,
    int MaxContextLength,
    string ChatFormat,
    string ExecutionProvider) : IModelInfoBase
{
    /// <summary>
    /// Gets the model identifier (IModelInfoBase.Id).
    /// </summary>
    string IModelInfoBase.Id => ModelId;

    /// <summary>
    /// Gets the model alias (same as ModelId for Generator).
    /// </summary>
    string IModelInfoBase.Alias => ModelId;

    /// <summary>
    /// Gets the model description.
    /// </summary>
    string? IModelInfoBase.Description => $"{ChatFormat} model at {ModelPath}";
}

using LMSupply.Core;

namespace LMSupply.ImageGenerator.Models;

/// <summary>
/// Information about a loaded image generator model.
/// </summary>
public sealed class ImageGeneratorModelInfo : IModelInfoBase
{
    /// <summary>
    /// Model identifier (HuggingFace repo ID or local path).
    /// </summary>
    public required string ModelId { get; init; }

    // IModelInfoBase explicit implementation
    string IModelInfoBase.Id => ModelId;
    string IModelInfoBase.Alias => ModelName ?? ModelId;

    /// <summary>
    /// Friendly name of the model.
    /// </summary>
    public string? ModelName { get; init; }

    /// <summary>
    /// Gets the model description.
    /// </summary>
    public string? Description => $"{Architecture} image generator";

    /// <summary>
    /// Model architecture type.
    /// </summary>
    public required string Architecture { get; init; }

    /// <summary>
    /// Execution provider being used.
    /// </summary>
    public required ExecutionProvider Provider { get; init; }

    /// <summary>
    /// Whether FP16 precision is being used.
    /// </summary>
    public required bool IsFp16 { get; init; }

    /// <summary>
    /// Default image width for this model.
    /// </summary>
    public required int DefaultWidth { get; init; }

    /// <summary>
    /// Default image height for this model.
    /// </summary>
    public required int DefaultHeight { get; init; }

    /// <summary>
    /// Recommended number of inference steps.
    /// </summary>
    public required int RecommendedSteps { get; init; }

    /// <summary>
    /// Recommended guidance scale.
    /// </summary>
    public required float RecommendedGuidanceScale { get; init; }

    /// <summary>
    /// Total size of model files in bytes.
    /// </summary>
    public long? ModelSizeBytes { get; init; }

    /// <summary>
    /// Path to the model directory on disk.
    /// </summary>
    public string? ModelPath { get; init; }
}

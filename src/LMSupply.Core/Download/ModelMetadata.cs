namespace LMSupply.Download;

/// <summary>
/// Metadata about a model from HuggingFace Hub API.
/// </summary>
public sealed record ModelMetadata
{
    /// <summary>
    /// Model description from the model card.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>
    /// License identifier (e.g., "mit", "apache-2.0").
    /// </summary>
    public string? License { get; init; }

    /// <summary>
    /// Author or organization name.
    /// </summary>
    public string? Author { get; init; }

    /// <summary>
    /// Total download count from HuggingFace Hub.
    /// </summary>
    public long Downloads { get; init; }

    /// <summary>
    /// Number of likes on HuggingFace Hub.
    /// </summary>
    public long Likes { get; init; }

    /// <summary>
    /// Tags associated with the model (e.g., "pytorch", "onnx", "en").
    /// </summary>
    public IReadOnlyList<string> Tags { get; init; } = [];

    /// <summary>
    /// Pipeline tag indicating primary use case (e.g., "feature-extraction", "text-generation").
    /// </summary>
    public string? PipelineTag { get; init; }

    /// <summary>
    /// Library name (e.g., "sentence-transformers", "transformers").
    /// </summary>
    public string? LibraryName { get; init; }

    /// <summary>
    /// Last modification timestamp on HuggingFace Hub.
    /// </summary>
    public DateTime? LastModifiedOnHub { get; init; }

    /// <summary>
    /// Timestamp when this metadata was fetched.
    /// </summary>
    public DateTime FetchedAt { get; init; } = DateTime.UtcNow;

    /// <summary>
    /// Whether the model is gated (requires acceptance of terms).
    /// </summary>
    public bool IsGated { get; init; }

    /// <summary>
    /// Whether the model is private.
    /// </summary>
    public bool IsPrivate { get; init; }
}

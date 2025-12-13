namespace LocalAI.Captioner;

/// <summary>
/// Configuration options for the image captioner.
/// </summary>
public sealed class CaptionerOptions
{
    /// <summary>
    /// Maximum number of tokens to generate in the caption.
    /// Default is 50.
    /// </summary>
    public int MaxLength { get; set; } = 50;

    /// <summary>
    /// Number of beams for beam search decoding.
    /// 1 = greedy decoding (default), higher values explore more candidates.
    /// </summary>
    public int NumBeams { get; set; } = 1;

    /// <summary>
    /// Temperature for sampling. Lower values make output more deterministic.
    /// Default is 1.0 (no temperature scaling).
    /// </summary>
    public float Temperature { get; set; } = 1.0f;

    /// <summary>
    /// Optional text prompt to start caption generation.
    /// </summary>
    public string? Prompt { get; set; }

    /// <summary>
    /// Execution provider for ONNX Runtime.
    /// Default is Auto (automatically selects best available).
    /// </summary>
    public ExecutionProvider Provider { get; set; } = ExecutionProvider.Auto;

    /// <summary>
    /// Custom cache directory for models.
    /// If null, uses the default HuggingFace cache directory.
    /// </summary>
    public string? CacheDirectory { get; set; }
}

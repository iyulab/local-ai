namespace LMSupply;

/// <summary>
/// Base class for all LMSupply model options.
/// Provides common configuration properties shared across all packages.
/// </summary>
public abstract class LMSupplyOptionsBase
{
    /// <summary>
    /// Gets or sets the custom cache directory for model files.
    /// <para>Default: null (uses HuggingFace standard cache location: ~/.cache/huggingface/hub)</para>
    /// </summary>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Gets or sets the execution provider for inference.
    /// <para>Default: <see cref="ExecutionProvider.Auto"/> (automatically selects the best available provider)</para>
    /// </summary>
    public ExecutionProvider Provider { get; set; } = ExecutionProvider.Auto;

    /// <summary>
    /// Gets or sets the number of threads to use for inference.
    /// <para>Default: null (uses ONNX Runtime default, typically all available cores)</para>
    /// <para>Set to a specific value to limit CPU usage, useful for:</para>
    /// <list type="bullet">
    ///   <item>Running multiple models concurrently</item>
    ///   <item>Leaving CPU headroom for other tasks</item>
    ///   <item>Reducing power consumption</item>
    /// </list>
    /// </summary>
    /// <remarks>
    /// This setting primarily affects CPU inference. GPU inference may have different threading behavior
    /// controlled by the GPU driver.
    /// </remarks>
    public int? ThreadCount { get; set; }
}

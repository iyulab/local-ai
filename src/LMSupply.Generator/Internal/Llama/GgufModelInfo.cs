namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// Metadata for a registered GGUF model.
/// </summary>
public sealed record GgufModelInfo
{
    /// <summary>
    /// HuggingFace repository ID (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF").
    /// </summary>
    public required string RepoId { get; init; }

    /// <summary>
    /// Human-friendly display name.
    /// </summary>
    public required string DisplayName { get; init; }

    /// <summary>
    /// Default GGUF file name to download if not specified.
    /// </summary>
    public required string DefaultFile { get; init; }

    /// <summary>
    /// Chat template format identifier (e.g., "llama3", "chatml", "gemma").
    /// </summary>
    public required string ChatFormat { get; init; }

    /// <summary>
    /// Maximum supported context length.
    /// </summary>
    public required int ContextLength { get; init; }

    /// <summary>
    /// Approximate parameter count (for memory estimation).
    /// </summary>
    public long ParameterCount { get; init; }

    /// <summary>
    /// License tier classification.
    /// </summary>
    public required LicenseTier License { get; init; }

    /// <summary>
    /// License name (e.g., "MIT", "Llama 3.2 Community License").
    /// </summary>
    public string? LicenseName { get; init; }

    /// <summary>
    /// Description of usage restrictions, if any.
    /// </summary>
    public string? LicenseRestrictions { get; init; }

    /// <summary>
    /// Optional subfolder within the repository.
    /// </summary>
    public string? Subfolder { get; init; }
}

/// <summary>
/// GGUF quantization types ordered by quality (higher = better quality, larger size).
/// </summary>
public enum GgufQuantization
{
    /// <summary>2-bit quantization (smallest, lowest quality)</summary>
    Q2_K = 0,

    /// <summary>3-bit quantization</summary>
    Q3_K_S = 10,
    Q3_K_M = 11,
    Q3_K_L = 12,

    /// <summary>4-bit quantization (recommended balance)</summary>
    Q4_K_S = 20,
    Q4_K_M = 21,  // Default recommended
    Q4_0 = 22,
    Q4_1 = 23,

    /// <summary>5-bit quantization</summary>
    Q5_K_S = 30,
    Q5_K_M = 31,
    Q5_0 = 32,
    Q5_1 = 33,

    /// <summary>6-bit quantization</summary>
    Q6_K = 40,

    /// <summary>8-bit quantization (highest quality)</summary>
    Q8_0 = 50,

    /// <summary>16-bit float (full precision)</summary>
    F16 = 100
}

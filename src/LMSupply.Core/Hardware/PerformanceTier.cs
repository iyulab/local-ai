namespace LMSupply.Hardware;

/// <summary>
/// Performance tier classification based on available hardware resources.
/// Used for adaptive model selection across all domains.
/// </summary>
public enum PerformanceTier
{
    /// <summary>
    /// Low-end hardware: CPU only or GPU with less than 4GB VRAM.
    /// Suitable for lightweight models optimized for speed.
    /// </summary>
    Low = 0,

    /// <summary>
    /// Mid-range hardware: GPU with 4-8GB VRAM or CPU with 16GB+ system RAM.
    /// Suitable for balanced quality/performance models.
    /// </summary>
    Medium = 1,

    /// <summary>
    /// High-end hardware: GPU with 8-16GB VRAM.
    /// Suitable for high-quality models with extended context.
    /// </summary>
    High = 2,

    /// <summary>
    /// Ultra high-end hardware: GPU with 16GB+ VRAM (workstation/server class).
    /// Suitable for largest models with maximum quality.
    /// </summary>
    Ultra = 3
}

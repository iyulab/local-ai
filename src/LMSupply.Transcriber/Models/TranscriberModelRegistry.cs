using LMSupply.Hardware;

namespace LMSupply.Transcriber.Models;

/// <summary>
/// Registry for managing transcriber model configurations.
/// </summary>
public sealed class TranscriberModelRegistry
{
    private readonly Dictionary<string, TranscriberModelInfo> _models = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, TranscriberModelInfo> _byId = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Gets the default registry instance with pre-configured models.
    /// </summary>
    public static TranscriberModelRegistry Default { get; } = CreateDefault();

    private TranscriberModelRegistry() { }

    private static TranscriberModelRegistry CreateDefault()
    {
        var registry = new TranscriberModelRegistry();
        foreach (var model in DefaultModels.All)
        {
            registry.Register(model);
        }
        return registry;
    }

    /// <summary>
    /// Registers a model configuration.
    /// </summary>
    /// <param name="info">The model information to register.</param>
    public void Register(TranscriberModelInfo info)
    {
        _models[info.Alias] = info;
        _byId[info.Id] = info;
    }

    /// <summary>
    /// Tries to get model info by alias or ID.
    /// Supports "auto" alias which selects optimal model based on hardware.
    /// </summary>
    /// <param name="aliasOrId">The alias, HuggingFace model ID, or "auto".</param>
    /// <param name="info">The model information if found.</param>
    /// <returns>True if found, false otherwise.</returns>
    public bool TryGet(string aliasOrId, out TranscriberModelInfo? info)
    {
        // Handle "auto" alias - select optimal model based on hardware
        if (aliasOrId.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            info = GetAutoModel();
            return true;
        }

        if (_models.TryGetValue(aliasOrId, out info))
            return true;

        if (_byId.TryGetValue(aliasOrId, out info))
            return true;

        info = null;
        return false;
    }

    /// <summary>
    /// Gets the optimal model based on current hardware profile.
    /// Uses PerformanceTier to select appropriate model size.
    /// </summary>
    /// <remarks>
    /// Tier mapping:
    /// - Low:    Whisper Tiny (39M params) - ultra-fast
    /// - Medium: Whisper Base (74M params) - balanced
    /// - High:   Whisper Small (244M params) - quality
    /// - Ultra:  Whisper Large V3 Turbo (809M params) - highest quality
    /// </remarks>
    public static TranscriberModelInfo GetAutoModel()
    {
        var tier = HardwareProfile.Current.Tier;

        return tier switch
        {
            PerformanceTier.Ultra => DefaultModels.WhisperLargeV3Turbo,
            PerformanceTier.High => DefaultModels.WhisperSmall,
            PerformanceTier.Medium => DefaultModels.WhisperBase,
            _ => DefaultModels.WhisperTiny
        };
    }

    /// <summary>
    /// Gets all registered aliases including "auto".
    /// </summary>
    /// <returns>Collection of model aliases.</returns>
    public IEnumerable<string> GetAliases()
    {
        yield return "auto";
        foreach (var alias in _models.Keys)
        {
            yield return alias;
        }
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public IEnumerable<TranscriberModelInfo> GetAll() => _models.Values;
}

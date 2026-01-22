using LMSupply.Hardware;

namespace LMSupply.Synthesizer.Models;

/// <summary>
/// Registry for managing synthesizer model configurations.
/// </summary>
public sealed class SynthesizerModelRegistry
{
    private readonly Dictionary<string, SynthesizerModelInfo> _models = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, SynthesizerModelInfo> _byId = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Gets the default registry instance with pre-configured models.
    /// </summary>
    public static SynthesizerModelRegistry Default { get; } = CreateDefault();

    private SynthesizerModelRegistry() { }

    private static SynthesizerModelRegistry CreateDefault()
    {
        var registry = new SynthesizerModelRegistry();
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
    public void Register(SynthesizerModelInfo info)
    {
        _models[info.Alias] = info;
        _byId[info.Id] = info;
    }

    /// <summary>
    /// Tries to get model info by alias or ID.
    /// Supports "auto" alias which selects the default high-quality model.
    /// </summary>
    /// <param name="aliasOrId">The alias, HuggingFace model ID, or "auto".</param>
    /// <param name="info">The model information if found.</param>
    /// <returns>True if found, false otherwise.</returns>
    public bool TryGet(string aliasOrId, out SynthesizerModelInfo? info)
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
    /// Uses PerformanceTier to select appropriate model quality.
    /// </summary>
    /// <remarks>
    /// For TTS, all Piper models are similar in size (~64MB), so we select based on quality:
    /// - Low:    Ryan (fast, low quality) - 16MB
    /// - Medium: Lessac (default, medium quality) - 64MB
    /// - High:   Amy (quality, high quality) - 64MB
    /// - Ultra:  Amy (quality, high quality) - 64MB
    /// </remarks>
    public static SynthesizerModelInfo GetAutoModel()
    {
        var tier = HardwareProfile.Current.Tier;

        return tier switch
        {
            PerformanceTier.Ultra or PerformanceTier.High => DefaultModels.EnUsAmy,
            PerformanceTier.Medium => DefaultModels.EnUsLessac,
            _ => DefaultModels.EnUsRyan
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
    public IEnumerable<SynthesizerModelInfo> GetAll() => _models.Values;
}

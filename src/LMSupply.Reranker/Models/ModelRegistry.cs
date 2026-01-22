using LMSupply.Hardware;

namespace LMSupply.Reranker.Models;

/// <summary>
/// Registry for looking up model information by ID or alias.
/// </summary>
public sealed class ModelRegistry
{
    private readonly Dictionary<string, ModelInfo> _modelsByAlias;
    private readonly Dictionary<string, ModelInfo> _modelsById;
    private readonly Dictionary<string, ModelInfo> _modelsByName;

    /// <summary>
    /// Gets the default registry instance with built-in models.
    /// </summary>
    public static ModelRegistry Default { get; } = new(DefaultModels.All);

    /// <summary>
    /// Initializes a new registry with the specified models.
    /// </summary>
    /// <param name="models">Models to register.</param>
    public ModelRegistry(IEnumerable<ModelInfo> models)
    {
        ArgumentNullException.ThrowIfNull(models);

        var modelList = models.ToList();
        _modelsByAlias = new Dictionary<string, ModelInfo>(StringComparer.OrdinalIgnoreCase);
        _modelsById = new Dictionary<string, ModelInfo>(StringComparer.OrdinalIgnoreCase);
        _modelsByName = new Dictionary<string, ModelInfo>(StringComparer.OrdinalIgnoreCase);

        foreach (var model in modelList)
        {
            _modelsByAlias[model.Alias] = model;
            _modelsById[model.Id] = model;

            // Also index by model name (part after "/" in HuggingFace IDs)
            // e.g., "BAAI/bge-reranker-base" -> "bge-reranker-base"
            if (model.Id.Contains('/'))
            {
                var modelName = model.Id.Split('/').Last();
                _modelsByName.TryAdd(modelName, model);
            }
        }
    }

    /// <summary>
    /// Resolves a model identifier to its full information.
    /// Supports "auto" alias which selects optimal model based on hardware.
    /// </summary>
    /// <param name="modelIdOrAlias">Model ID, alias, local path, or "auto".</param>
    /// <returns>The model information.</returns>
    /// <exception cref="ModelNotFoundException">Thrown when model is not found.</exception>
    public ModelInfo Resolve(string modelIdOrAlias)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrAlias);

        // Handle "auto" alias - select optimal model based on hardware
        if (modelIdOrAlias.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            return GetAutoModel();
        }

        // Check if it's a local path
        if (IsLocalPath(modelIdOrAlias))
        {
            return CreateLocalModelInfo(modelIdOrAlias);
        }

        // Try alias first
        if (_modelsByAlias.TryGetValue(modelIdOrAlias, out var modelByAlias))
        {
            return modelByAlias;
        }

        // Try full ID
        if (_modelsById.TryGetValue(modelIdOrAlias, out var modelById))
        {
            return modelById;
        }

        // Try model name (e.g., "bge-reranker-base" matches "BAAI/bge-reranker-base")
        if (_modelsByName.TryGetValue(modelIdOrAlias, out var modelByName))
        {
            return modelByName;
        }

        // Assume it's a HuggingFace ID not in our registry
        if (modelIdOrAlias.Contains('/'))
        {
            return CreateHuggingFaceModelInfo(modelIdOrAlias);
        }

        throw new ModelNotFoundException(
            $"Model '{modelIdOrAlias}' not found. Use a built-in alias (default, quality, fast, multilingual), " +
            "a model name (e.g., bge-reranker-base), a HuggingFace model ID (org/model), or a local file path.",
            modelIdOrAlias);
    }

    /// <summary>
    /// Tries to resolve a model identifier.
    /// </summary>
    /// <param name="modelIdOrAlias">Model ID, alias, or local path.</param>
    /// <param name="modelInfo">The resolved model information.</param>
    /// <returns>True if resolved successfully.</returns>
    public bool TryResolve(string modelIdOrAlias, out ModelInfo? modelInfo)
    {
        try
        {
            modelInfo = Resolve(modelIdOrAlias);
            return true;
        }
        catch
        {
            modelInfo = null;
            return false;
        }
    }

    /// <summary>
    /// Gets all registered models.
    /// </summary>
    public IEnumerable<ModelInfo> GetAll() => _modelsById.Values;

    /// <summary>
    /// Gets all available aliases including "auto".
    /// </summary>
    public IEnumerable<string> GetAliases()
    {
        yield return "auto";
        foreach (var alias in _modelsByAlias.Keys)
        {
            yield return alias;
        }
    }

    /// <summary>
    /// Gets the optimal model based on current hardware profile.
    /// Uses PerformanceTier to select appropriate model size.
    /// </summary>
    /// <remarks>
    /// Tier mapping:
    /// - Low:    ms-marco-MiniLM-L-6-v2 (22M params) - fast, lightweight
    /// - Medium: bge-reranker-base (278M params) - balanced, multilingual
    /// - High:   bge-reranker-large (560M params) - highest accuracy
    /// - Ultra:  bge-reranker-large (560M params) - highest accuracy
    /// </remarks>
    public static ModelInfo GetAutoModel()
    {
        var tier = HardwareProfile.Current.Tier;

        return tier switch
        {
            PerformanceTier.Ultra or PerformanceTier.High
                => DefaultModels.BgeRerankerLarge,
            PerformanceTier.Medium
                => DefaultModels.BgeRerankerBase,
            _
                => DefaultModels.MsMarcoMiniLML6V2
        };
    }

    private static bool IsLocalPath(string path)
    {
        // Check for absolute or relative file paths
        return path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) ||
               Path.IsPathRooted(path) ||
               path.StartsWith("./", StringComparison.Ordinal) ||
               path.StartsWith("../", StringComparison.Ordinal) ||
               path.StartsWith(".\\", StringComparison.Ordinal) ||
               path.StartsWith("..\\", StringComparison.Ordinal);
    }

    private static ModelInfo CreateLocalModelInfo(string path)
    {
        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath) ?? ".";
        var fileName = Path.GetFileName(fullPath);

        return new ModelInfo
        {
            Id = fullPath,
            Alias = "local",
            DisplayName = $"Local: {fileName}",
            Parameters = 0,
            MaxSequenceLength = 512, // Default assumption
            SizeBytes = File.Exists(fullPath) ? new FileInfo(fullPath).Length : 0,
            OnnxFile = fileName,
            TokenizerFile = "tokenizer.json",
            Description = $"Local model from {directory}",
            IsMultilingual = false
        };
    }

    private static ModelInfo CreateHuggingFaceModelInfo(string modelId)
    {
        // Create a basic ModelInfo for unknown HuggingFace models
        var parts = modelId.Split('/');
        var name = parts.Length > 1 ? parts[1] : modelId;

        return new ModelInfo
        {
            Id = modelId,
            Alias = modelId,
            DisplayName = name,
            Parameters = 0, // Unknown
            MaxSequenceLength = 512, // Default assumption
            SizeBytes = 0, // Unknown
            OnnxFile = "onnx/model.onnx",
            TokenizerFile = "tokenizer.json",
            Description = $"HuggingFace model: {modelId}",
            IsMultilingual = false
        };
    }
}

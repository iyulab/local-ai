using LMSupply.Hardware;

namespace LMSupply.Embedder.Utils;

/// <summary>
/// Registry of pre-configured embedding models.
/// Updated: 2025-12 based on MTEB leaderboard rankings.
/// </summary>
internal static class ModelRegistry
{
    private static readonly Dictionary<string, ModelInfo> _models = new(StringComparer.OrdinalIgnoreCase)
    {
        // ===== Aliases (user-friendly names) =====
        ["default"] = new ModelInfo
        {
            RepoId = "BAAI/bge-small-en-v1.5",
            Dimensions = 384,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Cls,
            DoLowerCase = true,
            Description = "Default: BGE Small English v1.5, 33M params, MTEB top performer",
            Subfolder = "onnx"
        },
        ["fast"] = new ModelInfo
        {
            RepoId = "sentence-transformers/all-MiniLM-L6-v2",
            Dimensions = 384,
            MaxSequenceLength = 256,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = true,
            Description = "Fast: all-MiniLM-L6-v2, 22M params, ultra-lightweight",
            Subfolder = "onnx"
        },
        ["quality"] = new ModelInfo
        {
            RepoId = "Alibaba-NLP/gte-base-en-v1.5",
            Dimensions = 768,
            MaxSequenceLength = 8192,
            PoolingMode = PoolingMode.Cls,
            DoLowerCase = false,
            Description = "Quality: GTE Base English v1.5, 109M params, 8K context, Apache 2.0",
            Subfolder = "onnx"
        },
        ["large"] = new ModelInfo
        {
            RepoId = "nomic-ai/nomic-embed-text-v1.5",
            Dimensions = 768,
            MaxSequenceLength = 8192,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            Description = "Large: Nomic Embed v1.5, 137M params, 8K context, 2024 MTEB top",
            Subfolder = "onnx"
        },
        ["multilingual"] = new ModelInfo
        {
            RepoId = "intfloat/multilingual-e5-base",
            Dimensions = 768,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            Description = "Multilingual: E5 Base, 278M params, 100+ languages",
            Subfolder = "onnx"
        },

        // ===== Explicit model names =====
        ["all-MiniLM-L6-v2"] = new ModelInfo
        {
            RepoId = "sentence-transformers/all-MiniLM-L6-v2",
            Dimensions = 384,
            MaxSequenceLength = 256,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = true,
            Description = "22M params, ultra-fast, English",
            Subfolder = "onnx"
        },
        ["all-mpnet-base-v2"] = new ModelInfo
        {
            RepoId = "sentence-transformers/all-mpnet-base-v2",
            Dimensions = 768,
            MaxSequenceLength = 384,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = true,
            Description = "110M params, legacy quality model, English",
            Subfolder = "onnx"
        },
        ["bge-small-en-v1.5"] = new ModelInfo
        {
            RepoId = "BAAI/bge-small-en-v1.5",
            Dimensions = 384,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Cls,
            DoLowerCase = true,
            Description = "33M params, MTEB top performer for size, English",
            Subfolder = "onnx"
        },
        ["bge-base-en-v1.5"] = new ModelInfo
        {
            RepoId = "BAAI/bge-base-en-v1.5",
            Dimensions = 768,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Cls,
            DoLowerCase = true,
            Description = "110M params, excellent quality, English",
            Subfolder = "onnx"
        },
        ["bge-large-en-v1.5"] = new ModelInfo
        {
            RepoId = "BAAI/bge-large-en-v1.5",
            Dimensions = 1024,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Cls,
            DoLowerCase = true,
            Description = "335M params, highest accuracy BGE, English",
            Subfolder = "onnx"
        },
        ["nomic-embed-text-v1.5"] = new ModelInfo
        {
            RepoId = "nomic-ai/nomic-embed-text-v1.5",
            Dimensions = 768,
            MaxSequenceLength = 8192,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            Description = "137M params, 8K context, 2024 MTEB top performer",
            Subfolder = "onnx"
        },
        ["e5-small-v2"] = new ModelInfo
        {
            RepoId = "intfloat/e5-small-v2",
            Dimensions = 384,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            Description = "33M params, no prefix needed, English",
            Subfolder = "onnx"
        },
        ["e5-base-v2"] = new ModelInfo
        {
            RepoId = "intfloat/e5-base-v2",
            Dimensions = 768,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            Description = "110M params, excellent retrieval, English",
            Subfolder = "onnx"
        },
        ["multilingual-e5-small"] = new ModelInfo
        {
            RepoId = "intfloat/multilingual-e5-small",
            Dimensions = 384,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            Description = "118M params, 100+ languages, compact",
            Subfolder = "onnx"
        },
        ["multilingual-e5-base"] = new ModelInfo
        {
            RepoId = "intfloat/multilingual-e5-base",
            Dimensions = 768,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            Description = "278M params, 100+ languages, quality",
            Subfolder = "onnx"
        },
        ["multilingual-e5-large"] = new ModelInfo
        {
            RepoId = "intfloat/multilingual-e5-large",
            Dimensions = 1024,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            Description = "560M params, 100+ languages, highest quality",
            Subfolder = "onnx"
        },
        // GTE models (2024-2025 MTEB top performers)
        ["gte-base-en-v1.5"] = new ModelInfo
        {
            RepoId = "Alibaba-NLP/gte-base-en-v1.5",
            Dimensions = 768,
            MaxSequenceLength = 8192,
            PoolingMode = PoolingMode.Cls,
            DoLowerCase = false,
            Description = "109M params, 8K context, 2025 MTEB top performer",
            Subfolder = "onnx"
        },
        ["gte-large-en-v1.5"] = new ModelInfo
        {
            RepoId = "Alibaba-NLP/gte-large-en-v1.5",
            Dimensions = 1024,
            MaxSequenceLength = 8192,
            PoolingMode = PoolingMode.Cls,
            DoLowerCase = false,
            Description = "434M params, 8K context, highest accuracy GTE",
            Subfolder = "onnx"
        }
    };

    /// <summary>
    /// Tries to get model info by model ID.
    /// Supports "auto" alias which selects optimal model based on hardware.
    /// </summary>
    public static bool TryGetModel(string modelId, out ModelInfo? info)
    {
        if (modelId.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            info = GetAutoModel() with { Alias = "auto" };
            return true;
        }

        if (_models.TryGetValue(modelId, out info))
        {
            info = info with { Alias = modelId };
            return true;
        }

        return false;
    }

    /// <summary>
    /// Gets the optimal model based on current hardware profile.
    /// Uses PerformanceTier to select appropriate model size.
    /// </summary>
    /// <remarks>
    /// Tier mapping:
    /// - Low:    bge-small-en-v1.5 (33M params, 384 dims) - lightweight
    /// - Medium: bge-base-en-v1.5 (110M params, 768 dims) - balanced
    /// - High:   gte-large-en-v1.5 (434M params, 1024 dims, 8K context) - quality
    /// - Ultra:  gte-large-en-v1.5 (434M params, 1024 dims, 8K context) - quality
    /// </remarks>
    public static ModelInfo GetAutoModel()
    {
        var tier = HardwareProfile.Current.Tier;

        return tier switch
        {
            PerformanceTier.Ultra or PerformanceTier.High
                => _models["gte-large-en-v1.5"],
            PerformanceTier.Medium
                => _models["bge-base-en-v1.5"],
            _
                => _models["bge-small-en-v1.5"]
        };
    }

    /// <summary>
    /// Gets all available model IDs including "auto".
    /// </summary>
    public static IEnumerable<string> GetAvailableModels()
    {
        yield return "auto";
        foreach (var key in _models.Keys)
        {
            yield return key;
        }
    }
}

/// <summary>
/// Configuration information for a pre-configured model.
/// </summary>
public sealed record ModelInfo : IModelInfoBase
{
    /// <summary>
    /// Gets the unique identifier for this model (HuggingFace repository ID).
    /// </summary>
    public required string RepoId { get; init; }

    /// <summary>
    /// Gets the user-friendly alias for this model (e.g., "default", "fast").
    /// Set internally by the registry when resolving.
    /// </summary>
    public string Alias { get; internal init; } = string.Empty;

    /// <summary>
    /// Gets the embedding dimensions.
    /// </summary>
    public required int Dimensions { get; init; }

    /// <summary>
    /// Gets the maximum input sequence length.
    /// </summary>
    public required int MaxSequenceLength { get; init; }

    /// <summary>
    /// Gets the pooling mode for generating embeddings.
    /// </summary>
    public required PoolingMode PoolingMode { get; init; }

    /// <summary>
    /// Gets whether to lowercase input text.
    /// </summary>
    public required bool DoLowerCase { get; init; }

    /// <summary>
    /// Gets the model description.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>
    /// Gets the subfolder within the HuggingFace repository.
    /// </summary>
    public string? Subfolder { get; init; }

    // IModelInfoBase implementation
    string IModelInfoBase.Id => RepoId;
}

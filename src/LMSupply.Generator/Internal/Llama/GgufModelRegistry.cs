namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// Registry of well-known GGUF models with aliases for easy access.
/// Follows the same alias pattern as the ONNX ModelRegistry.
/// </summary>
public static class GgufModelRegistry
{
    private static readonly Dictionary<string, GgufModelInfo> _models = new(StringComparer.OrdinalIgnoreCase)
    {
        // ============================================================
        // TIER 1: MIT License - No restrictions
        // ============================================================

        // Note: Most GGUF models on HuggingFace are conversions of models
        // with conditional licenses. Pure MIT GGUF models are rare.

        // ============================================================
        // TIER 2: Conditional License - Usage restrictions apply
        // ============================================================

        // Default: Balanced quality/speed (Llama 3.2 3B)
        ["gguf:default"] = new GgufModelInfo
        {
            RepoId = "bartowski/Llama-3.2-3B-Instruct-GGUF",
            DisplayName = "Llama 3.2 3B Instruct",
            DefaultFile = "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            ChatFormat = "llama3",
            ContextLength = 8192,
            ParameterCount = 3_000_000_000,
            License = LicenseTier.Conditional,
            LicenseName = "Llama 3.2 Community License",
            LicenseRestrictions = "700M MAU limit for commercial use"
        },

        // Fast: Small model for quick inference
        ["gguf:fast"] = new GgufModelInfo
        {
            RepoId = "bartowski/Llama-3.2-1B-Instruct-GGUF",
            DisplayName = "Llama 3.2 1B Instruct",
            DefaultFile = "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            ChatFormat = "llama3",
            ContextLength = 8192,
            ParameterCount = 1_000_000_000,
            License = LicenseTier.Conditional,
            LicenseName = "Llama 3.2 Community License",
            LicenseRestrictions = "700M MAU limit for commercial use"
        },

        // Quality: Higher quality model
        ["gguf:quality"] = new GgufModelInfo
        {
            RepoId = "bartowski/Qwen2.5-7B-Instruct-GGUF",
            DisplayName = "Qwen 2.5 7B Instruct",
            DefaultFile = "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            ChatFormat = "chatml",
            ContextLength = 32768,
            ParameterCount = 7_000_000_000,
            License = LicenseTier.Conditional,
            LicenseName = "Qwen License",
            LicenseRestrictions = "Commercial use requires agreement"
        },

        // Large: Larger model for best quality
        ["gguf:large"] = new GgufModelInfo
        {
            RepoId = "bartowski/Qwen2.5-14B-Instruct-GGUF",
            DisplayName = "Qwen 2.5 14B Instruct",
            DefaultFile = "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            ChatFormat = "chatml",
            ContextLength = 32768,
            ParameterCount = 14_000_000_000,
            License = LicenseTier.Conditional,
            LicenseName = "Qwen License",
            LicenseRestrictions = "Commercial use requires agreement"
        },

        // Multilingual: Good for non-English tasks
        ["gguf:multilingual"] = new GgufModelInfo
        {
            RepoId = "bartowski/gemma-2-9b-it-GGUF",
            DisplayName = "Gemma 2 9B Instruct",
            DefaultFile = "gemma-2-9b-it-Q4_K_M.gguf",
            ChatFormat = "gemma",
            ContextLength = 8192,
            ParameterCount = 9_000_000_000,
            License = LicenseTier.Conditional,
            LicenseName = "Gemma Terms of Use",
            LicenseRestrictions = "Prohibited Use Policy applies"
        },

        // Korean: Korean language specialized
        ["gguf:korean"] = new GgufModelInfo
        {
            RepoId = "bartowski/EXAONE-3.5-7.8B-Instruct-GGUF",
            DisplayName = "EXAONE 3.5 7.8B Instruct",
            DefaultFile = "EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf",
            ChatFormat = "exaone",
            ContextLength = 32768,
            ParameterCount = 7_800_000_000,
            License = LicenseTier.Conditional,
            LicenseName = "EXAONE AI Model License",
            LicenseRestrictions = "LG AI Research terms apply"
        },

        // Code: Optimized for coding tasks
        ["gguf:code"] = new GgufModelInfo
        {
            RepoId = "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
            DisplayName = "Qwen 2.5 Coder 7B",
            DefaultFile = "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
            ChatFormat = "chatml",
            ContextLength = 32768,
            ParameterCount = 7_000_000_000,
            License = LicenseTier.Conditional,
            LicenseName = "Qwen License",
            LicenseRestrictions = "Commercial use requires agreement"
        },

        // Reasoning: DeepSeek R1 for complex reasoning
        ["gguf:reasoning"] = new GgufModelInfo
        {
            RepoId = "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
            DisplayName = "DeepSeek R1 Distill 8B",
            DefaultFile = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
            ChatFormat = "deepseek",
            ContextLength = 32768,
            ParameterCount = 8_000_000_000,
            License = LicenseTier.Conditional,
            LicenseName = "DeepSeek License",
            LicenseRestrictions = "Research and commercial use allowed"
        }
    };

    /// <summary>
    /// Resolves an alias to model information.
    /// Supports both "gguf:alias" format and plain "alias" format.
    /// </summary>
    /// <param name="aliasOrRepoId">The alias (e.g., "gguf:default", "default") or full repo ID.</param>
    /// <returns>Model information if found, null otherwise.</returns>
    public static GgufModelInfo? Resolve(string aliasOrRepoId)
    {
        if (string.IsNullOrWhiteSpace(aliasOrRepoId))
            return null;

        // Try direct lookup with gguf: prefix
        if (_models.TryGetValue(aliasOrRepoId, out var info))
            return info;

        // Try with gguf: prefix added
        if (!aliasOrRepoId.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase))
        {
            if (_models.TryGetValue($"gguf:{aliasOrRepoId}", out info))
                return info;
        }

        return null;
    }

    /// <summary>
    /// Gets all registered GGUF models.
    /// </summary>
    public static IReadOnlyList<GgufModelInfo> GetAllModels() =>
        _models.Values.ToList();

    /// <summary>
    /// Gets GGUF models filtered by license tier.
    /// </summary>
    public static IReadOnlyList<GgufModelInfo> GetModelsByLicense(LicenseTier tier) =>
        _models.Values.Where(m => m.License == tier).ToList();

    /// <summary>
    /// Checks if a string is a known GGUF alias.
    /// Only matches "gguf:"-prefixed aliases (e.g., "gguf:default", "gguf:fast").
    /// Plain aliases like "default" or "fast" are reserved for ONNX models.
    /// </summary>
    public static bool IsAlias(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return false;

        // Only match if it starts with "gguf:" prefix
        // Plain aliases without prefix are reserved for ONNX ModelRegistry
        return value.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase) &&
               _models.ContainsKey(value);
    }

    /// <summary>
    /// Gets all available alias names.
    /// </summary>
    public static IReadOnlyList<string> GetAliases() =>
        _models.Keys.ToList();
}

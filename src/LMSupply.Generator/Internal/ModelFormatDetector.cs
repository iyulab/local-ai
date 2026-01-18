using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Internal;

/// <summary>
/// Detects the model format (ONNX or GGUF) from a model identifier or path.
/// </summary>
internal static class ModelFormatDetector
{
    private static readonly string[] GgufExtensions = [".gguf"];
    private static readonly string[] OnnxExtensions = [".onnx"];
    private static readonly string[] OnnxRequiredFiles = ["genai_config.json", "model.onnx"];

    /// <summary>
    /// Detects the model format from a model identifier or local path.
    /// </summary>
    /// <param name="modelIdOrPath">HuggingFace model ID or local file/directory path.</param>
    /// <returns>The detected model format.</returns>
    public static ModelFormat Detect(string modelIdOrPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);

        // 0. Check if it's a GGUF registry alias (e.g., "gguf:default", "gguf:fast")
        if (GgufModelRegistry.IsAlias(modelIdOrPath))
        {
            return ModelFormat.Gguf;
        }

        // 1. Check file extension for direct file paths
        if (HasExtension(modelIdOrPath, GgufExtensions))
            return ModelFormat.Gguf;

        if (HasExtension(modelIdOrPath, OnnxExtensions))
            return ModelFormat.Onnx;

        // 2. Check local directory for model files
        if (Directory.Exists(modelIdOrPath))
        {
            return DetectFromDirectory(modelIdOrPath);
        }

        // 3. Check if it's a HuggingFace repo ID pattern
        if (IsHuggingFaceRepoId(modelIdOrPath))
        {
            return DetectFromRepoId(modelIdOrPath);
        }

        // 4. Check if it's a local file path (not yet existing)
        if (IsFilePath(modelIdOrPath))
        {
            // Infer from the path pattern even if file doesn't exist yet
            var fileName = Path.GetFileName(modelIdOrPath);
            if (HasExtension(fileName, GgufExtensions))
                return ModelFormat.Gguf;
        }

        // 5. Check registry for known ONNX models
        if (ModelRegistry.GetModel(modelIdOrPath) != null)
        {
            return ModelFormat.Onnx;
        }

        // 6. Default: Assume ONNX for backward compatibility
        // Unknown HuggingFace repos are assumed to be ONNX format
        return ModelFormat.Onnx;
    }

    /// <summary>
    /// Detects format from a local directory by examining its contents.
    /// </summary>
    private static ModelFormat DetectFromDirectory(string path)
    {
        // Check for ONNX GenAI structure (genai_config.json + model.onnx)
        if (HasOnnxGenAiStructure(path))
            return ModelFormat.Onnx;

        // Check for GGUF files in directory
        var ggufFiles = Directory.EnumerateFiles(path, "*.gguf", SearchOption.TopDirectoryOnly);
        if (ggufFiles.Any())
            return ModelFormat.Gguf;

        // Check subdirectories for ONNX structure (common pattern: cpu-int4-*/model.onnx)
        foreach (var subdir in Directory.EnumerateDirectories(path))
        {
            if (HasOnnxGenAiStructure(subdir))
                return ModelFormat.Onnx;
        }

        return ModelFormat.Unknown;
    }

    /// <summary>
    /// Detects format from HuggingFace repository ID.
    /// </summary>
    private static ModelFormat DetectFromRepoId(string repoId)
    {
        var lowerRepoId = repoId.ToLowerInvariant();

        // Check for GGUF indicators in repo name
        if (lowerRepoId.Contains("-gguf") ||
            lowerRepoId.Contains("_gguf") ||
            lowerRepoId.EndsWith("/gguf"))
        {
            return ModelFormat.Gguf;
        }

        // Check for ONNX indicators in repo name
        if (lowerRepoId.Contains("-onnx") ||
            lowerRepoId.Contains("_onnx") ||
            lowerRepoId.EndsWith("/onnx"))
        {
            return ModelFormat.Onnx;
        }

        // Check known GGUF providers
        if (IsKnownGgufProvider(repoId))
        {
            return ModelFormat.Gguf;
        }

        // Default to ONNX for backward compatibility
        return ModelFormat.Onnx;
    }

    /// <summary>
    /// Checks if the repository ID belongs to a known GGUF model provider.
    /// </summary>
    private static bool IsKnownGgufProvider(string repoId)
    {
        var parts = repoId.Split('/');
        if (parts.Length < 2) return false;

        var owner = parts[0].ToLowerInvariant();

        // Known GGUF model providers on HuggingFace
        return owner switch
        {
            "thebloke" => true,
            "bartowski" => true,
            "ggml-org" => true,
            "mradermacher" => true,
            "unsloth" when repoId.Contains("gguf", StringComparison.OrdinalIgnoreCase) => true,
            _ => false
        };
    }

    /// <summary>
    /// Checks if the path has ONNX GenAI model structure.
    /// </summary>
    private static bool HasOnnxGenAiStructure(string path)
    {
        // Must have genai_config.json
        if (!File.Exists(Path.Combine(path, "genai_config.json")))
            return false;

        // Must have at least one .onnx file
        return Directory.EnumerateFiles(path, "*.onnx", SearchOption.TopDirectoryOnly).Any();
    }

    /// <summary>
    /// Checks if the string is a valid HuggingFace repository ID pattern.
    /// </summary>
    private static bool IsHuggingFaceRepoId(string value)
    {
        // HuggingFace repo IDs: "owner/repo-name" or "owner/repo-name/subfolder"
        // Must not contain path separators other than '/'
        if (value.Contains('\\') || value.Contains(':'))
            return false;

        var parts = value.Split('/');
        return parts.Length >= 2 && parts.Length <= 4 &&
               !string.IsNullOrWhiteSpace(parts[0]) &&
               !string.IsNullOrWhiteSpace(parts[1]);
    }

    /// <summary>
    /// Checks if the string looks like a file path.
    /// </summary>
    private static bool IsFilePath(string value)
    {
        return value.Contains(Path.DirectorySeparatorChar) ||
               value.Contains(Path.AltDirectorySeparatorChar) ||
               Path.IsPathRooted(value);
    }

    /// <summary>
    /// Checks if the string ends with any of the specified extensions.
    /// </summary>
    private static bool HasExtension(string value, string[] extensions)
    {
        foreach (var ext in extensions)
        {
            if (value.EndsWith(ext, StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }
}

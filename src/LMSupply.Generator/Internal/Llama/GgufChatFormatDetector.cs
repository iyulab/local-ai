namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// Detects chat format from GGUF model filenames.
/// </summary>
internal static class GgufChatFormatDetector
{
    /// <summary>
    /// Detects the chat format from a GGUF filename or path.
    /// </summary>
    /// <param name="fileNameOrPath">The filename or full path to the GGUF file.</param>
    /// <returns>The detected chat format identifier.</returns>
    public static string DetectFromFilename(string fileNameOrPath)
    {
        var filename = Path.GetFileName(fileNameOrPath).ToLowerInvariant();

        // DeepSeek (check before llama due to "DeepSeek-R1-Distill-Llama")
        if (filename.Contains("deepseek"))
            return "deepseek";

        // Llama family
        if (filename.Contains("llama"))
            return "llama3";

        // Qwen family (uses ChatML)
        if (filename.Contains("qwen"))
            return "chatml";

        // Mistral family
        if (filename.Contains("mistral") || filename.Contains("mixtral"))
            return "mistral";

        // Gemma family
        if (filename.Contains("gemma"))
            return "gemma";

        // Phi family
        if (filename.Contains("phi"))
            return "phi3";

        // EXAONE (Korean)
        if (filename.Contains("exaone"))
            return "exaone";

        // Yi family
        if (filename.Contains("yi-"))
            return "chatml";

        // InternLM
        if (filename.Contains("internlm"))
            return "chatml";

        // CodeLlama
        if (filename.Contains("codellama"))
            return "llama3";

        // Vicuna
        if (filename.Contains("vicuna"))
            return "vicuna";

        // OpenChat
        if (filename.Contains("openchat"))
            return "chatml";

        // Zephyr
        if (filename.Contains("zephyr"))
            return "zephyr";

        // Default to ChatML as it's widely compatible
        return "chatml";
    }

    /// <summary>
    /// Checks if the detected format is supported.
    /// </summary>
    /// <param name="format">The chat format identifier.</param>
    /// <returns>True if the format is supported.</returns>
    public static bool IsSupported(string format)
    {
        return format.ToLowerInvariant() switch
        {
            "llama3" or "llama2" => true,
            "chatml" => true,
            "mistral" => true,
            "gemma" => true,
            "phi3" => true,
            "exaone" => true,
            "deepseek" => true,
            "vicuna" => true,
            "zephyr" => true,
            _ => false
        };
    }
}

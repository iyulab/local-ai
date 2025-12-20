using System.Text.Json;
using System.Text.Json.Serialization;

namespace LMSupply.Console.Host.Models.OpenAI;

/// <summary>
/// Translation request
/// </summary>
public sealed record TranslateRequest
{
    /// <summary>
    /// Model ID (e.g., "en-ko", "ko-en", or HuggingFace repo ID)
    /// </summary>
    public string Model { get; init; } = "default";

    /// <summary>
    /// Text(s) to translate. Can be string or array of strings.
    /// </summary>
    public required JsonElement Input { get; init; }

    /// <summary>
    /// Source language code (auto-detected if not specified)
    /// </summary>
    [JsonPropertyName("source_language")]
    public string? SourceLanguage { get; init; }

    /// <summary>
    /// Target language code
    /// </summary>
    [JsonPropertyName("target_language")]
    public string? TargetLanguage { get; init; }
}

/// <summary>
/// Translation response
/// </summary>
public sealed record TranslateResponse
{
    public required string Id { get; init; }
    public required string Model { get; init; }
    public required IReadOnlyList<TranslationResult> Translations { get; init; }
}

/// <summary>
/// Individual translation result
/// </summary>
public sealed record TranslationResult
{
    public int Index { get; init; }
    [JsonPropertyName("source_text")]
    public required string SourceText { get; init; }
    [JsonPropertyName("translated_text")]
    public required string TranslatedText { get; init; }
    [JsonPropertyName("source_language")]
    public string? SourceLanguage { get; init; }
    [JsonPropertyName("target_language")]
    public string? TargetLanguage { get; init; }
}

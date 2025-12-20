using System.Text.Json;
using System.Text.Json.Serialization;

namespace LMSupply.Console.Host.Models.OpenAI;

/// <summary>
/// OpenAI /v1/embeddings request
/// </summary>
public sealed record EmbeddingRequest
{
    /// <summary>
    /// Model ID
    /// </summary>
    public string Model { get; init; } = "default";

    /// <summary>
    /// Input text(s) to embed. Can be a string or array of strings.
    /// </summary>
    public required JsonElement Input { get; init; }

    /// <summary>
    /// Encoding format: "float" or "base64"
    /// </summary>
    [JsonPropertyName("encoding_format")]
    public string? EncodingFormat { get; init; }

    /// <summary>
    /// Number of dimensions (for models that support it)
    /// </summary>
    public int? Dimensions { get; init; }

    /// <summary>
    /// User identifier
    /// </summary>
    public string? User { get; init; }
}

/// <summary>
/// OpenAI /v1/embeddings response
/// </summary>
public sealed record EmbeddingResponse
{
    public string Object { get; init; } = "list";
    public required IReadOnlyList<EmbeddingData> Data { get; init; }
    public required string Model { get; init; }
    public required EmbeddingUsage Usage { get; init; }
}

/// <summary>
/// Individual embedding data
/// </summary>
public sealed record EmbeddingData
{
    public string Object { get; init; } = "embedding";
    public int Index { get; init; }
    public required float[] Embedding { get; init; }
}

/// <summary>
/// Embedding usage information
/// </summary>
public sealed record EmbeddingUsage
{
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; init; }
    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; init; }
}

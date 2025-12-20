using System.Text.Json.Serialization;

namespace LMSupply.Console.Host.Models.OpenAI;

/// <summary>
/// Rerank request (Cohere API compatible)
/// </summary>
public sealed record RerankRequest
{
    /// <summary>
    /// Model ID
    /// </summary>
    public string Model { get; init; } = "default";

    /// <summary>
    /// Query to match documents against
    /// </summary>
    public required string Query { get; init; }

    /// <summary>
    /// Documents to rerank
    /// </summary>
    public required IReadOnlyList<string> Documents { get; init; }

    /// <summary>
    /// Number of top results to return
    /// </summary>
    [JsonPropertyName("top_n")]
    public int? TopN { get; init; }

    /// <summary>
    /// Whether to return documents in the response
    /// </summary>
    [JsonPropertyName("return_documents")]
    public bool ReturnDocuments { get; init; } = true;
}

/// <summary>
/// Rerank response
/// </summary>
public sealed record RerankResponse
{
    public required string Id { get; init; }
    public required string Model { get; init; }
    public required IReadOnlyList<RerankResult> Results { get; init; }
}

/// <summary>
/// Individual rerank result
/// </summary>
public sealed record RerankResult
{
    public int Index { get; init; }
    [JsonPropertyName("relevance_score")]
    public float RelevanceScore { get; init; }
    public RerankDocument? Document { get; init; }
}

/// <summary>
/// Reranked document
/// </summary>
public sealed record RerankDocument
{
    public required string Text { get; init; }
}

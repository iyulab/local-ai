using System.Text.Json.Serialization;

namespace LMSupply.Console.Host.Models.OpenAI;

/// <summary>
/// OpenAI /v1/chat/completions request
/// </summary>
public sealed record ChatCompletionRequest
{
    /// <summary>
    /// Model ID (e.g., "microsoft/Phi-4-mini-instruct-onnx" or "default")
    /// </summary>
    public string Model { get; init; } = "default";

    /// <summary>
    /// Messages in the conversation
    /// </summary>
    public required IReadOnlyList<ChatCompletionMessage> Messages { get; init; }

    /// <summary>
    /// Sampling temperature (0-2)
    /// </summary>
    public float? Temperature { get; init; }

    /// <summary>
    /// Nucleus sampling (0-1)
    /// </summary>
    [JsonPropertyName("top_p")]
    public float? TopP { get; init; }

    /// <summary>
    /// Maximum tokens to generate
    /// </summary>
    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; init; }

    /// <summary>
    /// Whether to stream the response
    /// </summary>
    public bool Stream { get; init; } = false;

    /// <summary>
    /// Stop sequences
    /// </summary>
    public IReadOnlyList<string>? Stop { get; init; }

    /// <summary>
    /// Frequency penalty (-2 to 2)
    /// </summary>
    [JsonPropertyName("frequency_penalty")]
    public float? FrequencyPenalty { get; init; }

    /// <summary>
    /// Presence penalty (-2 to 2)
    /// </summary>
    [JsonPropertyName("presence_penalty")]
    public float? PresencePenalty { get; init; }

    /// <summary>
    /// User identifier for abuse detection
    /// </summary>
    public string? User { get; init; }
}

/// <summary>
/// Chat message
/// </summary>
public sealed record ChatCompletionMessage
{
    /// <summary>
    /// Role: system, user, assistant
    /// </summary>
    public required string Role { get; init; }

    /// <summary>
    /// Message content
    /// </summary>
    public required string Content { get; init; }

    /// <summary>
    /// Optional name for the participant
    /// </summary>
    public string? Name { get; init; }
}

/// <summary>
/// OpenAI /v1/chat/completions response
/// </summary>
public sealed record ChatCompletionResponse
{
    public required string Id { get; init; }
    public string Object { get; init; } = "chat.completion";
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    public required string Model { get; init; }
    public required IReadOnlyList<ChatCompletionChoice> Choices { get; init; }
    public Usage? Usage { get; init; }
}

/// <summary>
/// Chat completion choice
/// </summary>
public sealed record ChatCompletionChoice
{
    public int Index { get; init; }
    public required ChatCompletionMessage Message { get; init; }
    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; init; }
}

/// <summary>
/// Streaming chunk response
/// </summary>
public sealed record ChatCompletionChunk
{
    public required string Id { get; init; }
    public string Object { get; init; } = "chat.completion.chunk";
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    public required string Model { get; init; }
    public required IReadOnlyList<ChatCompletionChunkChoice> Choices { get; init; }
}

/// <summary>
/// Streaming chunk choice
/// </summary>
public sealed record ChatCompletionChunkChoice
{
    public int Index { get; init; }
    public required ChatCompletionDelta Delta { get; init; }
    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; init; }
}

/// <summary>
/// Streaming delta content
/// </summary>
public sealed record ChatCompletionDelta
{
    public string? Role { get; init; }
    public string? Content { get; init; }
}

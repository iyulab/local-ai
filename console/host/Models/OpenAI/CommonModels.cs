namespace LMSupply.Console.Host.Models.OpenAI;

/// <summary>
/// OpenAI API compatible error response
/// </summary>
public sealed record ErrorResponse
{
    public required ErrorDetail Error { get; init; }
}

public sealed record ErrorDetail
{
    public required string Message { get; init; }
    public required string Type { get; init; }
    public string? Param { get; init; }
    public string? Code { get; init; }
}

/// <summary>
/// OpenAI API compatible usage information
/// </summary>
public sealed record Usage
{
    public int PromptTokens { get; init; }
    public int CompletionTokens { get; init; }
    public int TotalTokens { get; init; }
}

/// <summary>
/// Model information (OpenAI /v1/models compatible)
/// </summary>
public sealed record ModelInfo
{
    public required string Id { get; init; }
    public string Object { get; init; } = "model";
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    public string OwnedBy { get; init; } = "lmsupply";
}

/// <summary>
/// Model list response (OpenAI /v1/models compatible)
/// </summary>
public sealed record ModelListResponse
{
    public string Object { get; init; } = "list";
    public required IReadOnlyList<ModelInfo> Data { get; init; }
}

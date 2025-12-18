namespace LMSupply.Console.Host.Models.Requests;

/// <summary>
/// 채팅 요청
/// </summary>
public sealed record ChatRequest
{
    /// <summary>
    /// 모델 ID (예: "microsoft/Phi-4-mini-instruct-onnx" 또는 "default")
    /// </summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// 메시지 목록
    /// </summary>
    public required IReadOnlyList<ChatMessageDto> Messages { get; init; }

    /// <summary>
    /// 생성 옵션
    /// </summary>
    public ChatOptionsDto? Options { get; init; }
}

/// <summary>
/// 채팅 메시지
/// </summary>
public sealed record ChatMessageDto
{
    /// <summary>
    /// 역할 (system, user, assistant)
    /// </summary>
    public required string Role { get; init; }

    /// <summary>
    /// 내용
    /// </summary>
    public required string Content { get; init; }
}

/// <summary>
/// 생성 옵션
/// </summary>
public sealed record ChatOptionsDto
{
    public int MaxTokens { get; init; } = 2048;
    public float Temperature { get; init; } = 0.7f;
    public float TopP { get; init; } = 0.9f;
    public int TopK { get; init; } = 50;
    public float RepetitionPenalty { get; init; } = 1.0f;
    public IReadOnlyList<string>? StopSequences { get; init; }
}

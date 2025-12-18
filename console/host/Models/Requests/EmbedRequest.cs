namespace LMSupply.Console.Host.Models.Requests;

/// <summary>
/// 임베딩 요청
/// </summary>
public sealed record EmbedRequest
{
    /// <summary>
    /// 모델 ID
    /// </summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// 임베딩할 텍스트 (단일)
    /// </summary>
    public string? Text { get; init; }

    /// <summary>
    /// 임베딩할 텍스트 목록 (배치)
    /// </summary>
    public IReadOnlyList<string>? Texts { get; init; }
}

/// <summary>
/// 유사도 계산 요청
/// </summary>
public sealed record SimilarityRequest
{
    /// <summary>
    /// 모델 ID
    /// </summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// 첫 번째 텍스트
    /// </summary>
    public required string Text1 { get; init; }

    /// <summary>
    /// 두 번째 텍스트
    /// </summary>
    public required string Text2 { get; init; }
}

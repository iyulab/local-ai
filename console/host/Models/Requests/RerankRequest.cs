namespace LMSupply.Console.Host.Models.Requests;

/// <summary>
/// 리랭킹 요청
/// </summary>
public sealed record RerankRequest
{
    /// <summary>
    /// 모델 ID
    /// </summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// 검색 쿼리
    /// </summary>
    public required string Query { get; init; }

    /// <summary>
    /// 리랭킹할 문서 목록
    /// </summary>
    public required IReadOnlyList<string> Documents { get; init; }

    /// <summary>
    /// 상위 K개만 반환 (null이면 전체)
    /// </summary>
    public int? TopK { get; init; }
}

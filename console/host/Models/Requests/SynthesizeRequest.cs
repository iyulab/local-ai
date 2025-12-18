namespace LMSupply.Console.Host.Models.Requests;

/// <summary>
/// TTS 요청
/// </summary>
public sealed record SynthesizeRequest
{
    /// <summary>
    /// 모델 ID
    /// </summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// 합성할 텍스트
    /// </summary>
    public required string Text { get; init; }

    /// <summary>
    /// 음성 속도 (0.5 ~ 2.0)
    /// </summary>
    public float Speed { get; init; } = 1.0f;
}

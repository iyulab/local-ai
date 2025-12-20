using System.Text.Json.Serialization;

namespace LMSupply.Console.Host.Models.OpenAI;

/// <summary>
/// Image caption response
/// </summary>
public sealed record CaptionResponse
{
    public required string Id { get; init; }
    public required string Model { get; init; }
    public required string Caption { get; init; }
    public float? Confidence { get; init; }
    public IReadOnlyList<string>? Alternatives { get; init; }
}

/// <summary>
/// Visual QA response
/// </summary>
public sealed record VqaResponse
{
    public required string Id { get; init; }
    public required string Model { get; init; }
    public required string Question { get; init; }
    public required string Answer { get; init; }
    public float? Confidence { get; init; }
}

/// <summary>
/// OCR response
/// </summary>
public sealed record OcrResponse
{
    public required string Id { get; init; }
    public required string Model { get; init; }
    public required string Text { get; init; }
    public IReadOnlyList<OcrBlock>? Blocks { get; init; }
}

/// <summary>
/// OCR text block
/// </summary>
public sealed record OcrBlock
{
    public required string Text { get; init; }
    public float Confidence { get; init; }
    public BoundingBox? BoundingBox { get; init; }
}

/// <summary>
/// Bounding box coordinates
/// </summary>
public sealed record BoundingBox
{
    public float X { get; init; }
    public float Y { get; init; }
    public float Width { get; init; }
    public float Height { get; init; }
}

/// <summary>
/// Object detection response
/// </summary>
public sealed record DetectionResponse
{
    public required string Id { get; init; }
    public required string Model { get; init; }
    public required IReadOnlyList<DetectedObject> Objects { get; init; }
}

/// <summary>
/// Detected object
/// </summary>
public sealed record DetectedObject
{
    public required string Label { get; init; }
    public float Confidence { get; init; }
    [JsonPropertyName("bounding_box")]
    public required BoundingBox BoundingBox { get; init; }
}

/// <summary>
/// Segmentation response
/// </summary>
public sealed record SegmentationResponse
{
    public required string Id { get; init; }
    public required string Model { get; init; }
    public required IReadOnlyList<Segment> Segments { get; init; }
    public string? MaskBase64 { get; init; }
}

/// <summary>
/// Image segment
/// </summary>
public sealed record Segment
{
    public int Id { get; init; }
    public string? Label { get; init; }
    public float? Score { get; init; }
    [JsonPropertyName("bounding_box")]
    public BoundingBox? BoundingBox { get; init; }
    public string? MaskBase64 { get; init; }
}

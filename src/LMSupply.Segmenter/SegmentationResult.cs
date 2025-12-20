namespace LMSupply.Segmenter;

/// <summary>
/// Summary information for a detected segment.
/// </summary>
/// <param name="ClassId">The class ID of the segment.</param>
/// <param name="Label">Human-readable label for the segment.</param>
/// <param name="PixelCount">Number of pixels belonging to this segment.</param>
/// <param name="CoverageRatio">Ratio of image covered by this segment (0.0 to 1.0).</param>
public readonly record struct SegmentSummary(
    int ClassId,
    string Label,
    int PixelCount,
    float CoverageRatio);

/// <summary>
/// Represents the result of semantic segmentation.
/// </summary>
public sealed class SegmentationResult
{
    /// <summary>
    /// Gets the original image width.
    /// </summary>
    public int Width { get; init; }

    /// <summary>
    /// Gets the original image height.
    /// </summary>
    public int Height { get; init; }

    /// <summary>
    /// Gets the class index for each pixel.
    /// Array size is Width * Height, row-major order (y * Width + x).
    /// </summary>
    public int[] ClassMap { get; init; } = [];

    /// <summary>
    /// Gets the confidence scores for the predicted class at each pixel.
    /// Array size is Width * Height, row-major order.
    /// </summary>
    public float[] ConfidenceMap { get; init; } = [];

    /// <summary>
    /// Gets the number of unique classes found in this segmentation.
    /// </summary>
    public int UniqueClassCount => ClassMap.Distinct().Count();

    /// <summary>
    /// Gets the class index at a specific pixel location.
    /// </summary>
    /// <param name="x">X coordinate (column).</param>
    /// <param name="y">Y coordinate (row).</param>
    /// <returns>The class index at the specified location.</returns>
    public int GetClassAt(int x, int y)
    {
        ValidateCoordinates(x, y);
        return ClassMap[y * Width + x];
    }

    /// <summary>
    /// Gets the confidence score at a specific pixel location.
    /// </summary>
    /// <param name="x">X coordinate (column).</param>
    /// <param name="y">Y coordinate (row).</param>
    /// <returns>The confidence score at the specified location.</returns>
    public float GetConfidenceAt(int x, int y)
    {
        ValidateCoordinates(x, y);
        return ConfidenceMap[y * Width + x];
    }

    /// <summary>
    /// Gets a binary mask for a specific class.
    /// </summary>
    /// <param name="classId">The class index to extract.</param>
    /// <returns>A boolean array where true indicates the class is present.</returns>
    public bool[] GetClassMask(int classId)
    {
        var mask = new bool[Width * Height];
        for (int i = 0; i < ClassMap.Length; i++)
        {
            mask[i] = ClassMap[i] == classId;
        }
        return mask;
    }

    /// <summary>
    /// Gets the pixel count for each class.
    /// </summary>
    /// <returns>Dictionary mapping class ID to pixel count.</returns>
    public Dictionary<int, int> GetClassPixelCounts()
    {
        var counts = new Dictionary<int, int>();
        foreach (var classId in ClassMap)
        {
            counts.TryGetValue(classId, out int count);
            counts[classId] = count + 1;
        }
        return counts;
    }

    /// <summary>
    /// Gets the percentage of image covered by each class.
    /// </summary>
    /// <returns>Dictionary mapping class ID to coverage percentage (0-100).</returns>
    public Dictionary<int, float> GetClassCoveragePercentages()
    {
        var totalPixels = (float)(Width * Height);
        var counts = GetClassPixelCounts();
        return counts.ToDictionary(
            kvp => kvp.Key,
            kvp => kvp.Value / totalPixels * 100f);
    }

    /// <summary>
    /// Gets the top N segments sorted by coverage, with optional label resolution.
    /// </summary>
    /// <param name="count">Maximum number of segments to return.</param>
    /// <param name="labels">Optional class labels for resolving class IDs to names.</param>
    /// <returns>List of segment summaries sorted by coverage (highest first).</returns>
    public IReadOnlyList<SegmentSummary> GetTopSegments(int count, IReadOnlyList<string>? labels = null)
    {
        var totalPixels = Width * Height;
        var pixelCounts = GetClassPixelCounts();

        return pixelCounts
            .OrderByDescending(kvp => kvp.Value)
            .Take(count)
            .Select(kvp => new SegmentSummary(
                ClassId: kvp.Key,
                Label: ResolveLabel(kvp.Key, labels),
                PixelCount: kvp.Value,
                CoverageRatio: (float)Math.Round((double)kvp.Value / totalPixels, 4)))
            .ToList();
    }

    private static string ResolveLabel(int classId, IReadOnlyList<string>? labels)
    {
        if (labels is null || classId < 0 || classId >= labels.Count)
            return "unknown";
        return labels[classId];
    }

    private void ValidateCoordinates(int x, int y)
    {
        if (x < 0 || x >= Width)
            throw new ArgumentOutOfRangeException(nameof(x), $"X must be between 0 and {Width - 1}");
        if (y < 0 || y >= Height)
            throw new ArgumentOutOfRangeException(nameof(y), $"Y must be between 0 and {Height - 1}");
    }
}

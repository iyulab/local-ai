namespace LMSupply.Detector;

/// <summary>
/// Extension methods for IDetectorModel.
/// </summary>
public static class DetectorExtensions
{
    /// <summary>
    /// Detects objects in an image byte array with a custom confidence threshold.
    /// </summary>
    /// <param name="detector">The detector model.</param>
    /// <param name="imageData">Byte array containing image data.</param>
    /// <param name="confidenceThreshold">Minimum confidence threshold for filtering results.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of detected objects with confidence >= threshold, sorted by confidence (highest first).</returns>
    public static async Task<IReadOnlyList<DetectionResult>> DetectAsync(
        this IDetectorModel detector,
        byte[] imageData,
        float confidenceThreshold,
        CancellationToken cancellationToken = default)
    {
        var results = await detector.DetectAsync(imageData, cancellationToken);
        return results.Where(r => r.Confidence >= confidenceThreshold).ToList();
    }

    /// <summary>
    /// Detects objects in an image stream with a custom confidence threshold.
    /// </summary>
    /// <param name="detector">The detector model.</param>
    /// <param name="imageStream">Stream containing image data.</param>
    /// <param name="confidenceThreshold">Minimum confidence threshold for filtering results.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of detected objects with confidence >= threshold, sorted by confidence (highest first).</returns>
    public static async Task<IReadOnlyList<DetectionResult>> DetectAsync(
        this IDetectorModel detector,
        Stream imageStream,
        float confidenceThreshold,
        CancellationToken cancellationToken = default)
    {
        var results = await detector.DetectAsync(imageStream, cancellationToken);
        return results.Where(r => r.Confidence >= confidenceThreshold).ToList();
    }

    /// <summary>
    /// Detects objects in an image file with a custom confidence threshold.
    /// </summary>
    /// <param name="detector">The detector model.</param>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="confidenceThreshold">Minimum confidence threshold for filtering results.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of detected objects with confidence >= threshold, sorted by confidence (highest first).</returns>
    public static async Task<IReadOnlyList<DetectionResult>> DetectAsync(
        this IDetectorModel detector,
        string imagePath,
        float confidenceThreshold,
        CancellationToken cancellationToken = default)
    {
        var results = await detector.DetectAsync(imagePath, cancellationToken);
        return results.Where(r => r.Confidence >= confidenceThreshold).ToList();
    }
}

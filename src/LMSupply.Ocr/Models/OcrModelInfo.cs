namespace LMSupply.Ocr.Models;

/// <summary>
/// Combined metadata about an OCR pipeline (detection + recognition).
/// </summary>
/// <param name="DetectionModel">Information about the text detection model.</param>
/// <param name="RecognitionModel">Information about the text recognition model.</param>
public record OcrModelInfo(
    DetectionModelInfo DetectionModel,
    RecognitionModelInfo RecognitionModel) : IModelInfoBase
{
    /// <summary>
    /// Gets the combined model identifier.
    /// </summary>
    public string Id => $"{DetectionModel.Alias}+{RecognitionModel.Alias}";

    /// <summary>
    /// Gets the alias for this OCR pipeline configuration.
    /// </summary>
    public string Alias => Id;

    /// <summary>
    /// Gets the description.
    /// </summary>
    public string? Description => $"OCR pipeline: {DetectionModel.DisplayName} + {RecognitionModel.DisplayName}";

    /// <summary>
    /// Gets the supported language codes from the recognition model.
    /// </summary>
    public IReadOnlyList<string> SupportedLanguages => RecognitionModel.LanguageCodes;
}

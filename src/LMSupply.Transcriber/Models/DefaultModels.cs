namespace LMSupply.Transcriber.Models;

/// <summary>
/// Default transcription model configurations.
/// All models use MIT license (OpenAI Whisper).
/// </summary>
public static class DefaultModels
{
    /// <summary>
    /// Whisper Tiny - Ultra-fast, smallest model.
    /// MIT license, 39M params, ~150MB.
    /// </summary>
    public static TranscriberModelInfo WhisperTiny { get; } = new()
    {
        Id = "onnx-community/whisper-tiny",
        Alias = "fast",
        DisplayName = "Whisper Tiny",
        Architecture = "Whisper",
        ParametersM = 39f,
        SizeBytes = 150_000_000,
        WerLibriSpeech = 7.6f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        HiddenSize = 384,
        IsMultilingual = true,
        Description = "Whisper Tiny for ultra-fast transcription. Best for real-time applications.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Whisper Base - Default balanced model.
    /// MIT license, 74M params, ~290MB.
    /// </summary>
    public static TranscriberModelInfo WhisperBase { get; } = new()
    {
        Id = "onnx-community/whisper-base",
        Alias = "default",
        DisplayName = "Whisper Base",
        Architecture = "Whisper",
        ParametersM = 74f,
        SizeBytes = 290_000_000,
        WerLibriSpeech = 5.0f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        HiddenSize = 512,
        IsMultilingual = true,
        Description = "Whisper Base for balanced speed and accuracy.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Whisper Small - Quality model.
    /// MIT license, 244M params, ~970MB.
    /// </summary>
    public static TranscriberModelInfo WhisperSmall { get; } = new()
    {
        Id = "onnx-community/whisper-small",
        Alias = "quality",
        DisplayName = "Whisper Small",
        Architecture = "Whisper",
        ParametersM = 244f,
        SizeBytes = 970_000_000,
        WerLibriSpeech = 3.4f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        HiddenSize = 768,
        IsMultilingual = true,
        Description = "Whisper Small for higher accuracy transcription.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Whisper Medium - High quality model.
    /// MIT license, 769M params, ~3GB.
    /// </summary>
    public static TranscriberModelInfo WhisperMedium { get; } = new()
    {
        Id = "onnx-community/whisper-medium",
        Alias = "medium",
        DisplayName = "Whisper Medium",
        Architecture = "Whisper",
        ParametersM = 769f,
        SizeBytes = 3_000_000_000,
        WerLibriSpeech = 2.9f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        HiddenSize = 1024,
        IsMultilingual = true,
        Description = "Whisper Medium for high quality transcription.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Whisper Large V3 Turbo - Best balance of quality and speed for "large" alias.
    /// MIT license, 809M params, ~3GB. 8x faster than original Large V3.
    /// Note: Original whisper-large-v3 is gated and requires HuggingFace authentication.
    /// </summary>
    public static TranscriberModelInfo WhisperLargeV3 { get; } = new()
    {
        Id = "onnx-community/whisper-large-v3-turbo",
        Alias = "large",
        DisplayName = "Whisper Large V3 Turbo",
        Architecture = "Whisper",
        ParametersM = 809f,
        SizeBytes = 3_200_000_000,
        WerLibriSpeech = 2.7f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 128, // Large V3 uses 128 mel bins
        HiddenSize = 1280,
        IsMultilingual = true,
        Description = "Whisper Large V3 Turbo for highest quality transcription. 8x faster than original V3.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Whisper Large V3 Turbo - Fast high-quality model.
    /// MIT license, 809M params, ~3GB. 8x faster than Large V3.
    /// Released 2024-10, best for production workloads.
    /// </summary>
    public static TranscriberModelInfo WhisperLargeV3Turbo { get; } = new()
    {
        Id = "onnx-community/whisper-large-v3-turbo",
        Alias = "turbo",
        DisplayName = "Whisper Large V3 Turbo",
        Architecture = "Whisper",
        ParametersM = 809f,
        SizeBytes = 3_200_000_000,
        WerLibriSpeech = 2.7f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 128,
        HiddenSize = 1280,
        IsMultilingual = true,
        Description = "Whisper Large V3 Turbo - 8x faster than V3 with near-equal quality.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Distil-Whisper Large V3 - Distilled fast model.
    /// MIT license, 756M params, ~3GB. 6x faster than Large V3.
    /// Best balance of speed and quality for English.
    /// </summary>
    public static TranscriberModelInfo DistilWhisperLargeV3 { get; } = new()
    {
        Id = "distil-whisper/distil-large-v3",
        Alias = "distil",
        DisplayName = "Distil-Whisper Large V3",
        Architecture = "Whisper",
        ParametersM = 756f,
        SizeBytes = 3_000_000_000,
        WerLibriSpeech = 2.8f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 128,
        HiddenSize = 1280,
        IsMultilingual = false,
        SupportedLanguages = ["en"],
        Description = "Distil-Whisper Large V3 - 6x faster distilled model for English.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Whisper Base English-only - Optimized for English.
    /// MIT license, 74M params, ~290MB.
    /// </summary>
    public static TranscriberModelInfo WhisperBaseEn { get; } = new()
    {
        Id = "onnx-community/whisper-base.en",
        Alias = "english",
        DisplayName = "Whisper Base (English)",
        Architecture = "Whisper",
        ParametersM = 74f,
        SizeBytes = 290_000_000,
        WerLibriSpeech = 4.3f,
        MaxDurationSeconds = 30,
        SampleRate = 16000,
        NumMelBins = 80,
        HiddenSize = 512,
        IsMultilingual = false,
        SupportedLanguages = ["en"],
        Description = "Whisper Base optimized for English-only transcription.",
        License = "MIT",
        DecoderFile = "decoder_model_merged.onnx"
    };

    /// <summary>
    /// Gets all default models.
    /// </summary>
    public static IReadOnlyList<TranscriberModelInfo> All { get; } =
    [
        WhisperTiny,
        WhisperBase,
        WhisperSmall,
        WhisperMedium,
        WhisperLargeV3,
        WhisperLargeV3Turbo,
        DistilWhisperLargeV3,
        WhisperBaseEn
    ];
}

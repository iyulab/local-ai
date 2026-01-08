using LMSupply.Download;
using LMSupply.Exceptions;
using LMSupply.Ocr.Detection;
using LMSupply.Ocr.Models;
using LMSupply.Ocr.Recognition;

namespace LMSupply.Ocr;

/// <summary>
/// Main entry point for loading and using OCR models.
/// </summary>
public static class LocalOcr
{
    /// <summary>
    /// Loads an OCR pipeline with the specified detection and recognition models.
    /// </summary>
    /// <param name="detectionModel">
    /// Detection model ID (e.g., "default", "dbnet-v3") for auto-download,
    /// or a local path to a model file.
    /// </param>
    /// <param name="recognitionModel">
    /// Recognition model ID (e.g., "default", "crnn-en-v3", "crnn-korean-v3") for auto-download,
    /// or a local path to a model file. If null, uses the model for the language hint.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded OCR pipeline ready for inference.</returns>
    public static async Task<IOcr> LoadAsync(
        string detectionModel = "default",
        string? recognitionModel = null,
        OcrOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(detectionModel);
        options ??= new OcrOptions();

        // Resolve detection model
        var (detModelInfo, detModelPath) = await ResolveDetectionModelAsync(
            detectionModel, options, progress, cancellationToken).ConfigureAwait(false);

        // Resolve recognition model based on language hint if not specified
        recognitionModel ??= ModelRegistry.GetRecognitionModelForLanguage(options.LanguageHint).Alias;

        var (recModelInfo, recModelPath, dictPath) = await ResolveRecognitionModelAsync(
            recognitionModel, options, progress, cancellationToken).ConfigureAwait(false);

        // Create detector and recognizer
        var detector = await DbNetDetector.CreateAsync(detModelPath, detModelInfo, options)
            .ConfigureAwait(false);

        var recognizer = await CrnnRecognizer.CreateAsync(recModelPath, dictPath, recModelInfo, options)
            .ConfigureAwait(false);

        // Create and return pipeline
        return await OcrPipeline.CreateAsync(detector, recognizer, detModelInfo, recModelInfo)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Loads an OCR pipeline for a specific language.
    /// </summary>
    /// <param name="languageCode">ISO language code (e.g., "en", "ko", "zh", "ja").</param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded OCR pipeline ready for inference.</returns>
    public static async Task<IOcr> LoadForLanguageAsync(
        string languageCode,
        OcrOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new OcrOptions { LanguageHint = languageCode };
        options.LanguageHint = languageCode;

        var recognitionModel = ModelRegistry.GetRecognitionModelForLanguage(languageCode).Alias;
        return await LoadAsync("default", recognitionModel, options, progress, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets a list of pre-configured detection model IDs available for download.
    /// </summary>
    public static IEnumerable<string> GetAvailableDetectionModels()
        => ModelRegistry.GetAvailableDetectionModels();

    /// <summary>
    /// Gets a list of pre-configured recognition model IDs available for download.
    /// </summary>
    public static IEnumerable<string> GetAvailableRecognitionModels()
        => ModelRegistry.GetAvailableRecognitionModels();

    /// <summary>
    /// Gets a list of supported language codes.
    /// </summary>
    public static IEnumerable<string> GetSupportedLanguages()
        => ModelRegistry.GetSupportedLanguages();

    private static async Task<(DetectionModelInfo info, string modelPath)> ResolveDetectionModelAsync(
        string modelIdOrPath,
        OcrOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Check if it's a local file path
        if (File.Exists(modelIdOrPath))
        {
            // Try to find matching model info or create a default one
            var modelInfo = ModelRegistry.TryGetDetectionModel("default", out var info)
                ? info
                : throw new ModelNotFoundException("No default detection model configured", modelIdOrPath);

            return (modelInfo, modelIdOrPath);
        }

        // Check if it's a known model alias
        if (ModelRegistry.TryGetDetectionModel(modelIdOrPath, out var knownModel))
        {
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir);

            var modelDir = await downloader.DownloadModelAsync(
                knownModel.RepoId,
                files: [knownModel.ModelFile],
                subfolder: knownModel.Subfolder,
                progress: progress,
                cancellationToken: cancellationToken).ConfigureAwait(false);

            var modelPath = Path.Combine(modelDir, knownModel.ModelFile);

            if (!File.Exists(modelPath))
            {
                throw new ModelNotFoundException(
                    $"Detection model file not found: {knownModel.ModelFile}",
                    modelIdOrPath);
            }

            return (knownModel, modelPath);
        }

        // Check if it's a HuggingFace repo ID (contains '/')
        if (modelIdOrPath.Contains('/'))
        {
            return await ResolveHuggingFaceDetectionModelAsync(
                modelIdOrPath, options, progress, cancellationToken).ConfigureAwait(false);
        }

        throw new ModelNotFoundException(
            $"Unknown detection model '{modelIdOrPath}'. Use GetAvailableDetectionModels() to list available models, or provide a HuggingFace repo ID.",
            modelIdOrPath);
    }

    private static async Task<(RecognitionModelInfo info, string modelPath, string dictPath)> ResolveRecognitionModelAsync(
        string modelIdOrPath,
        OcrOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Check if it's a local file path
        if (File.Exists(modelIdOrPath))
        {
            // Try to find matching model info or create a default one
            var modelInfo = ModelRegistry.TryGetRecognitionModel("default", out var info)
                ? info
                : throw new ModelNotFoundException("No default recognition model configured", modelIdOrPath);

            // Look for dictionary file in the same directory
            var dictPath = Path.Combine(Path.GetDirectoryName(modelIdOrPath)!, modelInfo.DictFile);
            if (!File.Exists(dictPath))
            {
                throw new ModelNotFoundException(
                    $"Dictionary file not found: {modelInfo.DictFile}",
                    modelIdOrPath);
            }

            return (modelInfo, modelIdOrPath, dictPath);
        }

        // Check if it's a known model alias
        if (ModelRegistry.TryGetRecognitionModel(modelIdOrPath, out var knownModel))
        {
            var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
            using var downloader = new HuggingFaceDownloader(cacheDir);

            var modelDir = await downloader.DownloadModelAsync(
                knownModel.RepoId,
                files: [knownModel.ModelFile, knownModel.DictFile],
                subfolder: knownModel.Subfolder,
                progress: progress,
                cancellationToken: cancellationToken).ConfigureAwait(false);

            var modelPath = Path.Combine(modelDir, knownModel.ModelFile);
            var dictPath = Path.Combine(modelDir, knownModel.DictFile);

            if (!File.Exists(modelPath))
            {
                throw new ModelNotFoundException(
                    $"Recognition model file not found: {knownModel.ModelFile}",
                    modelIdOrPath);
            }

            if (!File.Exists(dictPath))
            {
                throw new ModelNotFoundException(
                    $"Dictionary file not found: {knownModel.DictFile}",
                    modelIdOrPath);
            }

            return (knownModel, modelPath, dictPath);
        }

        // Check if it's a HuggingFace repo ID (contains '/')
        if (modelIdOrPath.Contains('/'))
        {
            return await ResolveHuggingFaceRecognitionModelAsync(
                modelIdOrPath, options, progress, cancellationToken).ConfigureAwait(false);
        }

        throw new ModelNotFoundException(
            $"Unknown recognition model '{modelIdOrPath}'. Use GetAvailableRecognitionModels() to list available models, or provide a HuggingFace repo ID.",
            modelIdOrPath);
    }

    /// <summary>
    /// Resolves a detection model from a HuggingFace repository.
    /// Searches for common detection model patterns (det.onnx, detection.onnx, etc.)
    /// </summary>
    private static async Task<(DetectionModelInfo info, string modelPath)> ResolveHuggingFaceDetectionModelAsync(
        string repoId,
        OcrOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir);

        // Common detection model file patterns
        string[] detectionPatterns = ["det.onnx", "detection.onnx", "text_detection.onnx", "detector.onnx"];
        string[] subfolderPatterns = ["", "detection", "detection/v5", "detection/v3", "onnx"];

        string? modelPath = null;
        string? foundSubfolder = null;

        // Try to find detection model
        foreach (var subfolder in subfolderPatterns)
        {
            foreach (var pattern in detectionPatterns)
            {
                try
                {
                    var modelDir = await downloader.DownloadModelAsync(
                        repoId,
                        files: [pattern],
                        subfolder: string.IsNullOrEmpty(subfolder) ? null : subfolder,
                        progress: progress,
                        cancellationToken: cancellationToken).ConfigureAwait(false);

                    var candidatePath = Path.Combine(modelDir, pattern);
                    if (File.Exists(candidatePath))
                    {
                        modelPath = candidatePath;
                        foundSubfolder = subfolder;
                        break;
                    }
                }
                catch
                {
                    // File not found, try next pattern
                }
            }
            if (modelPath != null) break;
        }

        if (modelPath == null)
        {
            throw new ModelNotFoundException(
                $"No detection model found in HuggingFace repository '{repoId}'. " +
                $"Expected one of: {string.Join(", ", detectionPatterns)}",
                repoId);
        }

        // Create model info for the discovered model
        var modelInfo = new DetectionModelInfo(
            RepoId: repoId,
            Alias: repoId,
            DisplayName: $"HuggingFace: {repoId}",
            ModelFile: Path.GetFileName(modelPath),
            InputWidth: 960,
            InputHeight: 960)
        {
            Subfolder = foundSubfolder
        };

        return (modelInfo, modelPath);
    }

    /// <summary>
    /// Resolves a recognition model from a HuggingFace repository.
    /// Searches for common recognition model patterns (rec.onnx, recognition.onnx, etc.)
    /// </summary>
    private static async Task<(RecognitionModelInfo info, string modelPath, string dictPath)> ResolveHuggingFaceRecognitionModelAsync(
        string repoId,
        OcrOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir);

        // Parse repo ID for language hint (e.g., "org/paddleocr-japanese" -> try japanese subfolder)
        var repoName = repoId.Split('/').Last().ToLowerInvariant();

        // Common recognition model file patterns
        string[] recognitionPatterns = ["rec.onnx", "recognition.onnx", "text_recognition.onnx", "recognizer.onnx"];
        string[] dictPatterns = ["dict.txt", "dictionary.txt", "keys.txt", "vocab.txt"];

        // Build subfolder patterns based on language hint
        var subfolderPatterns = new List<string> { "" };
        if (options.LanguageHint != null)
        {
            var langSubfolders = GetLanguageSubfolders(options.LanguageHint);
            subfolderPatterns.InsertRange(0, langSubfolders);
        }
        subfolderPatterns.AddRange(["languages/english", "languages/latin", "onnx", "recognition"]);

        string? modelPath = null;
        string? dictPath = null;
        string? foundSubfolder = null;

        // Try to find recognition model and dictionary
        foreach (var subfolder in subfolderPatterns.Distinct())
        {
            foreach (var recPattern in recognitionPatterns)
            {
                foreach (var dictPattern in dictPatterns)
                {
                    try
                    {
                        var modelDir = await downloader.DownloadModelAsync(
                            repoId,
                            files: [recPattern, dictPattern],
                            subfolder: string.IsNullOrEmpty(subfolder) ? null : subfolder,
                            progress: progress,
                            cancellationToken: cancellationToken).ConfigureAwait(false);

                        var candidateModelPath = Path.Combine(modelDir, recPattern);
                        var candidateDictPath = Path.Combine(modelDir, dictPattern);

                        if (File.Exists(candidateModelPath) && File.Exists(candidateDictPath))
                        {
                            modelPath = candidateModelPath;
                            dictPath = candidateDictPath;
                            foundSubfolder = subfolder;
                            break;
                        }
                    }
                    catch
                    {
                        // Files not found, try next pattern
                    }
                }
                if (modelPath != null) break;
            }
            if (modelPath != null) break;
        }

        if (modelPath == null || dictPath == null)
        {
            throw new ModelNotFoundException(
                $"No recognition model found in HuggingFace repository '{repoId}'. " +
                $"Expected model file ({string.Join(", ", recognitionPatterns)}) and dictionary file ({string.Join(", ", dictPatterns)})",
                repoId);
        }

        // Create model info for the discovered model
        var modelInfo = new RecognitionModelInfo(
            RepoId: repoId,
            Alias: repoId,
            DisplayName: $"HuggingFace: {repoId}",
            ModelFile: Path.GetFileName(modelPath),
            DictFile: Path.GetFileName(dictPath),
            LanguageCodes: [options.LanguageHint ?? "en"])
        {
            Subfolder = foundSubfolder
        };

        return (modelInfo, modelPath, dictPath);
    }

    /// <summary>
    /// Gets possible subfolder names for a language code.
    /// </summary>
    private static string[] GetLanguageSubfolders(string languageCode)
    {
        var lang = languageCode.ToLowerInvariant().Split('-')[0];
        return lang switch
        {
            "en" => ["languages/english", "english", "en"],
            "ko" => ["languages/korean", "korean", "ko"],
            "zh" => ["languages/chinese", "chinese", "zh", "ch"],
            "ja" => ["languages/japanese", "japanese", "ja", "japan"],
            "ar" => ["languages/arabic", "arabic", "ar"],
            "ru" => ["languages/cyrillic", "languages/eslav", "cyrillic", "russian", "ru"],
            "de" or "fr" or "es" or "it" or "pt" => ["languages/latin", "latin"],
            "hi" => ["languages/hindi", "languages/devanagari", "hindi", "devanagari"],
            "th" => ["languages/thai", "thai", "th"],
            "el" => ["languages/greek", "greek", "el"],
            "ta" => ["languages/tamil", "tamil", "ta"],
            "te" => ["languages/telugu", "telugu", "te"],
            _ => [$"languages/{lang}", lang]
        };
    }
}

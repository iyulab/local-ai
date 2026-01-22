using System.Diagnostics;
using System.Runtime.CompilerServices;
using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Inference;
using LMSupply.Transcriber.Audio;
using LMSupply.Transcriber.Decoding;
using LMSupply.Transcriber.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Transcriber.Core;

/// <summary>
/// ONNX-based implementation of Whisper transcription model.
/// </summary>
internal sealed class OnnxTranscriberModel : ITranscriberModel
{
    private readonly TranscriberOptions _options;
    private readonly SemaphoreSlim _lock = new(1, 1);

    private InferenceSession? _encoderSession;
    private InferenceSession? _decoderSession;
    private SessionCreationResult? _encoderSessionInfo;
    private WhisperTokenizer? _tokenizer;
    private WhisperDecoder? _decoder;
    private TranscriberModelInfo? _modelInfo;
    private string? _modelPath;
    private bool _isInitialized;
    private bool _isDisposed;

    /// <inheritdoc />
    public string ModelId => _modelInfo?.Id ?? _options.ModelId;

    public string? Language => null; // Auto-detected per transcription

    /// <summary>
    /// Gets whether GPU acceleration is actually being used for inference.
    /// </summary>
    public bool IsGpuActive => _encoderSessionInfo?.IsGpuActive ?? false;

    /// <summary>
    /// Gets the list of active execution providers for the encoder session.
    /// </summary>
    public IReadOnlyList<string> ActiveProviders => _encoderSessionInfo?.ActiveProviders ?? [];

    /// <summary>
    /// Gets the execution provider that was requested.
    /// </summary>
    public ExecutionProvider RequestedProvider => _encoderSessionInfo?.RequestedProvider ?? ExecutionProvider.Auto;

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => _modelInfo?.SizeBytes * 2;

    public OnnxTranscriberModel(TranscriberOptions options)
    {
        _options = options.Clone();
    }

    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);
    }

    public TranscriberModelInfo? GetModelInfo() => _modelInfo;

    public async Task<TranscriptionResult> TranscribeAsync(
        string audioPath,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var samples = await AudioProcessor.LoadAudioAsync(audioPath, cancellationToken);
        return await TranscribeCoreAsync(samples, options, cancellationToken);
    }

    public async Task<TranscriptionResult> TranscribeAsync(
        Stream audioStream,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var samples = await AudioProcessor.LoadAudioAsync(audioStream, cancellationToken);
        return await TranscribeCoreAsync(samples, options, cancellationToken);
    }

    public async Task<TranscriptionResult> TranscribeAsync(
        byte[] audioData,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var samples = await AudioProcessor.LoadAudioAsync(audioData, cancellationToken);
        return await TranscribeCoreAsync(samples, options, cancellationToken);
    }

    public async IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(
        string audioPath,
        TranscribeOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);
        var samples = await AudioProcessor.LoadAudioAsync(audioPath, cancellationToken);

        var chunks = AudioProcessor.SplitIntoChunks(samples);
        var segmentId = 0;

        foreach (var chunk in chunks)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var chunkStartTime = segmentId * 30.0;
            var result = await TranscribeChunkAsync(chunk, options, cancellationToken);

            foreach (var segment in result.Segments)
            {
                yield return new TranscriptionSegment
                {
                    Id = segmentId++,
                    Start = chunkStartTime + segment.Start,
                    End = chunkStartTime + segment.End,
                    Text = segment.Text,
                    AvgLogProb = segment.AvgLogProb,
                    NoSpeechProb = segment.NoSpeechProb,
                    CompressionRatio = segment.CompressionRatio
                };
            }
        }
    }

    private async Task<TranscriptionResult> TranscribeCoreAsync(
        float[] samples,
        TranscribeOptions? options,
        CancellationToken cancellationToken)
    {
        await EnsureInitializedAsync(cancellationToken);

        var sw = Stopwatch.StartNew();
        var duration = AudioProcessor.GetDurationSeconds(samples);

        // For short audio, process as single chunk
        if (samples.Length <= 480000) // 30 seconds
        {
            var result = await TranscribeChunkAsync(samples, options, cancellationToken);
            sw.Stop();

            return new TranscriptionResult
            {
                Text = result.Text,
                Language = result.Language,
                LanguageProbability = result.LanguageProbability,
                Segments = result.Segments,
                DurationSeconds = duration,
                InferenceTimeMs = sw.Elapsed.TotalMilliseconds
            };
        }

        // For longer audio, process in chunks
        var chunks = AudioProcessor.SplitIntoChunks(samples);
        var allSegments = new List<TranscriptionSegment>();
        var textParts = new List<string>();
        string? detectedLanguage = null;
        float? languageProb = null;

        for (int i = 0; i < chunks.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var chunkResult = await TranscribeChunkAsync(chunks[i], options, cancellationToken);
            var chunkStartTime = i * 30.0;

            if (i == 0)
            {
                detectedLanguage = chunkResult.Language;
                languageProb = chunkResult.LanguageProbability;
            }

            textParts.Add(chunkResult.Text);

            foreach (var segment in chunkResult.Segments)
            {
                allSegments.Add(new TranscriptionSegment
                {
                    Id = allSegments.Count,
                    Start = chunkStartTime + segment.Start,
                    End = chunkStartTime + segment.End,
                    Text = segment.Text,
                    AvgLogProb = segment.AvgLogProb,
                    NoSpeechProb = segment.NoSpeechProb,
                    CompressionRatio = segment.CompressionRatio
                });
            }
        }

        sw.Stop();

        return new TranscriptionResult
        {
            Text = string.Join(" ", textParts),
            Language = detectedLanguage ?? "en",
            LanguageProbability = languageProb,
            Segments = allSegments,
            DurationSeconds = duration,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds
        };
    }

    private async Task<TranscriptionResult> TranscribeChunkAsync(
        float[] samples,
        TranscribeOptions? options,
        CancellationToken cancellationToken)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            // Log language settings for debugging
            Debug.WriteLine($"[OnnxTranscriberModel] Transcribing chunk - Language: {options?.Language ?? "auto-detect"}, " +
                $"WordTimestamps: {options?.WordTimestamps ?? false}");

            // Compute mel spectrogram
            var numMelBins = _modelInfo?.NumMelBins ?? 80;
            var melSpec = AudioProcessor.ComputeLogMelSpectrogram(samples, numMelBins);

            // Run encoder
            var encoderOutput = await RunEncoderAsync(melSpec, numMelBins, cancellationToken);

            // Run decoder with greedy decoding
            var (text, segments) = await RunDecoderAsync(encoderOutput, options, cancellationToken);

            var resultLanguage = options?.Language ?? "en"; // TODO: Implement language detection
            Debug.WriteLine($"[OnnxTranscriberModel] Transcription result - Language: {resultLanguage}, " +
                $"Text length: {text.Length}, Segments: {segments.Count}");

            return new TranscriptionResult
            {
                Text = text,
                Language = resultLanguage,
                LanguageProbability = null,
                Segments = segments
            };
        }
        finally
        {
            _lock.Release();
        }
    }

    private Task<float[]> RunEncoderAsync(float[] melSpec, int numMelBins, CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            var inputTensor = new DenseTensor<float>(melSpec, [1, numMelBins, 3000]);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_features", inputTensor)
            };

            using var results = _encoderSession!.Run(inputs);
            var output = results.First().AsTensor<float>();

            // Copy tensor output to array
            return output.ToArray();
        }, cancellationToken);
    }

    private async Task<(string text, List<TranscriptionSegment> segments)> RunDecoderAsync(
        float[] encoderOutput,
        TranscribeOptions? options,
        CancellationToken cancellationToken)
    {
        if (_decoder == null)
        {
            throw new InvalidOperationException("Decoder not initialized");
        }

        // Get encoder output dimensions
        // Whisper encoder output shape: [1, sequence_length, hidden_size]
        // sequence_length = 1500 (for 30s audio), hidden_size = d_model from config
        var hiddenSize = _modelInfo?.HiddenSize ?? 512; // Default for base model
        var sequenceLength = encoderOutput.Length / hiddenSize;

        var result = await _decoder.DecodeAsync(
            encoderOutput,
            sequenceLength,
            hiddenSize,
            options,
            cancellationToken);

        return (result.Text, result.Segments);
    }

    private async Task EnsureInitializedAsync(CancellationToken cancellationToken)
    {
        if (_isInitialized) return;

        await _lock.WaitAsync(cancellationToken);
        try
        {
            if (_isInitialized) return;

            // Resolve model info
            if (!TranscriberModelRegistry.Default.TryGet(_options.ModelId, out _modelInfo))
            {
                // Treat as HuggingFace model ID
                _modelInfo = new TranscriberModelInfo
                {
                    Id = _options.ModelId,
                    Alias = _options.ModelId,
                    DisplayName = _options.ModelId,
                    Architecture = "Whisper"
                };
            }

            // Download model if needed and get discovery result
            var (baseModelPath, discovery) = await ResolveModelPathAsync(cancellationToken);

            // Determine encoder/decoder paths using discovery result or fallback to legacy behavior
            string encoderPath;
            string decoderPath;
            string tokenizerPath;

            if (discovery != null)
            {
                // Use discovery result for accurate path resolution (handles subfolder structures)
                encoderPath = discovery.GetEncoderPath(baseModelPath)
                    ?? Path.Combine(discovery.GetOnnxDirectory(baseModelPath), _modelInfo!.EncoderFile);
                decoderPath = discovery.GetDecoderPath(baseModelPath)
                    ?? Path.Combine(discovery.GetOnnxDirectory(baseModelPath), _modelInfo!.DecoderFile);
                // Tokenizer files are typically in the base model directory
                tokenizerPath = baseModelPath;

                Debug.WriteLine($"[OnnxTranscriberModel] Using discovery-based paths - Subfolder: {discovery.Subfolder ?? "(root)"}, " +
                    $"Encoder: {Path.GetFileName(encoderPath)}, Decoder: {Path.GetFileName(decoderPath)}");
            }
            else
            {
                // Fallback for local paths without discovery
                encoderPath = Path.Combine(baseModelPath, _modelInfo!.EncoderFile);
                decoderPath = Path.Combine(baseModelPath, _modelInfo.DecoderFile);
                tokenizerPath = baseModelPath;
            }

            // Load encoder with GPU provider verification
            if (!File.Exists(encoderPath))
            {
                throw new FileNotFoundException($"Encoder model not found: {encoderPath}");
            }

            _encoderSessionInfo = await OnnxSessionFactory.CreateWithInfoAsync(
                encoderPath,
                _options.Provider,
                ConfigureSessionOptions,
                cancellationToken: cancellationToken);
            _encoderSession = _encoderSessionInfo.Session;

            // Log GPU provider status
            Debug.WriteLine($"[OnnxTranscriberModel] Encoder loaded - Requested: {_encoderSessionInfo.RequestedProvider}, " +
                $"Active providers: [{string.Join(", ", _encoderSessionInfo.ActiveProviders)}], GPU active: {_encoderSessionInfo.IsGpuActive}");

            if (_options.Provider != ExecutionProvider.Cpu && !_encoderSessionInfo.IsGpuActive)
            {
                Debug.WriteLine("[OnnxTranscriberModel] WARNING: GPU provider was requested but only CPU is active. " +
                    "Check CUDA/DirectML installation and GPU availability.");
            }

            // Load decoder if available - use same provider as encoder
            if (File.Exists(decoderPath))
            {
                var decoderSessionInfo = await OnnxSessionFactory.CreateWithInfoAsync(
                    decoderPath,
                    _options.Provider,
                    ConfigureSessionOptions,
                    cancellationToken: cancellationToken);
                _decoderSession = decoderSessionInfo.Session;

                Debug.WriteLine($"[OnnxTranscriberModel] Decoder loaded - Requested: {decoderSessionInfo.RequestedProvider}, " +
                    $"Active providers: [{string.Join(", ", decoderSessionInfo.ActiveProviders)}], GPU active: {decoderSessionInfo.IsGpuActive}");

                if (_options.Provider != ExecutionProvider.Cpu && !decoderSessionInfo.IsGpuActive)
                {
                    Debug.WriteLine("[OnnxTranscriberModel] WARNING: Decoder GPU provider was requested but only CPU is active.");
                }
            }

            // Store model path and load tokenizer
            _modelPath = tokenizerPath;
            _tokenizer = await WhisperTokenizer.LoadAsync(tokenizerPath, cancellationToken);

            // Create decoder if decoder session is available
            if (_decoderSession != null)
            {
                _decoder = new WhisperDecoder(_decoderSession, _tokenizer);
            }

            _isInitialized = true;
        }
        finally
        {
            _lock.Release();
        }
    }

    private async Task<(string modelPath, ModelDiscoveryResult? discovery)> ResolveModelPathAsync(CancellationToken cancellationToken)
    {
        // If it's a local directory path, return it without discovery
        if (Directory.Exists(_modelInfo!.Id))
        {
            return (_modelInfo.Id, null);
        }

        // Check if parent directory exists (for file paths)
        var parentDir = Path.GetDirectoryName(_modelInfo.Id);
        if (parentDir != null && Directory.Exists(parentDir))
        {
            return (parentDir, null);
        }

        // Download from HuggingFace using discovery for complete file set
        var cacheDir = _options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir);

        // Use discovery-based download to automatically find all model files
        // including external data files (*.onnx_data) for large models
        var (modelPath, discovery) = await downloader.DownloadWithDiscoveryAsync(
            _modelInfo.Id,
            preferences: new ModelPreferences
            {
                // Prefer onnx subfolder for Whisper models
                PreferredSubfolder = "onnx"
            },
            cancellationToken: cancellationToken);

        return (modelPath, discovery);
    }

    private void ConfigureSessionOptions(SessionOptions options)
    {
        if (_options.ThreadCount.HasValue)
        {
            options.IntraOpNumThreads = _options.ThreadCount.Value;
            options.InterOpNumThreads = _options.ThreadCount.Value;
        }
    }

    public ValueTask DisposeAsync()
    {
        if (_isDisposed) return ValueTask.CompletedTask;
        _isDisposed = true;

        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
        _tokenizer?.Dispose();
        _lock.Dispose();
        return ValueTask.CompletedTask;
    }
}

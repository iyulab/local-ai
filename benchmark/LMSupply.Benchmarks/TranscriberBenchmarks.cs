using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using LMSupply;
using LMSupply.Transcriber;

namespace LMSupply.Benchmarks;

/// <summary>
/// Benchmarks for LMSupply.Transcriber performance testing.
/// Measures model loading, warmup, and transcription times.
/// </summary>
[Config(typeof(TranscriberBenchmarkConfig))]
[MemoryDiagnoser]
[MarkdownExporter]
[HtmlExporter]
public class TranscriberBenchmarks
{
    private ITranscriberModel? _model;
    private byte[]? _audioData;

    /// <summary>
    /// Model alias to benchmark.
    /// </summary>
    [Params("fast", "default")]
    public string ModelAlias { get; set; } = "default";

    /// <summary>
    /// Whether to use segment timestamps.
    /// </summary>
    [Params(false, true)]
    public bool UseTimestamps { get; set; }

    [GlobalSetup]
    public async Task GlobalSetup()
    {
        // Load model once for all iterations
        var options = new TranscriberOptions
        {
            ModelId = ModelAlias,
            Provider = ExecutionProvider.Auto
        };

        _model = await LocalTranscriber.LoadAsync(options);
        await _model.WarmupAsync();

        // Generate synthetic audio data (silence) for consistent benchmarking
        // In production, you would use real audio files
        _audioData = GenerateSilentWavData(5.0); // 5 seconds of silence

        Console.WriteLine($"[Setup] Model: {_model.ModelId}");
        Console.WriteLine($"[Setup] GPU Active: {_model.IsGpuActive}");
        Console.WriteLine($"[Setup] Providers: [{string.Join(", ", _model.ActiveProviders)}]");
    }

    [GlobalCleanup]
    public async Task GlobalCleanup()
    {
        if (_model != null)
        {
            await _model.DisposeAsync();
        }
    }

    /// <summary>
    /// Benchmark: Transcribe audio data.
    /// </summary>
    [Benchmark(Description = "Transcribe audio")]
    public async Task<TranscriptionResult> TranscribeAudio()
    {
        var options = new TranscribeOptions
        {
            WordTimestamps = UseTimestamps
        };

        return await _model!.TranscribeAsync(_audioData!, options);
    }

    /// <summary>
    /// Generates a valid WAV file with silence for benchmarking.
    /// </summary>
    private static byte[] GenerateSilentWavData(double durationSeconds)
    {
        const int sampleRate = 16000;
        const int bitsPerSample = 16;
        const int channels = 1;

        var numSamples = (int)(sampleRate * durationSeconds);
        var dataSize = numSamples * channels * (bitsPerSample / 8);
        var fileSize = 44 + dataSize;

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // RIFF header
        writer.Write("RIFF"u8);
        writer.Write(fileSize - 8);
        writer.Write("WAVE"u8);

        // fmt chunk
        writer.Write("fmt "u8);
        writer.Write(16); // chunk size
        writer.Write((short)1); // PCM format
        writer.Write((short)channels);
        writer.Write(sampleRate);
        writer.Write(sampleRate * channels * (bitsPerSample / 8)); // byte rate
        writer.Write((short)(channels * (bitsPerSample / 8))); // block align
        writer.Write((short)bitsPerSample);

        // data chunk
        writer.Write("data"u8);
        writer.Write(dataSize);

        // Write silence (zeros)
        var buffer = new byte[dataSize];
        writer.Write(buffer);

        return ms.ToArray();
    }
}

/// <summary>
/// Custom configuration for Transcriber benchmarks.
/// </summary>
public class TranscriberBenchmarkConfig : ManualConfig
{
    public TranscriberBenchmarkConfig()
    {
        // Use shorter run for development/testing
        // For accurate results, use default or increase iterations
        AddJob(Job.ShortRun
            .WithWarmupCount(1)
            .WithIterationCount(3)
            .WithId("ShortRun"));

        // Add custom columns
        AddColumn(StatisticColumn.Mean);
        AddColumn(StatisticColumn.StdDev);
        AddColumn(StatisticColumn.Median);
    }
}

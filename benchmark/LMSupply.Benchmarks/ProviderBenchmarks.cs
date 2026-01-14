using BenchmarkDotNet.Attributes;
using LMSupply;
using LMSupply.Transcriber;

namespace LMSupply.Benchmarks;

/// <summary>
/// Benchmarks comparing CPU vs GPU provider performance.
/// Useful for diagnosing GPU utilization issues.
/// Uses Transcriber which has GPU status properties for verification.
/// </summary>
[Config(typeof(TranscriberBenchmarkConfig))]
[MemoryDiagnoser]
[MarkdownExporter]
[HtmlExporter]
public class ProviderBenchmarks
{
    private ITranscriberModel? _cpuModel;
    private ITranscriberModel? _gpuModel;
    private byte[]? _audioData;

    [GlobalSetup]
    public async Task GlobalSetup()
    {
        // Generate synthetic audio data for consistent benchmarking
        _audioData = GenerateSilentWavData(2.0); // 2 seconds of silence

        // Load CPU model
        Console.WriteLine("[Setup] Loading CPU model...");
        var cpuOptions = new TranscriberOptions
        {
            ModelId = "fast",
            Provider = ExecutionProvider.Cpu
        };
        _cpuModel = await LocalTranscriber.LoadAsync(cpuOptions);
        await _cpuModel.WarmupAsync();
        Console.WriteLine($"[Setup] CPU Model Providers: [{string.Join(", ", _cpuModel.ActiveProviders)}]");
        Console.WriteLine($"[Setup] CPU GPU Active: {_cpuModel.IsGpuActive}");

        // Load GPU model (Auto will try GPU first)
        Console.WriteLine("[Setup] Loading GPU/Auto model...");
        var gpuOptions = new TranscriberOptions
        {
            ModelId = "fast",
            Provider = ExecutionProvider.Auto
        };
        _gpuModel = await LocalTranscriber.LoadAsync(gpuOptions);
        await _gpuModel.WarmupAsync();
        Console.WriteLine($"[Setup] GPU Model Providers: [{string.Join(", ", _gpuModel.ActiveProviders)}]");
        Console.WriteLine($"[Setup] GPU Active: {_gpuModel.IsGpuActive}");

        if (!_gpuModel.IsGpuActive)
        {
            Console.WriteLine("[Warning] GPU is not active. Both benchmarks will use CPU.");
        }
    }

    [GlobalCleanup]
    public async Task GlobalCleanup()
    {
        if (_cpuModel != null) await _cpuModel.DisposeAsync();
        if (_gpuModel != null) await _gpuModel.DisposeAsync();
    }

    /// <summary>
    /// Benchmark: CPU provider transcription.
    /// </summary>
    [Benchmark(Baseline = true, Description = "CPU Provider")]
    public async Task<TranscriptionResult> CpuTranscription()
    {
        return await _cpuModel!.TranscribeAsync(_audioData!);
    }

    /// <summary>
    /// Benchmark: GPU/Auto provider transcription.
    /// </summary>
    [Benchmark(Description = "GPU/Auto Provider")]
    public async Task<TranscriptionResult> GpuTranscription()
    {
        return await _gpuModel!.TranscribeAsync(_audioData!);
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

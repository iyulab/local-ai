using BenchmarkDotNet.Attributes;
using LMSupply;
using LMSupply.Embedder;

namespace LMSupply.Benchmarks;

/// <summary>
/// Benchmarks for LMSupply.Embedder performance testing.
/// Measures model loading, single embedding, and batch embedding times.
/// </summary>
[Config(typeof(TranscriberBenchmarkConfig))]
[MemoryDiagnoser]
[MarkdownExporter]
[HtmlExporter]
public class EmbedderBenchmarks
{
    private IEmbeddingModel? _model;
    private string[]? _sentences;

    /// <summary>
    /// Model alias to benchmark.
    /// </summary>
    [Params("fast", "default")]
    public string ModelAlias { get; set; } = "default";

    /// <summary>
    /// Batch size for batch embedding tests.
    /// </summary>
    [Params(1, 8, 32)]
    public int BatchSize { get; set; } = 8;

    [GlobalSetup]
    public async Task GlobalSetup()
    {
        // Load model once for all iterations
        var options = new EmbedderOptions
        {
            Provider = ExecutionProvider.Auto
        };

        _model = await LocalEmbedder.LoadAsync(ModelAlias, options);
        await _model.WarmupAsync();

        // Generate test sentences
        _sentences = GenerateTestSentences(BatchSize);

        Console.WriteLine($"[Setup] Model: {_model.ModelId}");
        Console.WriteLine($"[Setup] Dimensions: {_model.Dimensions}");
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
    /// Benchmark: Embed a single sentence.
    /// </summary>
    [Benchmark(Description = "Single embedding")]
    public async Task<float[]> EmbedSingle()
    {
        return await _model!.EmbedAsync(_sentences![0]);
    }

    /// <summary>
    /// Benchmark: Embed a batch of sentences.
    /// </summary>
    [Benchmark(Description = "Batch embedding")]
    public async Task<float[][]> EmbedBatch()
    {
        return await _model!.EmbedAsync(_sentences!);
    }

    private static string[] GenerateTestSentences(int count)
    {
        var baseSentences = new[]
        {
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require significant computational resources.",
            "Natural language processing has revolutionized human-computer interaction.",
            "Deep learning enables computers to learn from large amounts of data.",
            "Semantic search improves information retrieval accuracy.",
            "Vector embeddings capture the meaning of text in numerical form.",
            "Neural networks are inspired by the structure of the human brain.",
            "Transformers have become the dominant architecture for NLP tasks."
        };

        var sentences = new string[count];
        for (int i = 0; i < count; i++)
        {
            sentences[i] = baseSentences[i % baseSentences.Length];
        }
        return sentences;
    }
}

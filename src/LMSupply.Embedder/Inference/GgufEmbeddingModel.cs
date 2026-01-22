using LLama;
using LLama.Common;
using LMSupply.Embedder.Utils;
using LMSupply.Llama;

// ExecutionProvider is in the root LMSupply namespace
using ExecutionProvider = LMSupply.ExecutionProvider;

namespace LMSupply.Embedder.Inference;

/// <summary>
/// GGUF model implementation for embeddings using LLamaSharp.
/// Uses llama.cpp's built-in embedding support for GGUF models with embedding architecture.
/// </summary>
internal sealed class GgufEmbeddingModel : IEmbeddingModel
{
    private readonly LLamaWeights _weights;
    private readonly LLamaEmbedder _embedder;
    private readonly EmbedderOptions _options;
    private readonly string _modelPath;
    private readonly bool _isGpuActive;
    private bool _disposed;

    private GgufEmbeddingModel(
        string modelId,
        string modelPath,
        LLamaWeights weights,
        LLamaEmbedder embedder,
        int dimensions,
        EmbedderOptions options,
        bool isGpuActive)
    {
        ModelId = modelId;
        _modelPath = modelPath;
        _weights = weights;
        _embedder = embedder;
        Dimensions = dimensions;
        _options = options;
        _isGpuActive = isGpuActive;
    }

    /// <summary>
    /// Loads a GGUF embedding model from the specified path.
    /// </summary>
    /// <param name="modelId">Model identifier.</param>
    /// <param name="modelPath">Full path to the GGUF file.</param>
    /// <param name="options">Embedder options.</param>
    /// <param name="progress">Progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Loaded embedding model.</returns>
    public static async Task<GgufEmbeddingModel> LoadAsync(
        string modelId,
        string modelPath,
        EmbedderOptions options,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // 1. Ensure llama.cpp runtime is initialized
        progress?.Report(new DownloadProgress
        {
            FileName = "llama.cpp runtime",
            BytesDownloaded = 0,
            TotalBytes = 0
        });

        await LlamaRuntimeManager.Instance.EnsureInitializedAsync(
            options.Provider,
            progress,
            cancellationToken);

        progress?.Report(new DownloadProgress
        {
            FileName = Path.GetFileName(modelPath),
            BytesDownloaded = 0,
            TotalBytes = 100
        });

        // 2. Configure model parameters for embedding mode
        var modelParams = new ModelParams(modelPath)
        {
            Embeddings = true, // Enable embedding mode
            ContextSize = (uint)options.MaxSequenceLength,
            GpuLayerCount = CalculateGpuLayers(options.Provider),
            BatchSize = 512
        };

        // 3. Load model weights
        progress?.Report(new DownloadProgress
        {
            FileName = Path.GetFileName(modelPath),
            BytesDownloaded = 30,
            TotalBytes = 100
        });

        var weights = await LLamaWeights.LoadFromFileAsync(modelParams, cancellationToken);

        // 4. Create embedder
        progress?.Report(new DownloadProgress
        {
            FileName = Path.GetFileName(modelPath),
            BytesDownloaded = 80,
            TotalBytes = 100
        });

        var embedder = new LLamaEmbedder(weights, modelParams);

        // Get embedding dimension from the model
        var dimensions = embedder.EmbeddingSize;

        progress?.Report(new DownloadProgress
        {
            FileName = Path.GetFileName(modelPath),
            BytesDownloaded = 100,
            TotalBytes = 100
        });

        // Determine if GPU is active based on GPU layer count
        var isGpuActive = CalculateGpuLayers(options.Provider) != 0;

        return new GgufEmbeddingModel(
            modelId,
            modelPath,
            weights,
            embedder,
            dimensions,
            options,
            isGpuActive);
    }

    /// <inheritdoc />
    public string ModelId { get; }

    /// <inheritdoc />
    public int Dimensions { get; }

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => File.Exists(_modelPath) ? new FileInfo(_modelPath).Length * 2 : null;

    /// <inheritdoc />
    public bool IsGpuActive => _isGpuActive;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => _isGpuActive
        ? new[] { "LLamaSharpGPU", "CPU" }
        : new[] { "CPU" };

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _options.Provider;

    /// <inheritdoc />
    public async ValueTask<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        var embeddings = await _embedder.GetEmbeddings(text, cancellationToken);

        // LLamaEmbedder.GetEmbeddings returns IReadOnlyList<float[]>, take the first one
        return embeddings.Count > 0 ? embeddings[0] : [];
    }

    /// <inheritdoc />
    public async ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        var results = new float[texts.Count][];

        // Process texts sequentially (LLamaEmbedder doesn't support batch processing directly)
        for (var i = 0; i < texts.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var embeddings = await _embedder.GetEmbeddings(texts[i], cancellationToken);
            results[i] = embeddings.Count > 0 ? embeddings[0] : [];
        }

        return results;
    }

    /// <inheritdoc />
    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        // Perform a minimal embedding to warm up the model
        await EmbedAsync("warmup", cancellationToken);
    }

    /// <inheritdoc />
    public ModelInfo? GetModelInfo() => new()
    {
        RepoId = ModelId,
        Dimensions = Dimensions,
        MaxSequenceLength = _options.MaxSequenceLength,
        PoolingMode = PoolingMode.Mean, // GGUF models handle pooling internally
        DoLowerCase = false,
        Description = "GGUF embedding model via LLamaSharp"
    };

    private static int CalculateGpuLayers(ExecutionProvider provider)
    {
        return provider switch
        {
            ExecutionProvider.Cpu => 0,
            ExecutionProvider.Auto => -1, // All layers on GPU
            ExecutionProvider.Cuda => -1,
            ExecutionProvider.DirectML => 0, // DirectML uses different API
            ExecutionProvider.CoreML => -1,
            _ => -1
        };
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return ValueTask.CompletedTask;
        }

        _disposed = true;
        _embedder.Dispose();
        _weights.Dispose();

        return ValueTask.CompletedTask;
    }
}

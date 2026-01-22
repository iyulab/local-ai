using System.Runtime.CompilerServices;
using LLama;
using LLama.Common;
using LLama.Sampling;
using LMSupply.Download;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;
using LMSupply.Llama;
using LMSupply.Runtime;

namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// GGUF model implementation using LLamaSharp (llama.cpp binding).
/// </summary>
internal sealed class GgufGeneratorModel : IGeneratorModel
{
    private readonly LLamaWeights _weights;
    private readonly LLamaContext _context;
    private readonly IChatFormatter _chatFormatter;
    private readonly GeneratorOptions _options;
    private readonly string _modelPath;
    private readonly SemaphoreSlim _concurrencyLimiter;
    private bool _disposed;

    private GgufGeneratorModel(
        string modelId,
        string modelPath,
        LLamaWeights weights,
        LLamaContext context,
        IChatFormatter chatFormatter,
        GeneratorOptions options,
        int maxContextLength)
    {
        ModelId = modelId;
        _modelPath = modelPath;
        _weights = weights;
        _context = context;
        _chatFormatter = chatFormatter;
        _options = options;
        MaxContextLength = maxContextLength;

        // Initialize concurrency limiter
        _concurrencyLimiter = new SemaphoreSlim(
            Math.Max(1, options.MaxConcurrentRequests),
            Math.Max(1, options.MaxConcurrentRequests));
    }

    /// <summary>
    /// Loads a GGUF model from the specified path.
    /// </summary>
    /// <param name="modelId">Model identifier.</param>
    /// <param name="modelPath">Full path to the GGUF file.</param>
    /// <param name="chatFormatter">Chat formatter for the model.</param>
    /// <param name="options">Generator options.</param>
    /// <param name="progress">Progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Loaded generator model.</returns>
    public static async Task<GgufGeneratorModel> LoadAsync(
        string modelId,
        string modelPath,
        IChatFormatter chatFormatter,
        GeneratorOptions options,
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

        // 2. Configure model parameters
        var contextLength = (uint)(options.MaxContextLength ?? 4096);
        var gpuLayers = CalculateGpuLayers(options.Provider);

        var modelParams = new ModelParams(modelPath)
        {
            ContextSize = contextLength,
            GpuLayerCount = gpuLayers,
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

        // 4. Create context
        progress?.Report(new DownloadProgress
        {
            FileName = Path.GetFileName(modelPath),
            BytesDownloaded = 80,
            TotalBytes = 100
        });

        var context = weights.CreateContext(modelParams);

        progress?.Report(new DownloadProgress
        {
            FileName = Path.GetFileName(modelPath),
            BytesDownloaded = 100,
            TotalBytes = 100
        });

        return new GgufGeneratorModel(
            modelId,
            modelPath,
            weights,
            context,
            chatFormatter,
            options,
            (int)contextLength);
    }

    /// <inheritdoc />
    public string ModelId { get; }

    /// <inheritdoc />
    public int MaxContextLength { get; }

    /// <inheritdoc />
    public IChatFormatter ChatFormatter => _chatFormatter;

    /// <inheritdoc />
    public bool IsGpuActive => CalculateGpuLayers(_options.Provider) != 0;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => IsGpuActive
        ? new[] { "LLamaSharpGPU", "CPU" }
        : new[] { "CPU" };

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _options.Provider;

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => File.Exists(_modelPath) ? new FileInfo(_modelPath).Length * 2 : null;

    /// <inheritdoc />
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        options ??= GenerationOptions.Default;

        // Acquire concurrency slot
        await _concurrencyLimiter.WaitAsync(cancellationToken);
        try
        {
            // Create sampling pipeline with parameters
            var samplingPipeline = new DefaultSamplingPipeline
            {
                Temperature = options.Temperature,
                TopP = options.TopP,
                TopK = options.TopK,
                RepeatPenalty = options.RepetitionPenalty
            };

            // Create inference parameters
            var inferenceParams = new InferenceParams
            {
                MaxTokens = options.MaxNewTokens ?? options.MaxTokens,
                SamplingPipeline = samplingPipeline,
                AntiPrompts = options.StopSequences?.ToList() ?? []
            };

            // Create executor for this generation
            var executor = new StatelessExecutor(_weights, _context.Params);

            // Track accumulated text for stop sequence detection
            var accumulatedText = new StringBuilder();

            // Initialize reasoning token filter if filtering is enabled
            var useReasoningFilter = options.FilterReasoningTokens || options.ExtractReasoningTokens;
            var reasoningFilter = useReasoningFilter
                ? new ReasoningTokenFilter(options.ExtractReasoningTokens)
                : null;

            await foreach (var token in executor.InferAsync(prompt, inferenceParams, cancellationToken))
            {
                accumulatedText.Append(token);

                // Check stop sequences
                if (ShouldStop(accumulatedText.ToString(), options.StopSequences))
                {
                    // Flush any remaining content from reasoning filter
                    if (reasoningFilter != null)
                    {
                        var remaining = reasoningFilter.Flush();
                        if (!string.IsNullOrEmpty(remaining))
                        {
                            yield return remaining;
                        }
                    }
                    yield break;
                }

                // Apply reasoning token filter if enabled
                if (reasoningFilter != null)
                {
                    var filtered = reasoningFilter.Process(token);
                    if (!string.IsNullOrEmpty(filtered))
                    {
                        yield return filtered;
                    }
                }
                else
                {
                    yield return token;
                }
            }

            // Flush remaining content when generation completes normally
            if (reasoningFilter != null)
            {
                var remaining = reasoningFilter.Flush();
                if (!string.IsNullOrEmpty(remaining))
                {
                    yield return remaining;
                }
            }
        }
        finally
        {
            _concurrencyLimiter.Release();
        }
    }

    /// <inheritdoc />
    public IAsyncEnumerable<string> GenerateChatAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var prompt = _chatFormatter.FormatPrompt(messages);

        // Merge stop sequences from formatter
        options ??= GenerationOptions.Default;
        var mergedOptions = MergeStopSequences(options);

        return GenerateAsync(prompt, mergedOptions, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<string> GenerateCompleteAsync(
        string prompt,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var sb = new StringBuilder();

        await foreach (var token in GenerateAsync(prompt, options, cancellationToken))
        {
            sb.Append(token);
        }

        return sb.ToString();
    }

    /// <inheritdoc />
    public async Task<string> GenerateChatCompleteAsync(
        IEnumerable<ChatMessage> messages,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var sb = new StringBuilder();

        await foreach (var token in GenerateChatAsync(messages, options, cancellationToken))
        {
            sb.Append(token);
        }

        return sb.ToString();
    }

    /// <inheritdoc />
    public Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        // Perform a minimal generation to warm up the model
        return GenerateCompleteAsync(
            "Hi",
            new GenerationOptions { MaxTokens = 5 },
            cancellationToken);
    }

    /// <inheritdoc />
    public GeneratorModelInfo GetModelInfo() => new(
        ModelId,
        _modelPath,
        MaxContextLength,
        _chatFormatter.FormatName,
        "LLamaSharp");

    private GenerationOptions MergeStopSequences(GenerationOptions options)
    {
        var merged = new List<string>();

        // 1. Stop sequences from chat formatter
        merged.AddRange(_chatFormatter.GetStopSequences());

        // 2. User-provided stop sequences
        var userStops = options.StopSequences ?? [];
        foreach (var stop in userStops)
        {
            if (!merged.Contains(stop, StringComparer.Ordinal))
            {
                merged.Add(stop);
            }
        }

        return new GenerationOptions
        {
            MaxTokens = options.MaxTokens,
            Temperature = options.Temperature,
            TopP = options.TopP,
            TopK = options.TopK,
            RepetitionPenalty = options.RepetitionPenalty,
            StopSequences = merged,
            IncludePromptInOutput = options.IncludePromptInOutput,
            DoSample = options.DoSample,
            NumBeams = options.NumBeams,
            PastPresentShareBuffer = options.PastPresentShareBuffer,
            MaxNewTokens = options.MaxNewTokens,
            FilterReasoningTokens = options.FilterReasoningTokens,
            ExtractReasoningTokens = options.ExtractReasoningTokens
        };
    }

    private static bool ShouldStop(string accumulatedText, IReadOnlyList<string>? stopSequences)
    {
        if (stopSequences == null || stopSequences.Count == 0)
        {
            return false;
        }

        foreach (var stop in stopSequences)
        {
            if (accumulatedText.EndsWith(stop, StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

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
        _concurrencyLimiter.Dispose();
        _context.Dispose();
        _weights.Dispose();

        return ValueTask.CompletedTask;
    }
}

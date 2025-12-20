using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.ImageGenerator.Pipeline;

/// <summary>
/// UNet model for LCM/Stable Diffusion latent denoising.
/// </summary>
internal sealed class UNetModel : IAsyncDisposable
{
    private readonly InferenceSession _session;
    private readonly string _sampleInput;
    private readonly string _timestepInput;
    private readonly string _encoderHiddenStatesInput;
    private readonly string _output;
    private bool _disposed;

    /// <summary>
    /// Latent channels (typically 4 for SD/LCM).
    /// </summary>
    public int LatentChannels { get; }

    private UNetModel(
        InferenceSession session,
        string sampleInput,
        string timestepInput,
        string encoderHiddenStatesInput,
        string output,
        int latentChannels)
    {
        _session = session;
        _sampleInput = sampleInput;
        _timestepInput = timestepInput;
        _encoderHiddenStatesInput = encoderHiddenStatesInput;
        _output = output;
        LatentChannels = latentChannels;
    }

    /// <summary>
    /// Loads the UNet model from the model directory.
    /// </summary>
    public static async Task<UNetModel> LoadAsync(
        string modelDir,
        SessionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var modelPath = FindUNetPath(modelDir);

        options ??= new SessionOptions();
        var session = await Task.Run(
            () => new InferenceSession(modelPath, options),
            cancellationToken);

        // Detect input/output names
        var inputs = session.InputMetadata;
        var outputs = session.OutputMetadata;

        // Common input patterns
        var sampleInput = FindInput(inputs, ["sample", "latent_model_input", "x"]);
        var timestepInput = FindInput(inputs, ["timestep", "t", "timesteps"]);
        var encoderInput = FindInput(inputs, ["encoder_hidden_states", "context", "text_embeds"]);
        var outputName = outputs.Keys.First();

        // Determine latent channels from sample input shape
        var sampleShape = inputs[sampleInput].Dimensions;
        var latentChannels = sampleShape.Length > 1 ? sampleShape[1] : 4;

        return new UNetModel(session, sampleInput, timestepInput, encoderInput, outputName, latentChannels);
    }

    /// <summary>
    /// Runs a single UNet forward pass.
    /// </summary>
    /// <param name="latents">Latent tensor [batch, channels, height, width].</param>
    /// <param name="timestep">Current timestep.</param>
    /// <param name="textEmbeddings">Text encoder output [batch, seqLen, hiddenSize].</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Predicted noise tensor.</returns>
    public async Task<DenseTensor<float>> ForwardAsync(
        DenseTensor<float> latents,
        long timestep,
        DenseTensor<float> textEmbeddings,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_sampleInput, latents),
            NamedOnnxValue.CreateFromTensor(_timestepInput, new DenseTensor<long>(new[] { timestep }, new[] { 1 })),
            NamedOnnxValue.CreateFromTensor(_encoderHiddenStatesInput, textEmbeddings)
        };

        var result = await Task.Run(() =>
        {
            using var outputs = _session.Run(inputs);
            var output = outputs.First();

            var outputTensor = output.AsTensor<float>();
            var dims = outputTensor.Dimensions.ToArray();
            var data = outputTensor.ToArray();

            return new DenseTensor<float>(data, dims);
        }, cancellationToken);

        return result;
    }

    private static string FindUNetPath(string modelDir)
    {
        var candidates = new[]
        {
            Path.Combine(modelDir, "unet", "model.onnx"),
            Path.Combine(modelDir, "unet.onnx"),
            Path.Combine(modelDir, "lcm_unet.onnx")
        };

        foreach (var path in candidates)
        {
            if (File.Exists(path))
                return path;
        }

        var files = Directory.GetFiles(modelDir, "*unet*.onnx", SearchOption.AllDirectories);
        if (files.Length > 0)
            return files[0];

        throw new FileNotFoundException($"Could not find UNet ONNX file in: {modelDir}");
    }

    private static string FindInput(IReadOnlyDictionary<string, NodeMetadata> inputs, string[] candidates)
    {
        foreach (var candidate in candidates)
        {
            var match = inputs.Keys.FirstOrDefault(k =>
                k.Equals(candidate, StringComparison.OrdinalIgnoreCase));
            if (match != null)
                return match;
        }

        // Fallback: try partial match
        foreach (var candidate in candidates)
        {
            var match = inputs.Keys.FirstOrDefault(k =>
                k.Contains(candidate, StringComparison.OrdinalIgnoreCase));
            if (match != null)
                return match;
        }

        throw new InvalidOperationException($"Could not find input matching: {string.Join(", ", candidates)}");
    }

    public ValueTask DisposeAsync()
    {
        if (_disposed) return ValueTask.CompletedTask;
        _disposed = true;

        _session.Dispose();
        return ValueTask.CompletedTask;
    }
}

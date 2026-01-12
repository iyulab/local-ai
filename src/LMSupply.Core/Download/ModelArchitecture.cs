namespace LMSupply.Core.Download;

/// <summary>
/// Represents the architecture pattern of an ONNX model.
/// </summary>
public enum ModelArchitecture
{
    /// <summary>Unknown or undetected architecture.</summary>
    Unknown,

    /// <summary>Single model file (e.g., Embedder, Reranker).</summary>
    SingleModel,

    /// <summary>Encoder-decoder architecture (e.g., Translator, Captioner, Transcriber).</summary>
    EncoderDecoder,

    /// <summary>Diffusion pipeline with multiple specialized components.</summary>
    DiffusionPipeline
}

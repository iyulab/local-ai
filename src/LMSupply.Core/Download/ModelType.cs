namespace LMSupply.Download;

/// <summary>
/// Type of AI model based on its primary function.
/// </summary>
public enum ModelType
{
    /// <summary>Unknown or unrecognized model type.</summary>
    Unknown,

    /// <summary>Text generation models (LLMs like Phi, Llama, Mistral).</summary>
    Generator,

    /// <summary>Text embedding models (BGE, E5, MiniLM).</summary>
    Embedder,

    /// <summary>Semantic reranking models (cross-encoders).</summary>
    Reranker,

    /// <summary>Speech-to-text models (Whisper).</summary>
    Transcriber,

    /// <summary>Text-to-speech models (Piper, VITS).</summary>
    Synthesizer,

    /// <summary>Object detection models.</summary>
    Detector,

    /// <summary>Image captioning models.</summary>
    Captioner,

    /// <summary>Translation models.</summary>
    Translator,

    /// <summary>Optical character recognition models.</summary>
    Ocr,

    /// <summary>Image segmentation models.</summary>
    Segmenter
}

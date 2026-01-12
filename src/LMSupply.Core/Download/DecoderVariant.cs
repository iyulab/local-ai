namespace LMSupply.Core.Download;

/// <summary>
/// Represents different decoder model variants for encoder-decoder architectures.
/// </summary>
public enum DecoderVariant
{
    /// <summary>Standard decoder (decoder_model.onnx).</summary>
    Standard,

    /// <summary>Merged decoder with past key-values (decoder_model_merged.onnx).</summary>
    Merged,

    /// <summary>Decoder with past key-values support (decoder_with_past_model.onnx).</summary>
    WithPast,

    /// <summary>Decoder-only architecture.</summary>
    DecoderOnly
}

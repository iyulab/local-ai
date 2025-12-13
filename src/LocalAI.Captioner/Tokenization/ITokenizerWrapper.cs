namespace LocalAI.Captioner.Tokenization;

/// <summary>
/// Wrapper interface for tokenizer implementations.
/// Provides a unified interface for different tokenizer types.
/// </summary>
public interface ITokenizerWrapper
{
    /// <summary>
    /// Encodes text into token IDs.
    /// </summary>
    /// <param name="text">Text to encode.</param>
    /// <returns>Array of token IDs.</returns>
    int[] Encode(string text);

    /// <summary>
    /// Decodes token IDs back to text.
    /// </summary>
    /// <param name="tokenIds">Token IDs to decode.</param>
    /// <returns>Decoded text.</returns>
    string Decode(int[] tokenIds);

    /// <summary>
    /// Decodes token IDs back to text, skipping special tokens.
    /// </summary>
    /// <param name="tokenIds">Token IDs to decode.</param>
    /// <returns>Decoded text without special tokens.</returns>
    string DecodeSkipSpecialTokens(int[] tokenIds);

    /// <summary>
    /// Gets the vocabulary size.
    /// </summary>
    int VocabSize { get; }

    /// <summary>
    /// Gets the BOS (beginning of sequence) token ID, if available.
    /// </summary>
    int? BosTokenId { get; }

    /// <summary>
    /// Gets the EOS (end of sequence) token ID, if available.
    /// </summary>
    int? EosTokenId { get; }

    /// <summary>
    /// Gets the PAD (padding) token ID, if available.
    /// </summary>
    int? PadTokenId { get; }
}

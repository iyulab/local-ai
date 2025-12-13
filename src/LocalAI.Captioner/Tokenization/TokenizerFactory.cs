using LocalAI.Captioner.Models;
using LocalAI.Exceptions;

namespace LocalAI.Captioner.Tokenization;

/// <summary>
/// Factory for creating tokenizer instances based on model type.
/// </summary>
public static class TokenizerFactory
{
    /// <summary>
    /// Creates a tokenizer for the specified model directory and type.
    /// </summary>
    /// <param name="modelDir">Directory containing tokenizer files.</param>
    /// <param name="tokenizerType">Type of tokenizer to create.</param>
    /// <returns>Tokenizer wrapper instance.</returns>
    public static async Task<ITokenizerWrapper> CreateAsync(string modelDir, TokenizerType tokenizerType)
    {
        return tokenizerType switch
        {
            TokenizerType.Gpt2 => await CreateGpt2TokenizerAsync(modelDir).ConfigureAwait(false),
            TokenizerType.Bert => throw new NotSupportedException("BERT tokenizer is not yet supported."),
            TokenizerType.SentencePiece => throw new NotSupportedException("SentencePiece tokenizer is not yet supported."),
            TokenizerType.Llama => throw new NotSupportedException("Llama tokenizer is not yet supported."),
            _ => throw new ArgumentOutOfRangeException(nameof(tokenizerType), tokenizerType, "Unknown tokenizer type")
        };
    }

    private static async Task<ITokenizerWrapper> CreateGpt2TokenizerAsync(string modelDir)
    {
        // Look for GPT-2 tokenizer files
        var vocabPath = Path.Combine(modelDir, "vocab.json");
        var mergesPath = Path.Combine(modelDir, "merges.txt");

        if (!File.Exists(vocabPath))
        {
            throw new ModelNotFoundException(
                $"GPT-2 vocabulary file not found: {vocabPath}",
                modelDir);
        }

        if (!File.Exists(mergesPath))
        {
            throw new ModelNotFoundException(
                $"GPT-2 merges file not found: {mergesPath}",
                modelDir);
        }

        return await Gpt2TokenizerWrapper.CreateAsync(vocabPath, mergesPath).ConfigureAwait(false);
    }
}

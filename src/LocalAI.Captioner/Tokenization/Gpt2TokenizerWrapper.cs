using System.Text.Json;
using Microsoft.ML.Tokenizers;

namespace LocalAI.Captioner.Tokenization;

/// <summary>
/// GPT-2 BPE tokenizer wrapper using Microsoft.ML.Tokenizers.
/// </summary>
internal sealed class Gpt2TokenizerWrapper : ITokenizerWrapper
{
    private readonly CodeGenTokenizer _tokenizer;
    private readonly Dictionary<int, string> _idToToken;
    private readonly int _vocabSize;
    private readonly int? _bosTokenId;
    private readonly int? _eosTokenId;
    private readonly int? _padTokenId;

    private Gpt2TokenizerWrapper(
        CodeGenTokenizer tokenizer,
        Dictionary<int, string> idToToken,
        int vocabSize,
        int? bosTokenId,
        int? eosTokenId,
        int? padTokenId)
    {
        _tokenizer = tokenizer;
        _idToToken = idToToken;
        _vocabSize = vocabSize;
        _bosTokenId = bosTokenId;
        _eosTokenId = eosTokenId;
        _padTokenId = padTokenId;
    }

    /// <inheritdoc />
    public int VocabSize => _vocabSize;

    /// <inheritdoc />
    public int? BosTokenId => _bosTokenId;

    /// <inheritdoc />
    public int? EosTokenId => _eosTokenId;

    /// <inheritdoc />
    public int? PadTokenId => _padTokenId;

    /// <summary>
    /// Creates a GPT-2 tokenizer from vocab and merges files.
    /// </summary>
    public static async Task<Gpt2TokenizerWrapper> CreateAsync(string vocabPath, string mergesPath)
    {
        // Load vocab.json to build id-to-token mapping
        var vocabJson = await File.ReadAllTextAsync(vocabPath).ConfigureAwait(false);
        var vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson)
            ?? throw new InvalidOperationException("Failed to parse vocab.json");

        var idToToken = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);
        var vocabSize = vocab.Count;

        // Create tokenizer using Microsoft.ML.Tokenizers
        using var vocabStream = File.OpenRead(vocabPath);
        using var mergesStream = File.OpenRead(mergesPath);

        var tokenizer = await Task.Run(() =>
        {
            return CodeGenTokenizer.Create(vocabStream, mergesStream);
        }).ConfigureAwait(false);

        // GPT-2 uses the same token for BOS, EOS, and PAD (endoftext token)
        int? eosToken = vocab.GetValueOrDefault("<|endoftext|>", -1);
        if (eosToken == -1) eosToken = null;

        return new Gpt2TokenizerWrapper(
            tokenizer,
            idToToken,
            vocabSize,
            bosTokenId: eosToken,  // GPT-2 uses endoftext as BOS
            eosTokenId: eosToken,
            padTokenId: eosToken);
    }

    /// <inheritdoc />
    public int[] Encode(string text)
    {
        var ids = _tokenizer.EncodeToIds(text);
        return [.. ids];
    }

    /// <inheritdoc />
    public string Decode(int[] tokenIds)
    {
        // Decode using the tokenizer
        var decoded = _tokenizer.Decode(tokenIds);

        // GPT-2 uses special encoding for spaces (Ġ represents space before word)
        // The tokenizer should handle this, but we might need post-processing
        return decoded ?? string.Empty;
    }

    /// <inheritdoc />
    public string DecodeSkipSpecialTokens(int[] tokenIds)
    {
        // Filter out special tokens
        var filtered = tokenIds
            .Where(id => id != _bosTokenId && id != _eosTokenId && id != _padTokenId)
            .ToArray();

        return Decode(filtered);
    }
}

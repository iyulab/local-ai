using Microsoft.ML.Tokenizers;

namespace LocalAI.Translator.Tokenization;

/// <summary>
/// SentencePiece-based tokenizer for machine translation models.
/// Supports OPUS-MT (MarianMT) models with source and target tokenizers.
/// </summary>
internal sealed class TranslatorTokenizer : IDisposable
{
    private readonly Tokenizer _sourceTokenizer;
    private readonly Tokenizer _targetTokenizer;
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<int, string> _reverseVocab;
    private readonly int _padTokenId;
    private readonly int _bosTokenId;
    private readonly int _eosTokenId;
    private readonly int _unkTokenId;
    private bool _disposed;

    /// <summary>
    /// Gets the vocabulary size.
    /// </summary>
    public int VocabSize => _vocab.Count;

    /// <summary>
    /// Gets the PAD token ID.
    /// </summary>
    public int PadTokenId => _padTokenId;

    /// <summary>
    /// Gets the BOS (beginning of sequence) token ID.
    /// </summary>
    public int BosTokenId => _bosTokenId;

    /// <summary>
    /// Gets the EOS (end of sequence) token ID.
    /// </summary>
    public int EosTokenId => _eosTokenId;

    /// <summary>
    /// Gets the UNK (unknown) token ID.
    /// </summary>
    public int UnkTokenId => _unkTokenId;

    private TranslatorTokenizer(
        Tokenizer sourceTokenizer,
        Tokenizer targetTokenizer,
        Dictionary<string, int> vocab,
        int padTokenId,
        int bosTokenId,
        int eosTokenId,
        int unkTokenId)
    {
        _sourceTokenizer = sourceTokenizer;
        _targetTokenizer = targetTokenizer;
        _vocab = vocab;
        _reverseVocab = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);
        _padTokenId = padTokenId;
        _bosTokenId = bosTokenId;
        _eosTokenId = eosTokenId;
        _unkTokenId = unkTokenId;
    }

    /// <summary>
    /// Creates a tokenizer from model directory.
    /// </summary>
    /// <param name="modelDir">Path to the model directory.</param>
    /// <returns>A configured tokenizer instance.</returns>
    public static TranslatorTokenizer Create(string modelDir)
    {
        // Load vocabulary
        var vocab = LoadVocabulary(modelDir);

        // Determine special token IDs
        var padTokenId = GetTokenId(vocab, "<pad>", "▁<pad>", 0);
        var bosTokenId = GetTokenId(vocab, "<s>", "▁<s>", padTokenId);
        var eosTokenId = GetTokenId(vocab, "</s>", "▁</s>", 0);
        var unkTokenId = GetTokenId(vocab, "<unk>", "▁<unk>", 3);

        // Load SentencePiece tokenizers
        var sourceTokenizer = LoadSentencePieceTokenizer(modelDir, "source.spm", "sentencepiece.bpe.model");
        var targetTokenizer = LoadSentencePieceTokenizer(modelDir, "target.spm", "sentencepiece.bpe.model");

        return new TranslatorTokenizer(
            sourceTokenizer,
            targetTokenizer,
            vocab,
            padTokenId,
            bosTokenId,
            eosTokenId,
            unkTokenId);
    }

    /// <summary>
    /// Encodes source text to token IDs.
    /// </summary>
    /// <param name="text">Input text in source language.</param>
    /// <param name="addSpecialTokens">Whether to add BOS/EOS tokens.</param>
    /// <returns>Array of token IDs.</returns>
    public long[] EncodeSource(string text, bool addSpecialTokens = true)
    {
        var tokens = new List<long>();

        if (addSpecialTokens && _bosTokenId != _padTokenId)
        {
            tokens.Add(_bosTokenId);
        }

        var encodedIds = _sourceTokenizer.EncodeToIds(text);
        foreach (var id in encodedIds)
        {
            tokens.Add(id);
        }

        if (addSpecialTokens)
        {
            tokens.Add(_eosTokenId);
        }

        return tokens.ToArray();
    }

    /// <summary>
    /// Decodes token IDs to target text.
    /// </summary>
    /// <param name="tokenIds">Array of token IDs.</param>
    /// <param name="skipSpecialTokens">Whether to skip special tokens in output.</param>
    /// <returns>Decoded text string.</returns>
    public string DecodeTarget(IEnumerable<long> tokenIds, bool skipSpecialTokens = true)
    {
        var ids = tokenIds.Select(id => (int)id);

        if (skipSpecialTokens)
        {
            ids = ids.Where(id =>
                id != _padTokenId &&
                id != _bosTokenId &&
                id != _eosTokenId);
        }

        var result = _targetTokenizer.Decode(ids);
        return result ?? string.Empty;
    }

    /// <summary>
    /// Gets decoder input IDs for autoregressive generation.
    /// </summary>
    /// <returns>Initial decoder input IDs.</returns>
    public long[] GetDecoderStartIds()
    {
        return [_padTokenId]; // MarianMT uses pad_token_id as decoder_start_token_id
    }

    /// <summary>
    /// Checks if a token ID is a special token.
    /// </summary>
    /// <param name="tokenId">Token ID to check.</param>
    /// <returns>True if the token is a special token.</returns>
    public bool IsSpecialToken(long tokenId)
    {
        return tokenId == _padTokenId ||
               tokenId == _bosTokenId ||
               tokenId == _eosTokenId ||
               tokenId == _unkTokenId;
    }

    private static Dictionary<string, int> LoadVocabulary(string modelDir)
    {
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);

        // Try vocab.json first (standard format)
        var vocabJsonPath = Path.Combine(modelDir, "vocab.json");
        if (File.Exists(vocabJsonPath))
        {
            var json = File.ReadAllText(vocabJsonPath);
            vocab = ParseVocabJson(json);
        }

        // Try tokenizer.json (HuggingFace format)
        if (vocab.Count == 0)
        {
            var tokenizerJsonPath = Path.Combine(modelDir, "tokenizer.json");
            if (File.Exists(tokenizerJsonPath))
            {
                var json = File.ReadAllText(tokenizerJsonPath);
                vocab = ParseTokenizerJson(json);
            }
        }

        // Create minimal vocabulary if none found
        if (vocab.Count == 0)
        {
            vocab = CreateMinimalVocabulary();
        }

        return vocab;
    }

    private static Dictionary<string, int> ParseVocabJson(string json)
    {
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);

        // Simple JSON parsing for vocab format: {"token": id, ...}
        var cleanJson = json.Trim();
        if (!cleanJson.StartsWith("{") || !cleanJson.EndsWith("}"))
            return vocab;

        cleanJson = cleanJson[1..^1]; // Remove { and }

        foreach (var entry in SplitJsonEntries(cleanJson))
        {
            var parts = entry.Split(':', 2);
            if (parts.Length != 2) continue;

            var token = parts[0].Trim().Trim('"');
            if (int.TryParse(parts[1].Trim(), out var id))
            {
                vocab[token] = id;
            }
        }

        return vocab;
    }

    private static Dictionary<string, int> ParseTokenizerJson(string json)
    {
        // Simplified parsing for tokenizer.json - look for "vocab" section
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);

        var vocabIndex = json.IndexOf("\"vocab\"", StringComparison.Ordinal);
        if (vocabIndex < 0)
            return vocab;

        var startBrace = json.IndexOf('{', vocabIndex);
        if (startBrace < 0)
            return vocab;

        var braceCount = 1;
        var endBrace = startBrace + 1;

        while (braceCount > 0 && endBrace < json.Length)
        {
            if (json[endBrace] == '{') braceCount++;
            else if (json[endBrace] == '}') braceCount--;
            endBrace++;
        }

        var vocabJson = json.Substring(startBrace, endBrace - startBrace);
        return ParseVocabJson(vocabJson);
    }

    private static IEnumerable<string> SplitJsonEntries(string json)
    {
        var entries = new List<string>();
        var current = new System.Text.StringBuilder();
        var inString = false;
        var escape = false;

        foreach (var c in json)
        {
            if (escape)
            {
                current.Append(c);
                escape = false;
                continue;
            }

            if (c == '\\')
            {
                current.Append(c);
                escape = true;
                continue;
            }

            if (c == '"')
            {
                inString = !inString;
                current.Append(c);
                continue;
            }

            if (c == ',' && !inString)
            {
                if (current.Length > 0)
                {
                    entries.Add(current.ToString().Trim());
                    current.Clear();
                }
                continue;
            }

            current.Append(c);
        }

        if (current.Length > 0)
        {
            entries.Add(current.ToString().Trim());
        }

        return entries;
    }

    private static Dictionary<string, int> CreateMinimalVocabulary()
    {
        // Create a minimal vocabulary for fallback
        return new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["<pad>"] = 0,
            ["<s>"] = 1,
            ["</s>"] = 2,
            ["<unk>"] = 3
        };
    }

    private static int GetTokenId(Dictionary<string, int> vocab, string primary, string alternative, int defaultId)
    {
        if (vocab.TryGetValue(primary, out var id))
            return id;
        if (vocab.TryGetValue(alternative, out id))
            return id;
        return defaultId;
    }

    private static Tokenizer LoadSentencePieceTokenizer(string modelDir, string primaryFile, string fallbackFile)
    {
        var primaryPath = Path.Combine(modelDir, primaryFile);
        var fallbackPath = Path.Combine(modelDir, fallbackFile);

        string? tokenizerPath = null;

        if (File.Exists(primaryPath))
            tokenizerPath = primaryPath;
        else if (File.Exists(fallbackPath))
            tokenizerPath = fallbackPath;

        if (tokenizerPath != null)
        {
            try
            {
                using var stream = File.OpenRead(tokenizerPath);
                return LlamaTokenizer.Create(stream);
            }
            catch
            {
                // Fall through to BPE tokenizer
            }
        }

        // Try loading from tokenizer.json (HuggingFace format) using LlamaTokenizer
        var tokenizerJsonPath = Path.Combine(modelDir, "tokenizer.json");
        if (File.Exists(tokenizerJsonPath))
        {
            try
            {
                using var stream = File.OpenRead(tokenizerJsonPath);
                return LlamaTokenizer.Create(stream);
            }
            catch
            {
                // Fall through to fallback
            }
        }

        // Create a fallback tokenizer using vocab.json if available
        return CreateFallbackTokenizer(modelDir);
    }

    private static Tokenizer CreateFallbackTokenizer(string modelDir)
    {
        // Try to create BPE tokenizer from vocab.json and merges.txt
        var vocabPath = Path.Combine(modelDir, "vocab.json");
        var mergesPath = Path.Combine(modelDir, "merges.txt");

        if (File.Exists(vocabPath) && File.Exists(mergesPath))
        {
            try
            {
                using var vocabStream = File.OpenRead(vocabPath);
                using var mergesStream = File.OpenRead(mergesPath);
                return CodeGenTokenizer.Create(vocabStream, mergesStream);
            }
            catch
            {
                // Fall through
            }
        }

        // Return a simple whitespace-based tokenizer as last resort
        // This creates a minimal tokenizer that at least won't crash
        throw new InvalidOperationException(
            $"Could not load tokenizer from model directory: {modelDir}. " +
            "Expected one of: source.spm, sentencepiece.bpe.model, tokenizer.json, or vocab.json + merges.txt");
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        // Tokenizers don't implement IDisposable, but we keep the pattern for future compatibility
        _disposed = true;
    }
}

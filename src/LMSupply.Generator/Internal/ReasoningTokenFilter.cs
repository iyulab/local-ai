using System.Text;
using System.Text.RegularExpressions;

namespace LMSupply.Generator.Internal;

/// <summary>
/// Filters reasoning tokens (e.g., &lt;think&gt;...&lt;/think&gt;) from streaming text generation.
/// Supports DeepSeek R1 and other reasoning models.
/// </summary>
internal sealed partial class ReasoningTokenFilter
{
    private readonly StringBuilder _buffer = new();
    private readonly StringBuilder _reasoningContent = new();
    private bool _inReasoningBlock;
    private bool _extractReasoning;

    // Reasoning tag patterns for various models
    private static readonly string[] OpenTags = ["<think>", "<｜begin▁of▁thinking｜>"];
    private static readonly string[] CloseTags = ["</think>", "<｜end▁of▁thinking｜>"];

    /// <summary>
    /// Gets the accumulated reasoning content (if extraction is enabled).
    /// </summary>
    public string ReasoningContent => _reasoningContent.ToString();

    /// <summary>
    /// Initializes a new instance of the ReasoningTokenFilter.
    /// </summary>
    /// <param name="extractReasoning">Whether to capture reasoning content separately.</param>
    public ReasoningTokenFilter(bool extractReasoning = false)
    {
        _extractReasoning = extractReasoning;
    }

    /// <summary>
    /// Processes a token and returns filtered output.
    /// </summary>
    /// <param name="token">The token to process.</param>
    /// <returns>The filtered output (empty if inside reasoning block).</returns>
    public string Process(string token)
    {
        _buffer.Append(token);
        var text = _buffer.ToString();

        var output = new StringBuilder();

        while (text.Length > 0)
        {
            if (_inReasoningBlock)
            {
                // Look for closing tag
                var (closeTag, closeIdx) = FindFirstTag(text, CloseTags);

                if (closeIdx >= 0)
                {
                    // Found closing tag
                    if (_extractReasoning)
                    {
                        _reasoningContent.Append(text[..closeIdx]);
                    }
                    text = text[(closeIdx + closeTag!.Length)..];
                    _inReasoningBlock = false;
                }
                else
                {
                    // Still in reasoning block, check for partial close tag at end
                    var partialMatch = GetPartialTagMatch(text, CloseTags);
                    if (partialMatch > 0)
                    {
                        // Keep partial match in buffer
                        if (_extractReasoning)
                        {
                            _reasoningContent.Append(text[..^partialMatch]);
                        }
                        _buffer.Clear();
                        _buffer.Append(text[^partialMatch..]);
                        return output.ToString();
                    }

                    // No close tag found, consume all as reasoning
                    if (_extractReasoning)
                    {
                        _reasoningContent.Append(text);
                    }
                    _buffer.Clear();
                    return output.ToString();
                }
            }
            else
            {
                // Look for opening tag
                var (openTag, openIdx) = FindFirstTag(text, OpenTags);

                if (openIdx >= 0)
                {
                    // Found opening tag
                    output.Append(text[..openIdx]);
                    text = text[(openIdx + openTag!.Length)..];
                    _inReasoningBlock = true;
                }
                else
                {
                    // No open tag found, check for partial open tag at end
                    var partialMatch = GetPartialTagMatch(text, OpenTags);
                    if (partialMatch > 0)
                    {
                        // Keep partial match in buffer
                        output.Append(text[..^partialMatch]);
                        _buffer.Clear();
                        _buffer.Append(text[^partialMatch..]);
                        return output.ToString();
                    }

                    // No tags found, output everything
                    output.Append(text);
                    _buffer.Clear();
                    return output.ToString();
                }
            }
        }

        _buffer.Clear();
        return output.ToString();
    }

    /// <summary>
    /// Flushes any remaining buffered content.
    /// </summary>
    /// <returns>Any remaining filtered content.</returns>
    public string Flush()
    {
        var remaining = _buffer.ToString();
        _buffer.Clear();

        if (_inReasoningBlock)
        {
            // If still in reasoning block, content was never closed
            if (_extractReasoning)
            {
                _reasoningContent.Append(remaining);
            }
            return string.Empty;
        }

        return remaining;
    }

    /// <summary>
    /// Resets the filter state for reuse.
    /// </summary>
    public void Reset()
    {
        _buffer.Clear();
        _reasoningContent.Clear();
        _inReasoningBlock = false;
    }

    private static (string? tag, int index) FindFirstTag(string text, string[] tags)
    {
        string? foundTag = null;
        int foundIdx = -1;

        foreach (var tag in tags)
        {
            var idx = text.IndexOf(tag, StringComparison.Ordinal);
            if (idx >= 0 && (foundIdx < 0 || idx < foundIdx))
            {
                foundIdx = idx;
                foundTag = tag;
            }
        }

        return (foundTag, foundIdx);
    }

    private static int GetPartialTagMatch(string text, string[] tags)
    {
        int maxPartial = 0;

        foreach (var tag in tags)
        {
            for (int len = Math.Min(tag.Length - 1, text.Length); len > 0; len--)
            {
                if (text.EndsWith(tag[..len], StringComparison.Ordinal))
                {
                    maxPartial = Math.Max(maxPartial, len);
                    break;
                }
            }
        }

        return maxPartial;
    }
}

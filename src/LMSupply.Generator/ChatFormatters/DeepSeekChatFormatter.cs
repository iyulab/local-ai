using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// Chat formatter for DeepSeek models.
/// Similar to ChatML but with specific tokens for reasoning models.
/// </summary>
public sealed class DeepSeekChatFormatter : IChatFormatter
{
    /// <inheritdoc />
    public string FormatName => "deepseek";

    /// <inheritdoc />
    public string FormatPrompt(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();

        foreach (var message in messages)
        {
            var role = message.Role switch
            {
                ChatRole.System => "system",
                ChatRole.User => "user",
                ChatRole.Assistant => "assistant",
                _ => throw new ArgumentOutOfRangeException(nameof(message.Role))
            };

            sb.Append("<|");
            sb.Append(role);
            sb.Append("|>\n");
            sb.Append(message.Content);
            sb.Append('\n');
        }

        // Add assistant prompt to start generation
        sb.Append("<|assistant|>\n");

        return sb.ToString();
    }

    /// <inheritdoc />
    public string GetStopToken() => "<|end|>";

    /// <inheritdoc />
    public IReadOnlyList<string> GetStopSequences() =>
        ["<|end|>", "<|user|>", "<|endoftext|>"];
}

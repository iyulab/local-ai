using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// Chat formatter for EXAONE models (Korean-specialized).
/// Format: [|system|]{content}[|endofturn|][|user|]{content}[|endofturn|][|assistant|]
/// </summary>
public sealed class ExaoneChatFormatter : IChatFormatter
{
    private const string SystemTag = "[|system|]";
    private const string UserTag = "[|user|]";
    private const string AssistantTag = "[|assistant|]";
    private const string EndOfTurn = "[|endofturn|]";

    /// <inheritdoc />
    public string FormatName => "exaone";

    /// <inheritdoc />
    public string FormatPrompt(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();

        foreach (var message in messages)
        {
            var tag = message.Role switch
            {
                ChatRole.System => SystemTag,
                ChatRole.User => UserTag,
                ChatRole.Assistant => AssistantTag,
                _ => throw new ArgumentOutOfRangeException(nameof(message.Role))
            };

            sb.Append(tag);
            sb.Append(message.Content);
            sb.Append(EndOfTurn);
        }

        // Add assistant tag to start generation
        sb.Append(AssistantTag);

        return sb.ToString();
    }

    /// <inheritdoc />
    public string GetStopToken() => EndOfTurn;

    /// <inheritdoc />
    public IReadOnlyList<string> GetStopSequences() =>
        [EndOfTurn, UserTag, SystemTag];
}

using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// Chat formatter for Mistral/Mixtral models.
/// Format: [INST] {user_message} [/INST] {assistant_message}
/// </summary>
public sealed class MistralChatFormatter : IChatFormatter
{
    private const string InstStart = "[INST]";
    private const string InstEnd = "[/INST]";
    private const string BosToken = "<s>";
    private const string EosToken = "</s>";

    /// <inheritdoc />
    public string FormatName => "mistral";

    /// <inheritdoc />
    public string FormatPrompt(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();
        var messagesList = messages.ToList();
        string? systemMessage = null;

        // Extract system message if present
        var systemMsgIndex = messagesList.FindIndex(m => m.Role == ChatRole.System);
        if (systemMsgIndex >= 0)
        {
            systemMessage = messagesList[systemMsgIndex].Content;
            messagesList.RemoveAt(systemMsgIndex);
        }

        sb.Append(BosToken);

        for (var i = 0; i < messagesList.Count; i++)
        {
            var message = messagesList[i];

            if (message.Role == ChatRole.User)
            {
                sb.Append(InstStart);
                sb.Append(' ');

                // Include system message with first user message
                if (systemMessage != null && i == 0)
                {
                    sb.Append(systemMessage);
                    sb.Append("\n\n");
                }

                sb.Append(message.Content);
                sb.Append(' ');
                sb.Append(InstEnd);
            }
            else if (message.Role == ChatRole.Assistant)
            {
                sb.Append(' ');
                sb.Append(message.Content);
                sb.Append(EosToken);
            }
        }

        return sb.ToString();
    }

    /// <inheritdoc />
    public string GetStopToken() => EosToken;

    /// <inheritdoc />
    public IReadOnlyList<string> GetStopSequences() =>
        [EosToken, InstStart];
}

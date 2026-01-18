using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.ChatFormatters;

/// <summary>
/// Chat formatter for Gemma 2 models.
/// Format: &lt;start_of_turn&gt;user\n{content}&lt;end_of_turn&gt;\n&lt;start_of_turn&gt;model\n
/// </summary>
public sealed class GemmaChatFormatter : IChatFormatter
{
    private const string StartOfTurn = "<start_of_turn>";
    private const string EndOfTurn = "<end_of_turn>";

    /// <inheritdoc />
    public string FormatName => "gemma";

    /// <inheritdoc />
    public string FormatPrompt(IEnumerable<ChatMessage> messages)
    {
        var sb = new StringBuilder();

        foreach (var message in messages)
        {
            var role = message.Role switch
            {
                ChatRole.System => "user", // Gemma treats system as user
                ChatRole.User => "user",
                ChatRole.Assistant => "model",
                _ => throw new ArgumentOutOfRangeException(nameof(message.Role))
            };

            sb.Append(StartOfTurn);
            sb.Append(role);
            sb.Append('\n');
            sb.Append(message.Content);
            sb.Append(EndOfTurn);
            sb.Append('\n');
        }

        // Add model prompt to start generation
        sb.Append(StartOfTurn);
        sb.Append("model");
        sb.Append('\n');

        return sb.ToString();
    }

    /// <inheritdoc />
    public string GetStopToken() => EndOfTurn;

    /// <inheritdoc />
    public IReadOnlyList<string> GetStopSequences() =>
        [EndOfTurn, StartOfTurn];
}

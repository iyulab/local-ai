namespace LMSupply.Generator.Models;

/// <summary>
/// Result from generation with reasoning content extraction.
/// </summary>
/// <param name="Response">The main response text (with reasoning filtered out).</param>
/// <param name="Reasoning">The extracted reasoning content (from &lt;think&gt; tags).</param>
public readonly record struct ReasoningResult(string Response, string Reasoning);

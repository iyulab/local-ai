using LMSupply;
using LMSupply.Generator;
using LMSupply.Generator.Models;

Console.WriteLine("=== LMSupply GGUF Generator Sample ===\n");

// Example 1: Simple text generation
Console.WriteLine("1. Simple Text Generation");
Console.WriteLine("-".PadRight(40, '-'));

var progress = new Progress<DownloadProgress>(p =>
{
    if (p.TotalBytes > 0)
    {
        var percent = (double)p.BytesDownloaded / p.TotalBytes;
        Console.Write($"\rDownloading {p.FileName}: {percent:P0}");
    }
});

await using var model = await LocalGenerator.LoadAsync(
    "gguf:fast",  // Llama 3.2 1B - smallest and fastest
    progress: progress
);

Console.WriteLine();

// Get model info
var info = model.GetModelInfo();
Console.WriteLine($"Model: {info.ModelId}");
Console.WriteLine($"Context: {info.MaxContextLength}");
Console.WriteLine($"Format: {info.ChatFormat}");
Console.WriteLine($"Provider: {info.ExecutionProvider}");
Console.WriteLine();

// Generate text
Console.Write("Prompt: Hello, my name is\nResponse: ");
await foreach (var token in model.GenerateAsync(
    "Hello, my name is",
    new GenerationOptions { MaxTokens = 50 }))
{
    Console.Write(token);
}
Console.WriteLine("\n");

// Example 2: Chat completion
Console.WriteLine("2. Chat Completion");
Console.WriteLine("-".PadRight(40, '-'));

var messages = new[]
{
    ChatMessage.System("You are a helpful assistant. Keep responses brief."),
    ChatMessage.User("What is the capital of France?")
};

Console.WriteLine($"System: {messages[0].Content}");
Console.WriteLine($"User: {messages[1].Content}");
Console.Write("Assistant: ");

await foreach (var token in model.GenerateChatAsync(
    messages,
    new GenerationOptions { MaxTokens = 100 }))
{
    Console.Write(token);
}
Console.WriteLine("\n");

// Example 3: Multi-turn conversation
Console.WriteLine("3. Multi-turn Conversation");
Console.WriteLine("-".PadRight(40, '-'));

var conversation = new List<ChatMessage>
{
    ChatMessage.System("You are a math tutor. Be concise."),
    ChatMessage.User("What is 15 * 7?")
};

Console.WriteLine($"User: {conversation[1].Content}");
Console.Write("Assistant: ");

var response = "";
await foreach (var token in model.GenerateChatAsync(
    conversation,
    new GenerationOptions { MaxTokens = 50 }))
{
    Console.Write(token);
    response += token;
}
Console.WriteLine();

// Add assistant response and continue
conversation.Add(ChatMessage.Assistant(response.Trim()));
conversation.Add(ChatMessage.User("Now divide that by 3"));

Console.WriteLine($"User: Now divide that by 3");
Console.Write("Assistant: ");

await foreach (var token in model.GenerateChatAsync(
    conversation,
    new GenerationOptions { MaxTokens = 50 }))
{
    Console.Write(token);
}
Console.WriteLine("\n");

Console.WriteLine("=== Sample Complete ===");

using System.Text.Json;
using LMSupply;
using LMSupply.Generator;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;
using LMSupply.Llama;

Console.WriteLine("=== LMSupply GGUF Generator Sample ===\n");

// Progress reporter
var progress = new Progress<DownloadProgress>(p =>
{
    if (p.TotalBytes > 0)
    {
        var percent = (double)p.BytesDownloaded / p.TotalBytes;
        Console.Write($"\rDownloading {p.FileName}: {percent:P0}");
    }
});

// Load model
await using var model = await LocalGenerator.LoadAsync(
    "gguf:fast",  // Llama 3.2 1B - smallest and fastest
    progress: progress
);
Console.WriteLine();

var info = model.GetModelInfo();
Console.WriteLine($"Model: {info.ModelId}");
Console.WriteLine($"Context: {info.MaxContextLength}");
Console.WriteLine($"Format: {info.ChatFormat}");

// Display engine/backend information
var runtimeManager = LlamaRuntimeManager.Instance;
Console.WriteLine();
Console.WriteLine("=== Engine Information ===");
Console.WriteLine(runtimeManager.GetEnvironmentSummary());
Console.WriteLine();

// =============================================================================
// 1. Basic Text Generation
// =============================================================================
await RunExample("1. Basic Text Generation", async () =>
{
    Console.Write("Prompt: Hello, my name is\nResponse: ");
    await foreach (var token in model.GenerateAsync(
        "Hello, my name is",
        new GenerationOptions { MaxTokens = 50 }))
    {
        Console.Write(token);
    }
    Console.WriteLine();
});

// =============================================================================
// 2. Chat Completion
// =============================================================================
await RunExample("2. Chat Completion", async () =>
{
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
    Console.WriteLine();
});

// =============================================================================
// 3. Temperature Comparison
// =============================================================================
await RunExample("3. Temperature Comparison", async () =>
{
    var prompt = "Write a creative one-sentence story about a robot:";

    Console.WriteLine($"Prompt: {prompt}\n");

    // Low temperature (deterministic)
    Console.Write("Temperature 0.1 (Precise): ");
    await foreach (var token in model.GenerateAsync(
        prompt,
        new GenerationOptions { Temperature = 0.1f, MaxTokens = 50 }))
    {
        Console.Write(token);
    }
    Console.WriteLine("\n");

    // High temperature (creative)
    Console.Write("Temperature 0.9 (Creative): ");
    await foreach (var token in model.GenerateAsync(
        prompt,
        new GenerationOptions { Temperature = 0.9f, MaxTokens = 50 }))
    {
        Console.Write(token);
    }
    Console.WriteLine();
});

// =============================================================================
// 4. Chain-of-Thought (Thinking)
// =============================================================================
await RunExample("4. Chain-of-Thought Reasoning", async () =>
{
    var messages = new[]
    {
        ChatMessage.System("""
            You are a helpful assistant that shows your reasoning process.
            When solving problems, first show your thinking inside <thinking> tags,
            then provide the final answer.
            """),
        ChatMessage.User("If a train travels at 60 mph for 2.5 hours, how far does it go?")
    };

    Console.WriteLine($"User: {messages[1].Content}");
    Console.Write("Assistant: ");

    await foreach (var token in model.GenerateChatAsync(
        messages,
        new GenerationOptions { MaxTokens = 200, Temperature = 0.3f }))
    {
        Console.Write(token);
    }
    Console.WriteLine();
});

// =============================================================================
// 5. JSON Structured Output
// =============================================================================
await RunExample("5. JSON Structured Output", async () =>
{
    var messages = new[]
    {
        ChatMessage.System("""
            You are a helpful assistant that responds ONLY in valid JSON format.
            Do not include any text outside of the JSON object.
            """),
        ChatMessage.User("""
            Extract the following information from this text and return as JSON:
            "John Smith is a 35-year-old software engineer from Seattle who enjoys hiking."

            Required fields: name, age, occupation, city, hobbies (array)
            """)
    };

    Console.WriteLine($"User: Extract info from 'John Smith is a 35-year-old...'");
    Console.Write("Assistant: ");

    var response = "";
    await foreach (var token in model.GenerateChatAsync(
        messages,
        new GenerationOptions { MaxTokens = 150, Temperature = 0.1f }))
    {
        Console.Write(token);
        response += token;
    }
    Console.WriteLine("\n");

    // Try to validate JSON
    try
    {
        var jsonStart = response.IndexOf('{');
        var jsonEnd = response.LastIndexOf('}');
        if (jsonStart >= 0 && jsonEnd > jsonStart)
        {
            var jsonStr = response.Substring(jsonStart, jsonEnd - jsonStart + 1);
            var doc = JsonDocument.Parse(jsonStr);
            Console.WriteLine("✓ Valid JSON parsed successfully!");
        }
    }
    catch (JsonException)
    {
        Console.WriteLine("✗ JSON parsing failed (model may need fine-tuning for structured output)");
    }
});

// =============================================================================
// 6. Tool/Function Calling Simulation
// =============================================================================
await RunExample("6. Tool/Function Calling", async () =>
{
    var toolDefinition = """
        You have access to the following tools:

        1. get_weather(city: string) -> Returns current weather for a city
        2. calculate(expression: string) -> Evaluates a math expression
        3. search_web(query: string) -> Searches the web for information

        When you need to use a tool, respond with a JSON object in this format:
        {"tool": "tool_name", "arguments": {"arg1": "value1"}}

        After the tool result, provide your final answer.
        """;

    var messages = new[]
    {
        ChatMessage.System(toolDefinition),
        ChatMessage.User("What's the weather like in Tokyo today?")
    };

    Console.WriteLine("User: What's the weather like in Tokyo today?");
    Console.Write("Assistant (tool call): ");

    var toolCallResponse = "";
    await foreach (var token in model.GenerateChatAsync(
        messages,
        new GenerationOptions { MaxTokens = 100, Temperature = 0.1f }))
    {
        Console.Write(token);
        toolCallResponse += token;
    }
    Console.WriteLine("\n");

    // Simulate tool execution
    if (toolCallResponse.Contains("get_weather"))
    {
        Console.WriteLine("[Simulating tool execution: get_weather('Tokyo')]");
        Console.WriteLine("Tool Result: { \"temp\": 22, \"condition\": \"Partly cloudy\", \"humidity\": 65 }\n");

        // Continue conversation with tool result
        var continueMessages = new List<ChatMessage>(messages)
        {
            ChatMessage.Assistant(toolCallResponse.Trim()),
            ChatMessage.User("Tool result: Temperature 22°C, Partly cloudy, Humidity 65%")
        };

        Console.Write("Assistant (final answer): ");
        await foreach (var token in model.GenerateChatAsync(
            continueMessages,
            new GenerationOptions { MaxTokens = 100 }))
        {
            Console.Write(token);
        }
        Console.WriteLine();
    }
});

// =============================================================================
// 7. Multi-turn Conversation with Memory
// =============================================================================
await RunExample("7. Multi-turn Conversation", async () =>
{
    var conversation = new List<ChatMessage>
    {
        ChatMessage.System("You are a helpful math tutor. Be concise."),
    };

    // Turn 1
    conversation.Add(ChatMessage.User("What is 15 * 7?"));
    Console.WriteLine("User: What is 15 * 7?");
    Console.Write("Assistant: ");

    var response1 = await CollectResponse(model, conversation);
    Console.WriteLine();

    // Turn 2 (references previous context)
    conversation.Add(ChatMessage.Assistant(response1));
    conversation.Add(ChatMessage.User("Now divide that by 3"));
    Console.WriteLine("User: Now divide that by 3");
    Console.Write("Assistant: ");

    var response2 = await CollectResponse(model, conversation);
    Console.WriteLine();

    // Turn 3 (references both previous results)
    conversation.Add(ChatMessage.Assistant(response2));
    conversation.Add(ChatMessage.User("What's the square root of the original result?"));
    Console.WriteLine("User: What's the square root of the original result?");
    Console.Write("Assistant: ");

    await CollectResponse(model, conversation);
    Console.WriteLine();
});

// =============================================================================
// 8. Role-playing / Persona
// =============================================================================
await RunExample("8. Role-playing / Persona", async () =>
{
    var question = "Explain what an API is.";

    // Persona 1: Teacher
    var teacher = new[]
    {
        ChatMessage.System("You are a friendly elementary school teacher. Explain things simply with examples kids can understand."),
        ChatMessage.User(question)
    };

    Console.WriteLine($"Question: {question}\n");
    Console.Write("👩‍🏫 Teacher: ");
    await foreach (var token in model.GenerateChatAsync(teacher, new GenerationOptions { MaxTokens = 100 }))
    {
        Console.Write(token);
    }
    Console.WriteLine("\n");

    // Persona 2: Pirate
    var pirate = new[]
    {
        ChatMessage.System("You are a pirate who explains technology. Use pirate slang and nautical metaphors."),
        ChatMessage.User(question)
    };

    Console.Write("🏴‍☠️ Pirate: ");
    await foreach (var token in model.GenerateChatAsync(pirate, new GenerationOptions { MaxTokens = 100 }))
    {
        Console.Write(token);
    }
    Console.WriteLine();
});

// =============================================================================
// 9. Text Summarization
// =============================================================================
await RunExample("9. Text Summarization", async () =>
{
    var longText = """
        Artificial intelligence (AI) has rapidly evolved from a theoretical concept to a transformative
        technology that affects nearly every aspect of modern life. Machine learning, a subset of AI,
        enables computers to learn from data and improve their performance over time without being
        explicitly programmed. Deep learning, which uses neural networks with many layers, has achieved
        remarkable results in image recognition, natural language processing, and game playing.
        However, AI also raises important ethical questions about privacy, job displacement, and the
        potential for bias in automated decision-making systems. As AI continues to advance, society
        must grapple with how to harness its benefits while mitigating its risks.
        """;

    var messages = new[]
    {
        ChatMessage.System("You are a summarization assistant. Provide concise summaries in 1-2 sentences."),
        ChatMessage.User($"Summarize this text:\n\n{longText}")
    };

    Console.WriteLine("Original: [Long text about AI - ~100 words]");
    Console.Write("Summary: ");

    await foreach (var token in model.GenerateChatAsync(
        messages,
        new GenerationOptions { MaxTokens = 80, Temperature = 0.3f }))
    {
        Console.Write(token);
    }
    Console.WriteLine();
});

// =============================================================================
// 10. Code Generation
// =============================================================================
await RunExample("10. Code Generation", async () =>
{
    var messages = new[]
    {
        ChatMessage.System("You are a coding assistant. Write clean, well-commented code."),
        ChatMessage.User("Write a Python function to check if a string is a palindrome.")
    };

    Console.WriteLine("User: Write a Python function to check if a string is a palindrome.");
    Console.Write("Assistant:\n");

    await foreach (var token in model.GenerateChatAsync(
        messages,
        new GenerationOptions { MaxTokens = 200, Temperature = 0.2f }))
    {
        Console.Write(token);
    }
    Console.WriteLine();
});

// =============================================================================
// 11. Few-shot Learning
// =============================================================================
await RunExample("11. Few-shot Learning", async () =>
{
    var messages = new[]
    {
        ChatMessage.System("You classify sentiment as positive, negative, or neutral."),
        ChatMessage.User("The movie was fantastic! I loved every minute of it."),
        ChatMessage.Assistant("positive"),
        ChatMessage.User("The service was okay, nothing special."),
        ChatMessage.Assistant("neutral"),
        ChatMessage.User("I wasted my money on this terrible product."),
        ChatMessage.Assistant("negative"),
        ChatMessage.User("The weather today is quite pleasant and I feel great!")
    };

    Console.WriteLine("Few-shot examples provided for sentiment classification");
    Console.WriteLine("New input: 'The weather today is quite pleasant and I feel great!'");
    Console.Write("Predicted sentiment: ");

    await foreach (var token in model.GenerateChatAsync(
        messages,
        new GenerationOptions { MaxTokens = 10, Temperature = 0.1f }))
    {
        Console.Write(token);
    }
    Console.WriteLine();
});

Console.WriteLine("\n=== All Samples Complete ===");

// Helper methods
static async Task RunExample(string title, Func<Task> example)
{
    Console.WriteLine($"\n{title}");
    Console.WriteLine(new string('=', 60));
    try
    {
        await example();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: {ex.Message}");
    }
    Console.WriteLine();
}

static async Task<string> CollectResponse(ITextGenerator model, IEnumerable<ChatMessage> messages)
{
    var response = "";
    await foreach (var token in model.GenerateChatAsync(
        messages,
        new GenerationOptions { MaxTokens = 50 }))
    {
        Console.Write(token);
        response += token;
    }
    return response.Trim();
}

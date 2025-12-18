using FluentAssertions;
using LMSupply.Generator.Abstractions;

namespace LMSupply.Generator.Tests;

/// <summary>
/// Integration tests for Generator models.
/// These tests require actual model downloads and GPU/CPU inference.
/// </summary>
[Trait("Category", "Integration")]
public class GeneratorIntegrationTests
{
    private const string Phi35Model = "microsoft/Phi-3.5-mini-instruct-onnx";
    private const string Phi4Model = "microsoft/Phi-4-mini-instruct-onnx";
    private const string TestPrompt = "What is 2+2? Answer with just the number.";

    // Cached model paths (for direct path loading)
    private static readonly string Phi4CachedPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".cache", "huggingface", "hub",
        "models--microsoft--Phi-4-mini-instruct-onnx",
        "snapshots", "main");

    #region Phi-3.5 Tests

    [Fact]
    public async Task Phi35_DirectML_LoadAndGenerate_ShouldWork()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi35Model, options);

        // Assert - Load
        model.Should().NotBeNull();
        model.ModelId.Should().Be(Phi35Model);

        var info = model.GetModelInfo();
        info.ExecutionProvider.Should().Contain("DML");

        // Act - Generate
        var result = await model.GenerateCompleteAsync(
            TestPrompt,
            new Models.GenerationOptions { MaxTokens = 50 });

        // Assert - Generate
        result.Should().NotBeNullOrWhiteSpace();
        result.Should().Contain("4");
    }

    [Fact]
    public async Task Phi35_CPU_LoadAndGenerate_ShouldWork()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.Cpu,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi35Model, options);

        // Assert - Load
        model.Should().NotBeNull();
        model.ModelId.Should().Be(Phi35Model);

        var info = model.GetModelInfo();
        info.ExecutionProvider.ToUpperInvariant().Should().Contain("CPU");

        // Act - Generate
        var result = await model.GenerateCompleteAsync(
            TestPrompt,
            new Models.GenerationOptions { MaxTokens = 50 });

        // Assert - Generate
        result.Should().NotBeNullOrWhiteSpace();
        result.Should().Contain("4");
    }

    [Fact]
    public async Task Phi35_DirectML_WarmupAsync_ShouldNotThrow()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi35Model, options);

        // Assert
        var warmupAction = () => model.WarmupAsync();
        await warmupAction.Should().NotThrowAsync();
    }

    #endregion

    #region Phi-4 Tests

    [Fact]
    public async Task Phi4_DirectML_LoadAndGenerate_ShouldWork()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi4Model, options);

        // Assert - Load
        model.Should().NotBeNull();
        model.ModelId.Should().Be(Phi4Model);

        var info = model.GetModelInfo();
        info.ExecutionProvider.Should().Contain("DML");

        // Act - Generate
        var result = await model.GenerateCompleteAsync(
            TestPrompt,
            new Models.GenerationOptions { MaxTokens = 50 });

        // Assert - Generate
        result.Should().NotBeNullOrWhiteSpace();
        result.Should().Contain("4");
    }

    [Fact]
    public async Task Phi4_CPU_LoadAndGenerate_ShouldWork()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.Cpu,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi4Model, options);

        // Assert - Load
        model.Should().NotBeNull();
        model.ModelId.Should().Be(Phi4Model);

        var info = model.GetModelInfo();
        info.ExecutionProvider.ToUpperInvariant().Should().Contain("CPU");

        // Act - Generate
        var result = await model.GenerateCompleteAsync(
            TestPrompt,
            new Models.GenerationOptions { MaxTokens = 50 });

        // Assert - Generate
        result.Should().NotBeNullOrWhiteSpace();
        result.Should().Contain("4");
    }

    [Fact]
    public async Task Phi4_DirectML_WarmupAsync_ShouldNotThrow()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadAsync(Phi4Model, options);

        // Assert
        var warmupAction = () => model.WarmupAsync();
        await warmupAction.Should().NotThrowAsync();
    }

    #endregion

    #region Comparison Tests

    [Fact]
    public async Task Phi35_vs_Phi4_DirectML_BothShouldGenerateValidResponse()
    {
        // Phi-3.5
        var phi35Options = new GeneratorOptions { Provider = ExecutionProvider.DirectML };
        await using var phi35Model = await LocalGenerator.LoadAsync(Phi35Model, phi35Options);
        var phi35Result = await phi35Model.GenerateCompleteAsync(
            TestPrompt,
            new Models.GenerationOptions { MaxTokens = 50 });

        // Phi-4
        var phi4Options = new GeneratorOptions { Provider = ExecutionProvider.DirectML };
        await using var phi4Model = await LocalGenerator.LoadAsync(Phi4Model, phi4Options);
        var phi4Result = await phi4Model.GenerateCompleteAsync(
            TestPrompt,
            new Models.GenerationOptions { MaxTokens = 50 });

        // Assert - both should produce valid responses
        phi35Result.Should().NotBeNullOrWhiteSpace("Phi-3.5 should generate response");
        phi4Result.Should().NotBeNullOrWhiteSpace("Phi-4 should generate response");

        // Both should answer "4" for 2+2
        phi35Result.Should().Contain("4", "Phi-3.5 should correctly answer 2+2");
        phi4Result.Should().Contain("4", "Phi-4 should correctly answer 2+2");
    }

    #endregion

    #region Streaming Tests

    [Fact]
    public async Task Phi35_DirectML_StreamingGeneration_ShouldWork()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML
        };

        await using var model = await LocalGenerator.LoadAsync(Phi35Model, options);

        // Act
        var tokens = new List<string>();
        await foreach (var token in model.GenerateAsync(TestPrompt, new Models.GenerationOptions { MaxTokens = 50 }))
        {
            tokens.Add(token);
        }

        // Assert
        tokens.Should().NotBeEmpty();
        string.Join("", tokens).Should().Contain("4");
    }

    [Fact]
    public async Task Phi4_DirectML_StreamingGeneration_ShouldWork()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML
        };

        await using var model = await LocalGenerator.LoadAsync(Phi4Model, options);

        // Act
        var tokens = new List<string>();
        await foreach (var token in model.GenerateAsync(TestPrompt, new Models.GenerationOptions { MaxTokens = 50 }))
        {
            tokens.Add(token);
        }

        // Assert
        tokens.Should().NotBeEmpty();
        string.Join("", tokens).Should().Contain("4");
    }

    #endregion

    #region Direct Path Loading Tests (Using Cached Models)

    [Fact]
    public async Task Phi4_FromPath_DirectML_LoadAndGenerate_ShouldWork()
    {
        // Require cached model
        Assert.True(Directory.Exists(Phi4CachedPath), $"Phi-4 model not cached at {Phi4CachedPath}. Run `huggingface-cli download microsoft/Phi-4-mini-instruct-onnx` first.");

        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML,
            Verbose = true
        };

        // Act - Load directly from path
        await using var model = await LocalGenerator.LoadFromPathAsync(Phi4CachedPath, options);

        // Assert - Load
        model.Should().NotBeNull();

        var info = model.GetModelInfo();
        // DirectML provider name varies - just check it's not CPU
        info.ExecutionProvider.Should().NotBe("Cpu", "Should be using DirectML, not CPU");

        // Act - Generate
        var result = await model.GenerateCompleteAsync(
            TestPrompt,
            new Models.GenerationOptions { MaxTokens = 50 });

        // Assert - Generate
        result.Should().NotBeNullOrWhiteSpace();
        result.Should().Contain("4");
    }

    [Fact]
    public async Task Phi4_FromPath_CPU_LoadAndGenerate_ShouldWork()
    {
        // Require cached model
        Assert.True(Directory.Exists(Phi4CachedPath), $"Phi-4 model not cached at {Phi4CachedPath}. Run `huggingface-cli download microsoft/Phi-4-mini-instruct-onnx` first.");

        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.Cpu,
            Verbose = true
        };

        // Act - Load directly from path
        await using var model = await LocalGenerator.LoadFromPathAsync(Phi4CachedPath, options);

        // Assert - Load
        model.Should().NotBeNull();

        var info = model.GetModelInfo();
        info.ExecutionProvider.ToUpperInvariant().Should().Contain("CPU");

        // Act - Generate
        var result = await model.GenerateCompleteAsync(
            TestPrompt,
            new Models.GenerationOptions { MaxTokens = 50 });

        // Assert - Generate
        result.Should().NotBeNullOrWhiteSpace();
        result.Should().Contain("4");
    }

    [Fact]
    public async Task Phi4_FromPath_DirectML_WarmupAsync_ShouldNotThrow()
    {
        // Require cached model
        Assert.True(Directory.Exists(Phi4CachedPath), $"Phi-4 model not cached at {Phi4CachedPath}. Run `huggingface-cli download microsoft/Phi-4-mini-instruct-onnx` first.");

        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.DirectML,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadFromPathAsync(Phi4CachedPath, options);

        // Assert
        var warmupAction = () => model.WarmupAsync();
        await warmupAction.Should().NotThrowAsync();
    }

    [Fact]
    public async Task Phi4_FromPath_CPU_WarmupAsync_ShouldNotThrow()
    {
        // Require cached model
        Assert.True(Directory.Exists(Phi4CachedPath), $"Phi-4 model not cached at {Phi4CachedPath}. Run `huggingface-cli download microsoft/Phi-4-mini-instruct-onnx` first.");

        // Arrange
        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.Cpu,
            Verbose = true
        };

        // Act
        await using var model = await LocalGenerator.LoadFromPathAsync(Phi4CachedPath, options);

        // Assert
        var warmupAction = () => model.WarmupAsync();
        await warmupAction.Should().NotThrowAsync();
    }

    #endregion
}

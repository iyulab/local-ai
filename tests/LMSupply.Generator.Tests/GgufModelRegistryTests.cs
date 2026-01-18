using FluentAssertions;
using LMSupply.Generator.Internal.Llama;
using Xunit;

namespace LMSupply.Generator.Tests;

public class GgufModelRegistryTests
{
    [Theory]
    [InlineData("gguf:default")]
    [InlineData("gguf:fast")]
    [InlineData("gguf:quality")]
    [InlineData("gguf:large")]
    [InlineData("gguf:multilingual")]
    [InlineData("gguf:korean")]
    [InlineData("gguf:code")]
    [InlineData("gguf:reasoning")]
    public void Resolve_WithPrefixedAlias_ReturnsModelInfo(string alias)
    {
        var result = GgufModelRegistry.Resolve(alias);

        result.Should().NotBeNull();
        result!.RepoId.Should().NotBeNullOrWhiteSpace();
        result.DefaultFile.Should().EndWith(".gguf");
        result.ChatFormat.Should().NotBeNullOrWhiteSpace();
    }

    [Theory]
    [InlineData("default")]
    [InlineData("fast")]
    [InlineData("quality")]
    [InlineData("korean")]
    public void Resolve_WithoutPrefix_ReturnsModelInfo(string alias)
    {
        var result = GgufModelRegistry.Resolve(alias);

        result.Should().NotBeNull();
        result!.RepoId.Should().Contain("/");
    }

    [Theory]
    [InlineData("unknown-model")]
    [InlineData("nonexistent")]
    [InlineData("")]
    [InlineData(null)]
    public void Resolve_WithInvalidAlias_ReturnsNull(string? alias)
    {
        var result = GgufModelRegistry.Resolve(alias!);

        result.Should().BeNull();
    }

    [Fact]
    public void GetAllModels_ReturnsNonEmptyList()
    {
        var models = GgufModelRegistry.GetAllModels();

        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m =>
        {
            m.RepoId.Should().NotBeNullOrWhiteSpace();
            m.DefaultFile.Should().EndWith(".gguf");
            m.ContextLength.Should().BeGreaterThan(0);
        });
    }

    [Fact]
    public void GetAliases_ReturnsExpectedAliases()
    {
        var aliases = GgufModelRegistry.GetAliases();

        aliases.Should().Contain("gguf:default");
        aliases.Should().Contain("gguf:fast");
        aliases.Should().Contain("gguf:quality");
    }

    [Theory]
    [InlineData("gguf:default", true)]
    [InlineData("gguf:fast", true)]
    [InlineData("gguf:quality", true)]
    [InlineData("default", false)] // Plain aliases are reserved for ONNX
    [InlineData("fast", false)]    // Plain aliases are reserved for ONNX
    [InlineData("unknown", false)]
    public void IsAlias_ReturnsCorrectResult(string value, bool expected)
    {
        var result = GgufModelRegistry.IsAlias(value);

        result.Should().Be(expected);
    }

    [Fact]
    public void DefaultModel_HasValidConfiguration()
    {
        var model = GgufModelRegistry.Resolve("gguf:default");

        model.Should().NotBeNull();
        model!.RepoId.Should().Contain("Llama");
        model.ChatFormat.Should().Be("llama3");
        model.DefaultFile.Should().Contain("Q4_K_M");
        model.ContextLength.Should().BeGreaterThanOrEqualTo(4096);
    }

    [Fact]
    public void AllModels_HaveValidChatFormats()
    {
        var validFormats = new[] { "llama3", "chatml", "gemma", "exaone", "deepseek" };
        var models = GgufModelRegistry.GetAllModels();

        models.Should().AllSatisfy(m =>
        {
            validFormats.Should().Contain(m.ChatFormat,
                $"Model {m.DisplayName} has unexpected chat format: {m.ChatFormat}");
        });
    }
}

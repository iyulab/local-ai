using FluentAssertions;
using LMSupply.Generator.Internal;
using LMSupply.Generator.Models;
using Xunit;

namespace LMSupply.Generator.Tests;

public class ModelFormatDetectorTests
{
    [Theory]
    [InlineData("model.gguf", ModelFormat.Gguf)]
    [InlineData("llama-3.2-3b-instruct-Q4_K_M.gguf", ModelFormat.Gguf)]
    [InlineData("C:/models/phi-4.gguf", ModelFormat.Gguf)]
    [InlineData("/home/user/models/qwen.gguf", ModelFormat.Gguf)]
    public void Detect_GgufFileExtension_ReturnsGguf(string path, ModelFormat expected)
    {
        var result = ModelFormatDetector.Detect(path);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("model.onnx", ModelFormat.Onnx)]
    [InlineData("phi-3.5-mini-instruct.onnx", ModelFormat.Onnx)]
    public void Detect_OnnxFileExtension_ReturnsOnnx(string path, ModelFormat expected)
    {
        var result = ModelFormatDetector.Detect(path);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("TheBloke/Llama-2-7B-GGUF", ModelFormat.Gguf)]
    [InlineData("bartowski/Llama-3.2-3B-Instruct-GGUF", ModelFormat.Gguf)]
    [InlineData("mradermacher/Qwen2.5-7B-GGUF", ModelFormat.Gguf)]
    public void Detect_KnownGgufProviders_ReturnsGguf(string repoId, ModelFormat expected)
    {
        var result = ModelFormatDetector.Detect(repoId);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("microsoft/Phi-4-mini-instruct-onnx", ModelFormat.Onnx)]
    [InlineData("microsoft/phi-4-onnx", ModelFormat.Onnx)]
    [InlineData("onnx-community/Llama-3.2-1B-Instruct-ONNX", ModelFormat.Onnx)]
    public void Detect_OnnxInRepoName_ReturnsOnnx(string repoId, ModelFormat expected)
    {
        var result = ModelFormatDetector.Detect(repoId);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("some-user/some-model-GGUF", ModelFormat.Gguf)]
    [InlineData("some-user/model_gguf", ModelFormat.Gguf)]
    public void Detect_GgufInRepoName_ReturnsGguf(string repoId, ModelFormat expected)
    {
        var result = ModelFormatDetector.Detect(repoId);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("microsoft/Phi-3.5-mini-instruct", ModelFormat.Onnx)]  // No format hint, defaults to ONNX
    [InlineData("meta-llama/Llama-3.2-3B-Instruct", ModelFormat.Onnx)]  // No format hint, defaults to ONNX
    public void Detect_NoFormatHint_DefaultsToOnnx(string repoId, ModelFormat expected)
    {
        var result = ModelFormatDetector.Detect(repoId);
        result.Should().Be(expected);
    }

    [Fact]
    public void Detect_EmptyString_ThrowsArgumentException()
    {
        var act = () => ModelFormatDetector.Detect("");
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Detect_NullString_ThrowsArgumentException()
    {
        var act = () => ModelFormatDetector.Detect(null!);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Detect_RegisteredOnnxModel_ReturnsOnnx()
    {
        // Models in ModelRegistry should be detected as ONNX
        var result = ModelFormatDetector.Detect("default");
        result.Should().Be(ModelFormat.Onnx);
    }

    [Theory]
    [InlineData("fast")]
    [InlineData("quality")]
    [InlineData("medium")]
    public void Detect_WellKnownAliases_ReturnsOnnx(string alias)
    {
        // Well-known aliases in registry should resolve to ONNX
        var result = ModelFormatDetector.Detect(alias);
        result.Should().Be(ModelFormat.Onnx);
    }
}

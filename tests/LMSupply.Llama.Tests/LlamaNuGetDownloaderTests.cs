using System.Runtime.InteropServices;
using FluentAssertions;
using LMSupply.Llama;
using LMSupply.Runtime;

namespace LMSupply.Llama.Tests;

public class LlamaNuGetDownloaderTests
{
    [Fact]
    public void Constructor_ShouldCreateInstance()
    {
        using var downloader = new LlamaNuGetDownloader();
        downloader.Should().NotBeNull();
    }

    [Fact]
    public void Constructor_WithCacheDirectory_ShouldCreateInstance()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "lmsupply-test-" + Guid.NewGuid());
        try
        {
            using var downloader = new LlamaNuGetDownloader(tempDir);
            downloader.Should().NotBeNull();
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    [Fact]
    public void Dispose_ShouldNotThrow()
    {
        var downloader = new LlamaNuGetDownloader();
        var action = () => downloader.Dispose();
        action.Should().NotThrow();
    }

    [Fact]
    public void Dispose_MultipleTimes_ShouldNotThrow()
    {
        var downloader = new LlamaNuGetDownloader();
        downloader.Dispose();
        var action = () => downloader.Dispose();
        action.Should().NotThrow();
    }

    [Fact]
    [Trait("Category", "Integration")]
    public async Task DownloadAsync_CpuBackend_ShouldDownloadSuccessfully()
    {
        // This is an integration test that actually downloads from NuGet
        // Skip if no network access
        Skip.If(!await IsNetworkAvailableAsync(), "Network not available");

        var tempDir = Path.Combine(Path.GetTempPath(), "lmsupply-download-test-" + Guid.NewGuid());
        try
        {
            using var downloader = new LlamaNuGetDownloader(tempDir);

            var os = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? OSPlatform.Windows
                   : RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? OSPlatform.Linux
                   : OSPlatform.OSX;

            var platform = new PlatformInfo
            {
                OS = os,
                Architecture = RuntimeInformation.ProcessArchitecture,
                RuntimeIdentifier = RuntimeInformation.RuntimeIdentifier
            };

            var result = await downloader.DownloadAsync(
                LlamaBackend.Cpu,
                platform,
                cancellationToken: CancellationToken.None);

            result.Should().NotBeNullOrEmpty();
            Directory.Exists(result).Should().BeTrue();
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    private static async Task<bool> IsNetworkAvailableAsync()
    {
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            using var response = await client.GetAsync("https://api.nuget.org/v3/index.json");
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }
}

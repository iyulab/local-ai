using System.IO.Compression;
using System.Reflection;
using System.Runtime.InteropServices;
using LMSupply.Download;

namespace LMSupply.Runtime;

/// <summary>
/// Downloads ONNX Runtime packages from NuGet.org and extracts native binaries.
/// Implements LMSupply's on-demand philosophy: binaries are downloaded only when first needed.
/// </summary>
public sealed class OnnxNuGetDownloader : IDisposable
{
    private const string NuGetFlatContainerBase = "https://api.nuget.org/v3-flatcontainer";

    // Package names for each provider
    private static readonly Dictionary<string, string> ProviderPackages = new()
    {
        ["cpu"] = "microsoft.ml.onnxruntime",
        ["directml"] = "microsoft.ml.onnxruntime.directml",
        ["cuda"] = "microsoft.ml.onnxruntime.gpu",
        ["cuda11"] = "microsoft.ml.onnxruntime.gpu",
        ["cuda12"] = "microsoft.ml.onnxruntime.gpu",
    };

    private readonly HttpClient _httpClient;
    private readonly string _cacheDirectory;

    public OnnxNuGetDownloader() : this(null)
    {
    }

    public OnnxNuGetDownloader(string? cacheDirectory)
    {
        _cacheDirectory = cacheDirectory ?? GetDefaultCacheDirectory();
        _httpClient = new HttpClient();
        _httpClient.DefaultRequestHeaders.Add("User-Agent", "LMSupply/1.0");
    }

    /// <summary>
    /// Downloads the ONNX Runtime package and extracts native binaries.
    /// Returns the path to the extracted native library directory.
    /// </summary>
    public async Task<string> DownloadAsync(
        string provider,
        PlatformInfo platform,
        string? version = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Get ONNX Runtime version from assembly if not specified
        version ??= GetOnnxRuntimeVersion();

        // Determine package name
        var providerKey = provider.ToLowerInvariant();
        if (!ProviderPackages.TryGetValue(providerKey, out var packageId))
        {
            packageId = ProviderPackages["cpu"]; // Fallback to CPU
        }

        // Check cache first
        var cachePath = GetCachePath(packageId, version, platform, providerKey);
        if (Directory.Exists(cachePath) && IsValidCache(cachePath, platform))
        {
            progress?.Report(new DownloadProgress
            {
                FileName = "ONNX Runtime native libraries",
                BytesDownloaded = 1,
                TotalBytes = 1
            });
            return cachePath;
        }

        // Download .nupkg from NuGet
        var nupkgUrl = $"{NuGetFlatContainerBase}/{packageId}/{version}/{packageId}.{version}.nupkg";

        progress?.Report(new DownloadProgress
        {
            FileName = $"{packageId}.{version}.nupkg",
            BytesDownloaded = 0,
            TotalBytes = 0
        });

        var tempDir = Path.Combine(Path.GetTempPath(), $"lmsupply-onnx-{Guid.NewGuid()}");

        try
        {
            Directory.CreateDirectory(tempDir);

            // Download nupkg
            var nupkgPath = Path.Combine(tempDir, $"{packageId}.{version}.nupkg");
            await DownloadFileAsync(nupkgUrl, nupkgPath, $"{packageId}.nupkg", progress, cancellationToken);

            // Extract native binaries
            progress?.Report(new DownloadProgress
            {
                FileName = "Extracting native libraries...",
                BytesDownloaded = 0,
                TotalBytes = 0
            });

            var extractedPath = await ExtractNativeBinariesAsync(
                nupkgPath, platform, cancellationToken);

            if (extractedPath is null)
            {
                throw new InvalidOperationException(
                    $"No native binaries found for {platform.RuntimeIdentifier} in {packageId}");
            }

            // Move to cache
            Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);
            if (Directory.Exists(cachePath))
            {
                Directory.Delete(cachePath, recursive: true);
            }
            Directory.Move(extractedPath, cachePath);

            progress?.Report(new DownloadProgress
            {
                FileName = "ONNX Runtime native libraries ready",
                BytesDownloaded = 1,
                TotalBytes = 1
            });

            return cachePath;
        }
        finally
        {
            // Cleanup temp directory
            try
            {
                if (Directory.Exists(tempDir))
                {
                    Directory.Delete(tempDir, recursive: true);
                }
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    /// <summary>
    /// Extracts native binaries from a .nupkg file for the specified platform.
    /// </summary>
    private async Task<string?> ExtractNativeBinariesAsync(
        string nupkgPath,
        PlatformInfo platform,
        CancellationToken cancellationToken)
    {
        var tempExtract = Path.Combine(Path.GetDirectoryName(nupkgPath)!, "extracted");
        Directory.CreateDirectory(tempExtract);

        // .nupkg is a ZIP file
        using (var archive = ZipFile.OpenRead(nupkgPath))
        {
            // NuGet package structure: runtimes/{rid}/native/
            var runtimeIdentifiers = GetRuntimeIdentifiers(platform);

            foreach (var rid in runtimeIdentifiers)
            {
                var nativePrefix = $"runtimes/{rid}/native/";

                var nativeEntries = archive.Entries
                    .Where(e => e.FullName.StartsWith(nativePrefix, StringComparison.OrdinalIgnoreCase)
                                && !string.IsNullOrEmpty(e.Name))
                    .ToList();

                if (nativeEntries.Count > 0)
                {
                    var outputDir = Path.Combine(tempExtract, rid);
                    Directory.CreateDirectory(outputDir);

                    foreach (var entry in nativeEntries)
                    {
                        // ONNX Runtime doesn't have subdirectories like LLamaSharp
                        var destPath = Path.Combine(outputDir, entry.Name);
                        entry.ExtractToFile(destPath, overwrite: true);

                        // Set executable permission on Unix
                        if (!platform.IsWindows)
                        {
                            await SetExecutableAsync(destPath, cancellationToken);
                        }
                    }

                    return outputDir;
                }
            }
        }

        return null;
    }

    /// <summary>
    /// Gets runtime identifiers to search for in order of preference.
    /// </summary>
    private static string[] GetRuntimeIdentifiers(PlatformInfo platform)
    {
        var arch = platform.Architecture == Architecture.Arm64 ? "arm64" : "x64";

        if (platform.IsWindows)
        {
            return [$"win-{arch}", "win"];
        }
        if (platform.IsLinux)
        {
            return [$"linux-{arch}", "linux"];
        }
        if (platform.IsMacOS)
        {
            return [$"osx-{arch}", "osx"];
        }

        return ["any"];
    }

    private string GetCachePath(string packageId, string version, PlatformInfo platform, string provider)
    {
        return Path.Combine(
            _cacheDirectory,
            "onnxruntime",
            provider,
            version,
            platform.RuntimeIdentifier);
    }

    private static bool IsValidCache(string path, PlatformInfo platform)
    {
        if (!Directory.Exists(path))
            return false;

        // Check for expected ONNX Runtime library file
        var expectedLib = platform.IsWindows ? "onnxruntime.dll" :
                          platform.IsMacOS ? "libonnxruntime.dylib" :
                          "libonnxruntime.so";

        return Directory.EnumerateFiles(path, expectedLib + "*").Any();
    }

    private async Task DownloadFileAsync(
        string url,
        string destinationPath,
        string fileName,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        using var response = await _httpClient.GetAsync(
            url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);

        response.EnsureSuccessStatusCode();

        var totalBytes = response.Content.Headers.ContentLength ?? 0;
        var downloadedBytes = 0L;

        await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
        await using var fileStream = new FileStream(
            destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true);

        var buffer = new byte[8192];
        int bytesRead;

        while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
            downloadedBytes += bytesRead;

            progress?.Report(new DownloadProgress
            {
                FileName = fileName,
                TotalBytes = totalBytes,
                BytesDownloaded = downloadedBytes
            });
        }
    }

    private static async Task SetExecutableAsync(string path, CancellationToken cancellationToken)
    {
        try
        {
            // Use chmod on Unix systems
            var process = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = "chmod",
                Arguments = $"+x \"{path}\"",
                UseShellExecute = false,
                CreateNoWindow = true
            });

            if (process != null)
            {
                await process.WaitForExitAsync(cancellationToken);
            }
        }
        catch
        {
            // Ignore errors - not critical
        }
    }

    /// <summary>
    /// Gets the ONNX Runtime version from the loaded assembly.
    /// </summary>
    private static string GetOnnxRuntimeVersion()
    {
        try
        {
            // Try to find ONNX Runtime assembly
            var assembly = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name == "Microsoft.ML.OnnxRuntime");

            if (assembly != null)
            {
                // Try to get informational version
                var infoVersionAttr = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>();
                if (infoVersionAttr != null)
                {
                    var ver = infoVersionAttr.InformationalVersion;
                    var plusIdx = ver.IndexOf('+');
                    if (plusIdx > 0)
                        ver = ver[..plusIdx];
                    if (!string.IsNullOrEmpty(ver) && ver != "0.0.0")
                        return ver;
                }

                // Fallback to assembly version
                var version = assembly.GetName().Version;
                if (version != null && version.Major > 0)
                {
                    return $"{version.Major}.{version.Minor}.{version.Build}";
                }
            }
        }
        catch
        {
            // Ignore
        }

        // Fallback to known version that matches Directory.Packages.props
        return "1.23.2";
    }

    private static string GetDefaultCacheDirectory()
    {
        var baseDir = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return Path.Combine(baseDir, "LMSupply", "cache", "runtimes");
    }

    public void Dispose()
    {
        _httpClient.Dispose();
    }
}

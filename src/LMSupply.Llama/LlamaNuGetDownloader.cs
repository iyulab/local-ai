using System.IO.Compression;
using System.Reflection;
using System.Runtime.InteropServices;
using LMSupply.Runtime;

namespace LMSupply.Llama;

/// <summary>
/// Downloads LLamaSharp backend packages from NuGet.org and extracts native binaries.
/// Implements LMSupply's on-demand philosophy: binaries are downloaded only when first needed.
/// </summary>
public sealed class LlamaNuGetDownloader : IDisposable
{
    private const string NuGetFlatContainerBase = "https://api.nuget.org/v3-flatcontainer";

    // Package names for each backend
    private static readonly Dictionary<LlamaBackend, string> BackendPackages = new()
    {
        [LlamaBackend.Cpu] = "llamasharp.backend.cpu",
        [LlamaBackend.Cuda12] = "llamasharp.backend.cuda12",
        [LlamaBackend.Cuda13] = "llamasharp.backend.cuda12", // Use CUDA 12 for now (CUDA 13 may not exist)
        [LlamaBackend.Vulkan] = "llamasharp.backend.vulkan",
        [LlamaBackend.Metal] = "llamasharp.backend.cpu", // Metal is included in CPU package for macOS
        [LlamaBackend.Rocm] = "llamasharp.backend.cpu", // Fallback to CPU if ROCm not available
    };

    private readonly HttpClient _httpClient;
    private readonly string _cacheDirectory;

    public LlamaNuGetDownloader() : this(null)
    {
    }

    public LlamaNuGetDownloader(string? cacheDirectory)
    {
        _cacheDirectory = cacheDirectory ?? GetDefaultCacheDirectory();
        _httpClient = new HttpClient();
        _httpClient.DefaultRequestHeaders.Add("User-Agent", "LMSupply/1.0");
    }

    /// <summary>
    /// Downloads the LLamaSharp backend package and extracts native binaries.
    /// Returns the path to the extracted native library directory.
    /// </summary>
    public async Task<string> DownloadAsync(
        LlamaBackend backend,
        PlatformInfo platform,
        string? version = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Get LLamaSharp version from assembly if not specified
        version ??= GetLLamaSharpVersion();

        // Determine package name
        if (!BackendPackages.TryGetValue(backend, out var packageId))
        {
            packageId = BackendPackages[LlamaBackend.Cpu]; // Fallback to CPU
        }

        // Check cache first
        var cachePath = GetCachePath(packageId, version, platform);
        if (Directory.Exists(cachePath) && IsValidCache(cachePath, platform))
        {
            progress?.Report(new DownloadProgress
            {
                FileName = "LLamaSharp native libraries",
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

        var tempDir = Path.Combine(Path.GetTempPath(), $"lmsupply-llama-{Guid.NewGuid()}");

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
                FileName = "LLamaSharp native libraries ready",
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
    /// Preserves subdirectory structure (e.g., avx/, avx2/, avx512/) for CPU variant selection.
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
            // NuGet package structure: runtimes/{rid}/native/{variant}/
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
                        // Preserve subdirectory structure (e.g., avx/, avx2/, avx512/)
                        var relativePath = entry.FullName[nativePrefix.Length..];
                        var destPath = Path.Combine(outputDir, relativePath.Replace('/', Path.DirectorySeparatorChar));

                        // Create subdirectory if needed
                        var destDir = Path.GetDirectoryName(destPath);
                        if (!string.IsNullOrEmpty(destDir))
                        {
                            Directory.CreateDirectory(destDir);
                        }

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

    private string GetCachePath(string packageId, string version, PlatformInfo platform)
    {
        return Path.Combine(
            _cacheDirectory,
            "llamasharp",
            version,
            platform.RuntimeIdentifier);
    }

    private static bool IsValidCache(string path, PlatformInfo platform)
    {
        if (!Directory.Exists(path))
            return false;

        // Check for expected llama library file
        var expectedLib = platform.IsWindows ? "llama.dll" :
                          platform.IsMacOS ? "libllama.dylib" :
                          "libllama.so";

        // Search recursively including subdirectories (avx/, avx2/, avx512/, noavx/)
        return Directory.EnumerateFiles(path, expectedLib, SearchOption.AllDirectories).Any();
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
    /// Gets the LLamaSharp version from the loaded assembly.
    /// </summary>
    private static string GetLLamaSharpVersion()
    {
        try
        {
            var assembly = typeof(LLama.LLamaWeights).Assembly;

            // Try to get informational version (includes NuGet version)
            var infoVersionAttr = assembly.GetCustomAttribute<
                AssemblyInformationalVersionAttribute>();
            if (infoVersionAttr != null)
            {
                // Extract version number (strip +commit hash if present)
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
        catch
        {
            // Ignore
        }

        // Fallback to known version that matches Directory.Packages.props
        return "0.25.0";
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

using System.Diagnostics;
using System.IO.Compression;
using LMSupply.Download;

namespace LMSupply.Runtime;

/// <summary>
/// Downloads ONNX Runtime packages from NuGet.org and extracts native binaries.
/// Implements LMSupply's on-demand philosophy: binaries are downloaded only when first needed.
/// Supports both standard ONNX Runtime and ONNX Runtime GenAI packages.
/// </summary>
public sealed class OnnxNuGetDownloader : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly NuGetPackageResolver _packageResolver;
    private readonly string _cacheDirectory;
    private readonly bool _ownsHttpClient;

    public OnnxNuGetDownloader() : this(null)
    {
    }

    public OnnxNuGetDownloader(string? cacheDirectory)
    {
        _cacheDirectory = cacheDirectory ?? GetDefaultCacheDirectory();

        // Create HttpClient first, then wrap in try-catch to ensure cleanup on failure
        var httpClient = new HttpClient();
        try
        {
            httpClient.DefaultRequestHeaders.Add("User-Agent", "LMSupply/1.0");
            _packageResolver = new NuGetPackageResolver(httpClient);
            _httpClient = httpClient;
            _ownsHttpClient = true;
        }
        catch
        {
            httpClient.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Downloads runtime binaries for the specified package type and provider.
    /// </summary>
    /// <param name="provider">The execution provider (cpu, directml, cuda12, etc.).</param>
    /// <param name="platform">The target platform info.</param>
    /// <param name="version">Optional version. If null, uses the latest stable version.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <param name="packageType">The package type: "onnxruntime" (default) or "onnxruntime-genai".</param>
    /// <returns>Path to the extracted native binaries directory.</returns>
    public async Task<string> DownloadAsync(
        string provider,
        PlatformInfo platform,
        string? version = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default,
        string packageType = RuntimePackageRegistry.PackageTypes.OnnxRuntime)
    {
        // Get package configuration from registry
        var config = RuntimePackageRegistry.GetPackageConfig(
            packageType,
            provider,
            platform.RuntimeIdentifier);

        if (config is null)
        {
            throw new InvalidOperationException(
                $"No package configuration found for {packageType}/{provider}");
        }

        // Resolve version dynamically if not specified
        version ??= await ResolveVersionAsync(config.PackageId, cancellationToken);

        // Check cache first
        var cachePath = GetCachePath(packageType, provider, version, platform);
        if (Directory.Exists(cachePath) && IsValidCache(cachePath, config, platform))
        {
            Debug.WriteLine($"[OnnxNuGetDownloader] Using cached binaries: {cachePath}");
            ReportCacheHit(progress);
            return cachePath;
        }

        // Download and extract
        return await DownloadAndExtractAsync(
            config,
            version,
            platform,
            cachePath,
            progress,
            cancellationToken);
    }

    /// <summary>
    /// Resolves the version to use, either from assembly or NuGet API.
    /// Validates that the version exists for the specific package before using it.
    /// </summary>
    private async Task<string> ResolveVersionAsync(
        string packageId,
        CancellationToken cancellationToken)
    {
        // First try to detect from loaded assembly
        var assemblyVersion = TryGetAssemblyVersion(packageId);
        if (!string.IsNullOrEmpty(assemblyVersion))
        {
            // Verify this version exists for the specific package
            // (e.g., DirectML may have different version than base ONNX Runtime)
            var availableVersions = await _packageResolver.GetVersionsAsync(packageId, cancellationToken);
            if (availableVersions.Contains(assemblyVersion, StringComparer.OrdinalIgnoreCase))
            {
                Debug.WriteLine($"[OnnxNuGetDownloader] Using assembly version: {assemblyVersion}");
                return assemblyVersion;
            }

            Debug.WriteLine($"[OnnxNuGetDownloader] Assembly version {assemblyVersion} not found for {packageId}, falling back to latest");
        }

        // Get latest stable version from NuGet
        var latestVersion = await _packageResolver.GetLatestVersionAsync(
            packageId,
            includePrerelease: false,
            cancellationToken);

        if (string.IsNullOrEmpty(latestVersion))
        {
            throw new InvalidOperationException(
                $"Could not determine version for package {packageId}. " +
                "Please specify a version explicitly.");
        }

        Debug.WriteLine($"[OnnxNuGetDownloader] Using latest NuGet version: {latestVersion}");
        return latestVersion;
    }

    /// <summary>
    /// Downloads a package and extracts native binaries.
    /// </summary>
    private async Task<string> DownloadAndExtractAsync(
        RuntimePackageRegistry.PackageConfig config,
        string version,
        PlatformInfo platform,
        string cachePath,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var downloadUrl = NuGetPackageResolver.GetPackageDownloadUrl(config.PackageId, version);

        progress?.Report(new DownloadProgress
        {
            FileName = $"{config.PackageId}.{version}.nupkg",
            BytesDownloaded = 0,
            TotalBytes = 0
        });

        var tempDir = Path.Combine(Path.GetTempPath(), $"lmsupply-onnx-{Guid.NewGuid()}");

        try
        {
            Directory.CreateDirectory(tempDir);

            // Download .nupkg
            var nupkgPath = Path.Combine(tempDir, $"{config.PackageId}.{version}.nupkg");
            await DownloadFileAsync(downloadUrl, nupkgPath, config.PackageId, progress, cancellationToken);

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
                    $"No native binaries found for {platform.RuntimeIdentifier} in {config.PackageId}");
            }

            // Move to cache
            EnsureCacheDirectory(cachePath);
            Directory.Move(extractedPath, cachePath);

            progress?.Report(new DownloadProgress
            {
                FileName = $"{config.NativeLibraryName} ready",
                BytesDownloaded = 1,
                TotalBytes = 1
            });

            return cachePath;
        }
        finally
        {
            CleanupTempDirectory(tempDir);
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

        using var archive = ZipFile.OpenRead(nupkgPath);

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

        return null;
    }

    /// <summary>
    /// Gets runtime identifiers to search for in order of preference.
    /// </summary>
    private static string[] GetRuntimeIdentifiers(PlatformInfo platform)
    {
        var arch = platform.Architecture == System.Runtime.InteropServices.Architecture.Arm64 ? "arm64" : "x64";

        if (platform.IsWindows)
            return [$"win-{arch}", "win"];
        if (platform.IsLinux)
            return [$"linux-{arch}", "linux"];
        if (platform.IsMacOS)
            return [$"osx-{arch}", "osx"];

        return ["any"];
    }

    private string GetCachePath(string packageType, string provider, string version, PlatformInfo platform)
    {
        return Path.Combine(
            _cacheDirectory,
            packageType,
            provider.ToLowerInvariant(),
            version,
            platform.RuntimeIdentifier);
    }

    private static bool IsValidCache(
        string path,
        RuntimePackageRegistry.PackageConfig config,
        PlatformInfo platform)
    {
        if (!Directory.Exists(path))
            return false;

        var expectedLib = RuntimePackageRegistry.GetNativeLibraryFileName(
            config.NativeLibraryName, platform);

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
            var process = Process.Start(new ProcessStartInfo
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

    private static string? TryGetAssemblyVersion(string packageId)
    {
        try
        {
            // Map package ID to assembly name
            var assemblyName = packageId.Replace(".ML.", ".ML.")
                .Replace("GenAI", "RuntimeGenAI");

            // Handle specific mappings
            if (packageId.Contains("OnnxRuntimeGenAI", StringComparison.OrdinalIgnoreCase))
            {
                assemblyName = "Microsoft.ML.OnnxRuntimeGenAI";
            }
            else if (packageId.Contains("OnnxRuntime", StringComparison.OrdinalIgnoreCase))
            {
                assemblyName = "Microsoft.ML.OnnxRuntime";
            }

            var assembly = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name?.Equals(assemblyName, StringComparison.OrdinalIgnoreCase) == true);

            if (assembly != null)
            {
                var infoAttr = assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
                    .OfType<System.Reflection.AssemblyInformationalVersionAttribute>()
                    .FirstOrDefault();

                if (infoAttr != null)
                {
                    var ver = infoAttr.InformationalVersion;
                    var plusIdx = ver.IndexOf('+');
                    if (plusIdx > 0)
                        ver = ver[..plusIdx];
                    if (!string.IsNullOrEmpty(ver) && ver != "0.0.0")
                        return ver;
                }

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

        return null;
    }

    private static void ReportCacheHit(IProgress<DownloadProgress>? progress)
    {
        progress?.Report(new DownloadProgress
        {
            FileName = "Using cached runtime (already downloaded)",
            BytesDownloaded = 1,
            TotalBytes = 1
        });
    }

    private static void EnsureCacheDirectory(string cachePath)
    {
        var parentDir = Path.GetDirectoryName(cachePath);
        if (!string.IsNullOrEmpty(parentDir))
        {
            Directory.CreateDirectory(parentDir);
        }

        if (Directory.Exists(cachePath))
        {
            Directory.Delete(cachePath, recursive: true);
        }
    }

    private static void CleanupTempDirectory(string tempDir)
    {
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

    private static string GetDefaultCacheDirectory()
    {
        var baseDir = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return Path.Combine(baseDir, "LMSupply", "cache", "runtimes");
    }

    public void Dispose()
    {
        _packageResolver.Dispose();
        if (_ownsHttpClient)
        {
            _httpClient.Dispose();
        }
    }
}

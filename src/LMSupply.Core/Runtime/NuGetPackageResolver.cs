using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace LMSupply.Runtime;

/// <summary>
/// Resolves NuGet package information using the NuGet API v3.
/// Provides dynamic version discovery without hardcoding.
/// </summary>
public sealed class NuGetPackageResolver : IDisposable
{
    private const string NuGetFlatContainerBase = "https://api.nuget.org/v3-flatcontainer";

    private readonly HttpClient _httpClient;
    private readonly Dictionary<string, string[]> _versionCache = new();
    private readonly SemaphoreSlim _cacheLock = new(1, 1);

    public NuGetPackageResolver(HttpClient? httpClient = null)
    {
        _httpClient = httpClient ?? new HttpClient();
        _httpClient.DefaultRequestHeaders.TryAddWithoutValidation("User-Agent", "LMSupply/1.0");
    }

    /// <summary>
    /// Gets all available versions for a package, sorted descending (latest first).
    /// </summary>
    public async Task<IReadOnlyList<string>> GetVersionsAsync(
        string packageId,
        CancellationToken cancellationToken = default)
    {
        var normalizedId = packageId.ToLowerInvariant();

        await _cacheLock.WaitAsync(cancellationToken);
        try
        {
            if (_versionCache.TryGetValue(normalizedId, out var cached))
            {
                return cached;
            }

            var url = $"{NuGetFlatContainerBase}/{normalizedId}/index.json";
            var response = await _httpClient.GetFromJsonAsync<VersionsResponse>(url, cancellationToken);

            var versions = response?.Versions ?? [];
            // Sort descending by semantic version (latest first)
            var sorted = versions
                .Select(v => (Original: v, Parsed: TryParseVersion(v)))
                .Where(x => x.Parsed != null)
                .OrderByDescending(x => x.Parsed)
                .Select(x => x.Original)
                .ToArray();

            _versionCache[normalizedId] = sorted;
            return sorted;
        }
        finally
        {
            _cacheLock.Release();
        }
    }

    /// <summary>
    /// Gets the latest stable version of a package.
    /// </summary>
    public async Task<string?> GetLatestVersionAsync(
        string packageId,
        bool includePrerelease = false,
        CancellationToken cancellationToken = default)
    {
        var versions = await GetVersionsAsync(packageId, cancellationToken);

        return includePrerelease
            ? versions.FirstOrDefault()
            : versions.FirstOrDefault(v => !IsPrerelease(v));
    }

    /// <summary>
    /// Constructs the download URL for a specific package version.
    /// </summary>
    public static string GetPackageDownloadUrl(string packageId, string version)
    {
        var normalizedId = packageId.ToLowerInvariant();
        var normalizedVersion = version.ToLowerInvariant();
        return $"{NuGetFlatContainerBase}/{normalizedId}/{normalizedVersion}/{normalizedId}.{normalizedVersion}.nupkg";
    }

    /// <summary>
    /// Validates that a package exists and returns its latest version.
    /// </summary>
    public async Task<(bool Exists, string? LatestVersion)> ValidatePackageAsync(
        string packageId,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var versions = await GetVersionsAsync(packageId, cancellationToken);
            return (versions.Count > 0, versions.FirstOrDefault(v => !IsPrerelease(v)));
        }
        catch (HttpRequestException)
        {
            return (false, null);
        }
    }

    private static bool IsPrerelease(string version)
    {
        return version.Contains('-', StringComparison.Ordinal);
    }

    private static Version? TryParseVersion(string versionString)
    {
        // Strip prerelease suffix for comparison
        var dashIndex = versionString.IndexOf('-');
        var cleanVersion = dashIndex > 0 ? versionString[..dashIndex] : versionString;

        return Version.TryParse(cleanVersion, out var version) ? version : null;
    }

    public void Dispose()
    {
        _cacheLock.Dispose();
    }

    private sealed record VersionsResponse(
        [property: JsonPropertyName("versions")] string[] Versions);
}

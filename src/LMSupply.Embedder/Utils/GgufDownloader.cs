using System.Net.Http.Json;
using System.Text.Json.Serialization;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Embedder.Utils;

/// <summary>
/// Downloads GGUF embedding model files from HuggingFace.
/// </summary>
internal sealed class GgufDownloader : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string _cacheDirectory;
    private bool _disposed;

    private const string HuggingFaceApiBase = "https://huggingface.co/api/models";
    private const string HuggingFaceFileBase = "https://huggingface.co";

    /// <summary>
    /// Default quantization preference order for embedding models.
    /// </summary>
    private static readonly string[] DefaultQuantizationPriority =
    [
        "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "F16"
    ];

    public GgufDownloader(string cacheDirectory)
    {
        _cacheDirectory = cacheDirectory;
        _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(30)
        };
        _httpClient.DefaultRequestHeaders.Add("User-Agent", "LMSupply/1.0");
    }

    /// <summary>
    /// Downloads a GGUF embedding model file from HuggingFace.
    /// </summary>
    public async Task<string> DownloadAsync(
        string repoId,
        string? preferredQuantization = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // List files in the repository
        var files = await ListRepoFilesAsync(repoId, cancellationToken);
        var ggufFiles = files.Where(f => f.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).ToList();

        if (ggufFiles.Count == 0)
        {
            throw new ModelNotFoundException(
                $"No GGUF files found in repository '{repoId}'.",
                repoId);
        }

        // Select best file based on quantization preference
        var selectedFile = SelectBestFile(ggufFiles, preferredQuantization);

        // Check cache
        var cachePath = GetCachePath(repoId, selectedFile.Path);
        if (File.Exists(cachePath))
        {
            progress?.Report(new DownloadProgress
            {
                FileName = selectedFile.Path,
                BytesDownloaded = 1,
                TotalBytes = 1
            });
            return cachePath;
        }

        // Download the file
        var downloadUrl = $"{HuggingFaceFileBase}/{repoId}/resolve/main/{selectedFile.Path}";

        progress?.Report(new DownloadProgress
        {
            FileName = selectedFile.Path,
            BytesDownloaded = 0,
            TotalBytes = selectedFile.Size
        });

        Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);
        await DownloadFileAsync(downloadUrl, cachePath, selectedFile.Path, selectedFile.Size, progress, cancellationToken);

        return cachePath;
    }

    private async Task<List<RepoFile>> ListRepoFilesAsync(string repoId, CancellationToken cancellationToken)
    {
        var url = $"{HuggingFaceApiBase}/{repoId}";
        var response = await _httpClient.GetFromJsonAsync<RepoInfo>(url, cancellationToken);

        return response?.Siblings?.ToList() ?? [];
    }

    private RepoFile SelectBestFile(List<RepoFile> files, string? preferredQuantization)
    {
        // If specific quantization requested, find it
        if (!string.IsNullOrEmpty(preferredQuantization))
        {
            var match = files.FirstOrDefault(f =>
                f.Path.Contains(preferredQuantization, StringComparison.OrdinalIgnoreCase));
            if (match != null)
                return match;
        }

        // Try default priority order
        foreach (var quant in DefaultQuantizationPriority)
        {
            var match = files.FirstOrDefault(f =>
                f.Path.Contains(quant, StringComparison.OrdinalIgnoreCase));
            if (match != null)
                return match;
        }

        // Fall back to smallest file (likely most quantized)
        return files.OrderBy(f => f.Size).First();
    }

    private string GetCachePath(string repoId, string filename)
    {
        var safeRepoId = repoId.Replace('/', '_').Replace('\\', '_');
        return Path.Combine(_cacheDirectory, "gguf-embeddings", safeRepoId, filename);
    }

    private async Task DownloadFileAsync(
        string url,
        string destinationPath,
        string fileName,
        long totalBytes,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        using var response = await _httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();

        var contentLength = response.Content.Headers.ContentLength ?? totalBytes;
        var downloadedBytes = 0L;

        await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
        await using var fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true);

        var buffer = new byte[81920]; // 80KB buffer
        int bytesRead;

        while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
            downloadedBytes += bytesRead;

            progress?.Report(new DownloadProgress
            {
                FileName = fileName,
                TotalBytes = contentLength,
                BytesDownloaded = downloadedBytes
            });
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _httpClient.Dispose();
            _disposed = true;
        }
    }

    private record RepoInfo
    {
        [JsonPropertyName("siblings")]
        public RepoFile[]? Siblings { get; init; }
    }

    private record RepoFile
    {
        [JsonPropertyName("rfilename")]
        public string Path { get; init; } = "";

        [JsonPropertyName("size")]
        public long Size { get; init; }
    }
}

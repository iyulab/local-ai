using LMSupply.Download;

namespace LMSupply.Console.Host.Services;

/// <summary>
/// HuggingFace cache management service (thin wrapper over CacheManager).
/// </summary>
public sealed class CacheService
{
    private readonly string _cacheDirectory;
    private readonly ILogger<CacheService> _logger;

    public CacheService(IConfiguration configuration, ILogger<CacheService> logger)
    {
        _cacheDirectory = configuration["ModelManager:CacheDirectory"]
            ?? CacheManager.GetDefaultCacheDirectory();
        _logger = logger;

        _logger.LogInformation("Cache directory: {CacheDirectory}", _cacheDirectory);
    }

    /// <summary>
    /// Cache directory path.
    /// </summary>
    public string CacheDirectory => _cacheDirectory;

    /// <summary>
    /// Gets all cached models.
    /// </summary>
    public IReadOnlyList<CachedModelInfo> GetCachedModels()
        => CacheManager.GetCachedModelsWithInfo(_cacheDirectory);

    /// <summary>
    /// Gets cached models by type (excludes incomplete models).
    /// </summary>
    public IReadOnlyList<CachedModelInfo> GetCachedModelsByType(ModelType type)
        => CacheManager.GetCachedModelsByType(_cacheDirectory, type);

    /// <summary>
    /// Deletes a cached model.
    /// </summary>
    /// <returns>True if the model was found and deleted, false otherwise.</returns>
    public bool DeleteModel(string repoId)
    {
        try
        {
            var deleted = CacheManager.DeleteModel(_cacheDirectory, repoId);
            if (deleted)
            {
                _logger.LogInformation("Deleted model: {RepoId}", repoId);
            }
            else
            {
                _logger.LogWarning("Model not found for deletion: {RepoId}", repoId);
            }
            return deleted;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to delete model: {RepoId}", repoId);
            return false;
        }
    }

    /// <summary>
    /// Gets total cache size.
    /// </summary>
    public long GetTotalCacheSize()
        => CacheManager.GetTotalCacheSize(_cacheDirectory);
}

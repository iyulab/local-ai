using System.Diagnostics;
using System.Runtime.InteropServices;

namespace LMSupply.Runtime;

/// <summary>
/// Provides dynamic detection and configuration of CUDA and cuDNN environments.
/// Eliminates hardcoded paths and version numbers by scanning the system.
/// </summary>
public sealed class CudaEnvironment
{
    private static CudaEnvironment? _instance;
    private static readonly object _lock = new();

    private readonly List<CudaInstallation> _cudaInstallations = new();
    private readonly List<CuDnnInstallation> _cudnnInstallations = new();
    private CudaInstallation? _primaryCuda;  // The one from CUDA_PATH
    private bool _isInitialized;

    /// <summary>
    /// Gets the singleton instance of CudaEnvironment.
    /// </summary>
    public static CudaEnvironment Instance
    {
        get
        {
            if (_instance is null)
            {
                lock (_lock)
                {
                    _instance ??= new CudaEnvironment();
                }
            }
            return _instance;
        }
    }

    private CudaEnvironment()
    {
    }

    /// <summary>
    /// Initializes the CUDA environment detection.
    /// </summary>
    public void Initialize()
    {
        if (_isInitialized) return;

        lock (_lock)
        {
            if (_isInitialized) return;

            DetectCudaInstallations();
            DetectCuDnnInstallations();
            _isInitialized = true;
        }
    }

    /// <summary>
    /// Gets all detected CUDA installations.
    /// </summary>
    public IReadOnlyList<CudaInstallation> CudaInstallations
    {
        get
        {
            Initialize();
            return _cudaInstallations;
        }
    }

    /// <summary>
    /// Gets all detected cuDNN installations.
    /// </summary>
    public IReadOnlyList<CuDnnInstallation> CuDnnInstallations
    {
        get
        {
            Initialize();
            return _cudnnInstallations;
        }
    }

    /// <summary>
    /// Gets the primary CUDA installation (from CUDA_PATH environment variable).
    /// This is the toolkit the user has configured as their primary CUDA installation.
    /// </summary>
    public CudaInstallation? PrimaryCuda
    {
        get
        {
            Initialize();
            return _primaryCuda;
        }
    }

    /// <summary>
    /// Gets the best matching cuDNN installation for the given CUDA version.
    /// </summary>
    public CuDnnInstallation? GetBestCuDnn(int cudaMajorVersion)
    {
        Initialize();
        return _cudnnInstallations
            .Where(c => c.CudaMajorVersion == cudaMajorVersion)
            .OrderByDescending(c => c.Version)
            .FirstOrDefault();
    }

    /// <summary>
    /// Checks if CUDA runtime libraries are available for the specified CUDA major version.
    /// </summary>
    public (bool Available, string[] MissingLibraries) CheckCudaLibraries(int cudaMajorVersion)
    {
        Initialize();
        var missing = new List<string>();

        var cuda = _cudaInstallations.FirstOrDefault(c => c.MajorVersion == cudaMajorVersion);
        if (cuda is null)
        {
            return (false, new[] { $"CUDA {cudaMajorVersion}.x not found" });
        }

        // Check for required cuBLAS libraries
        var cublasLibs = GetCublasLibraryNames(cudaMajorVersion);
        foreach (var lib in cublasLibs)
        {
            var libPath = Path.Combine(cuda.BinPath, lib);
            if (!File.Exists(libPath))
            {
                missing.Add(lib);
            }
        }

        return missing.Count == 0 ? (true, Array.Empty<string>()) : (false, missing.ToArray());
    }

    /// <summary>
    /// Checks if cuDNN is available and properly configured for the specified CUDA version.
    /// </summary>
    public (bool Available, string[] MissingLibraries, string? BinPath) CheckCuDnnLibraries(int cudaMajorVersion)
    {
        Initialize();
        var missing = new List<string>();

        var cudnn = GetBestCuDnn(cudaMajorVersion);
        if (cudnn is null)
        {
            return (false, new[] { $"cuDNN for CUDA {cudaMajorVersion}.x not found" }, null);
        }

        // Check for main cuDNN library
        var cudnnDll = GetCuDnnLibraryName(cudnn.MajorVersion);
        if (!File.Exists(Path.Combine(cudnn.BinPath, cudnnDll)))
        {
            missing.Add(cudnnDll);
        }

        // Check for zlib dependency (required for cuDNN 8.3+)
        if (cudnn.MajorVersion >= 8)
        {
            var zlibFound = CheckZlibAvailability(cudnn.BinPath);
            if (!zlibFound)
            {
                missing.Add("zlibwapi.dll (required dependency, download from NVIDIA or use zlib from Nsight Systems)");
            }
        }

        return missing.Count == 0
            ? (true, Array.Empty<string>(), cudnn.BinPath)
            : (false, missing.ToArray(), cudnn.BinPath);
    }

    /// <summary>
    /// Gets all paths that should be added to the DLL search path for CUDA/cuDNN.
    /// </summary>
    public IEnumerable<string> GetDllSearchPaths(int cudaMajorVersion)
    {
        Initialize();
        var paths = new List<string>();

        // Add CUDA bin path
        var cuda = _cudaInstallations.FirstOrDefault(c => c.MajorVersion == cudaMajorVersion);
        if (cuda is not null && Directory.Exists(cuda.BinPath))
        {
            paths.Add(cuda.BinPath);
        }

        // Add cuDNN paths (prioritize matching CUDA version)
        var matchingCudnn = _cudnnInstallations
            .Where(c => c.CudaMajorVersion == cudaMajorVersion)
            .OrderByDescending(c => c.Version);

        foreach (var cudnn in matchingCudnn)
        {
            if (Directory.Exists(cudnn.BinPath))
            {
                paths.Add(cudnn.BinPath);
            }
        }

        // Also add other cuDNN versions as fallback
        var otherCudnn = _cudnnInstallations
            .Where(c => c.CudaMajorVersion != cudaMajorVersion)
            .OrderByDescending(c => c.Version);

        foreach (var cudnn in otherCudnn)
        {
            if (Directory.Exists(cudnn.BinPath))
            {
                paths.Add(cudnn.BinPath);
            }
        }

        return paths.Distinct();
    }

    /// <summary>
    /// Gets diagnostic information about the CUDA/cuDNN environment.
    /// </summary>
    public string GetDiagnostics()
    {
        Initialize();
        var sb = new System.Text.StringBuilder();

        sb.AppendLine("=== CUDA Environment Diagnostics ===");
        sb.AppendLine();

        sb.AppendLine("CUDA Installations:");
        if (_cudaInstallations.Count == 0)
        {
            sb.AppendLine("  (none found)");
        }
        else
        {
            foreach (var cuda in _cudaInstallations)
            {
                sb.AppendLine($"  - CUDA {cuda.Version} at {cuda.Path}");
                sb.AppendLine($"    bin: {cuda.BinPath} (exists: {Directory.Exists(cuda.BinPath)})");
            }
        }

        sb.AppendLine();
        sb.AppendLine("cuDNN Installations:");
        if (_cudnnInstallations.Count == 0)
        {
            sb.AppendLine("  (none found)");
        }
        else
        {
            foreach (var cudnn in _cudnnInstallations)
            {
                sb.AppendLine($"  - cuDNN {cudnn.Version} for CUDA {cudnn.CudaMajorVersion}.x at {cudnn.BinPath}");
                var zlibStatus = CheckZlibAvailability(cudnn.BinPath) ? "found" : "MISSING";
                sb.AppendLine($"    zlibwapi.dll: {zlibStatus}");
            }
        }

        return sb.ToString();
    }

    #region Detection Methods

    private void DetectCudaInstallations()
    {
        if (!OperatingSystem.IsWindows()) return;

        // Primary CUDA from CUDA_PATH (this is the user's configured primary)
        var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        if (!string.IsNullOrEmpty(cudaPath) && Directory.Exists(cudaPath))
        {
            var version = ParseCudaVersionFromPath(cudaPath);
            if (version is not null)
            {
                _primaryCuda = new CudaInstallation(cudaPath, version);
                _cudaInstallations.Add(_primaryCuda);
            }
        }

        // Additional CUDA versions from CUDA_PATH_V{major}_{minor}
        var envVars = Environment.GetEnvironmentVariables();
        foreach (string key in envVars.Keys)
        {
            if (key.StartsWith("CUDA_PATH_V", StringComparison.OrdinalIgnoreCase) && key != "CUDA_PATH")
            {
                var path = envVars[key]?.ToString();
                if (!string.IsNullOrEmpty(path) && Directory.Exists(path))
                {
                    var version = ParseCudaVersionFromEnvVar(key) ?? ParseCudaVersionFromPath(path);
                    if (version is not null && !_cudaInstallations.Any(c => c.Path == path))
                    {
                        _cudaInstallations.Add(new CudaInstallation(path, version));
                    }
                }
            }
        }

        // Sort by version descending
        _cudaInstallations.Sort((a, b) => b.Version.CompareTo(a.Version));
    }

    private void DetectCuDnnInstallations()
    {
        if (!OperatingSystem.IsWindows()) return;

        // Standard cuDNN installation paths
        var searchPaths = new[]
        {
            @"C:\Program Files\NVIDIA\CUDNN",
            @"C:\Program Files (x86)\NVIDIA\CUDNN",
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "NVIDIA", "CUDNN"),
        };

        foreach (var basePath in searchPaths.Distinct())
        {
            if (!Directory.Exists(basePath)) continue;

            // Scan version directories (v8.x, v9.x, etc.)
            foreach (var versionDir in Directory.GetDirectories(basePath, "v*"))
            {
                var cudnnVersion = ParseCuDnnVersionFromPath(versionDir);
                if (cudnnVersion is null) continue;

                var binDir = Path.Combine(versionDir, "bin");
                if (!Directory.Exists(binDir)) continue;

                // Check for CUDA version-specific subdirectories (e.g., bin\12.9)
                foreach (var cudaVersionDir in Directory.GetDirectories(binDir))
                {
                    var cudaVersion = ParseCudaVersionFromDirName(Path.GetFileName(cudaVersionDir));
                    if (cudaVersion is not null)
                    {
                        var cudnnDll = GetCuDnnLibraryName(cudnnVersion.Major);
                        if (File.Exists(Path.Combine(cudaVersionDir, cudnnDll)))
                        {
                            _cudnnInstallations.Add(new CuDnnInstallation(
                                cudaVersionDir,
                                cudnnVersion,
                                cudaVersion.Major));
                        }
                    }
                }

                // Also check the bin directory itself
                var mainCudnnDll = GetCuDnnLibraryName(cudnnVersion.Major);
                if (File.Exists(Path.Combine(binDir, mainCudnnDll)))
                {
                    // Determine CUDA version from the first cuda version directory found
                    var cudaMajor = Directory.GetDirectories(binDir)
                        .Select(d => ParseCudaVersionFromDirName(Path.GetFileName(d)))
                        .FirstOrDefault(v => v is not null)?.Major ?? 12; // Default to CUDA 12

                    if (!_cudnnInstallations.Any(c => c.BinPath == binDir))
                    {
                        _cudnnInstallations.Add(new CuDnnInstallation(binDir, cudnnVersion, cudaMajor));
                    }
                }
            }
        }

        // Also check if cuDNN is installed in CUDA bin directory
        foreach (var cuda in _cudaInstallations)
        {
            foreach (var cudnnMajor in new[] { 9, 8 })
            {
                var cudnnDll = GetCuDnnLibraryName(cudnnMajor);
                var cudnnPath = Path.Combine(cuda.BinPath, cudnnDll);
                if (File.Exists(cudnnPath))
                {
                    var version = new Version(cudnnMajor, 0);
                    if (!_cudnnInstallations.Any(c => c.BinPath == cuda.BinPath && c.MajorVersion == cudnnMajor))
                    {
                        _cudnnInstallations.Add(new CuDnnInstallation(cuda.BinPath, version, cuda.MajorVersion));
                    }
                }
            }
        }

        // Sort by version descending, then by CUDA version match
        _cudnnInstallations.Sort((a, b) =>
        {
            var versionCompare = b.Version.CompareTo(a.Version);
            return versionCompare != 0 ? versionCompare : b.CudaMajorVersion.CompareTo(a.CudaMajorVersion);
        });
    }

    #endregion

    #region Helper Methods

    private static Version? ParseCudaVersionFromPath(string path)
    {
        // Extract version from path like "C:\...\CUDA\v12.9" or "...\CUDA\v12.9"
        var dirName = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar));
        return ParseVersionString(dirName);
    }

    private static Version? ParseCudaVersionFromEnvVar(string envVarName)
    {
        // Extract version from "CUDA_PATH_V12_9" -> 12.9
        if (!envVarName.StartsWith("CUDA_PATH_V", StringComparison.OrdinalIgnoreCase))
            return null;

        var versionPart = envVarName.Substring(11).Replace('_', '.');
        return Version.TryParse(versionPart, out var version) ? version : null;
    }

    private static Version? ParseCuDnnVersionFromPath(string path)
    {
        // Extract version from "v9.17" or "v8.9.7"
        var dirName = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar));
        return ParseVersionString(dirName);
    }

    private static Version? ParseCudaVersionFromDirName(string dirName)
    {
        // Parse "12.9" or "12" from directory name
        if (Version.TryParse(dirName, out var version))
            return version;

        if (int.TryParse(dirName, out var major))
            return new Version(major, 0);

        return null;
    }

    private static Version? ParseVersionString(string input)
    {
        // Remove 'v' prefix if present
        var cleaned = input.StartsWith("v", StringComparison.OrdinalIgnoreCase)
            ? input.Substring(1)
            : input;

        return Version.TryParse(cleaned, out var version) ? version : null;
    }

    /// <summary>
    /// Gets the cuBLAS library names for the specified CUDA major version.
    /// </summary>
    public static string[] GetCublasLibraryNames(int cudaMajorVersion)
    {
        // cuBLAS libraries are named with CUDA major version suffix
        return new[]
        {
            $"cublas64_{cudaMajorVersion}.dll",
            $"cublasLt64_{cudaMajorVersion}.dll"
        };
    }

    /// <summary>
    /// Gets the cuDNN main library name for the specified cuDNN major version.
    /// </summary>
    public static string GetCuDnnLibraryName(int cudnnMajorVersion)
    {
        // cuDNN 8+: cudnn64_8.dll, cudnn64_9.dll, etc.
        return $"cudnn64_{cudnnMajorVersion}.dll";
    }

    /// <summary>
    /// Gets all cuDNN component library names for the specified version.
    /// </summary>
    public static string[] GetCuDnnComponentLibraries(int cudnnMajorVersion)
    {
        return new[]
        {
            $"cudnn64_{cudnnMajorVersion}.dll",
            $"cudnn_adv64_{cudnnMajorVersion}.dll",
            $"cudnn_cnn64_{cudnnMajorVersion}.dll",
            $"cudnn_ops64_{cudnnMajorVersion}.dll",
            $"cudnn_graph64_{cudnnMajorVersion}.dll",
            $"cudnn_engines_precompiled64_{cudnnMajorVersion}.dll",
            $"cudnn_engines_runtime_compiled64_{cudnnMajorVersion}.dll",
            $"cudnn_heuristic64_{cudnnMajorVersion}.dll",
        };
    }

    private bool CheckZlibAvailability(string cudnnBinPath)
    {
        const string zlibDll = "zlibwapi.dll";

        // Check in cuDNN directory
        if (File.Exists(Path.Combine(cudnnBinPath, zlibDll)))
            return true;

        // Check in CUDA bin directories
        foreach (var cuda in _cudaInstallations)
        {
            if (File.Exists(Path.Combine(cuda.BinPath, zlibDll)))
                return true;
        }

        // Check in system PATH
        var pathDirs = Environment.GetEnvironmentVariable("PATH")?.Split(Path.PathSeparator) ?? Array.Empty<string>();
        foreach (var dir in pathDirs)
        {
            if (!string.IsNullOrEmpty(dir) && File.Exists(Path.Combine(dir, zlibDll)))
                return true;
        }

        // Check System32
        var system32 = Environment.GetFolderPath(Environment.SpecialFolder.System);
        if (File.Exists(Path.Combine(system32, zlibDll)))
            return true;

        return false;
    }

    #endregion
}

/// <summary>
/// Represents a detected CUDA installation.
/// </summary>
public sealed class CudaInstallation
{
    public string Path { get; }
    public Version Version { get; }
    public int MajorVersion => Version.Major;
    public string BinPath => System.IO.Path.Combine(Path, "bin");

    public CudaInstallation(string path, Version version)
    {
        Path = path;
        Version = version;
    }

    public override string ToString() => $"CUDA {Version} at {Path}";
}

/// <summary>
/// Represents a detected cuDNN installation.
/// </summary>
public sealed class CuDnnInstallation
{
    public string BinPath { get; }
    public Version Version { get; }
    public int MajorVersion => Version.Major;
    public int CudaMajorVersion { get; }

    public CuDnnInstallation(string binPath, Version version, int cudaMajorVersion)
    {
        BinPath = binPath;
        Version = version;
        CudaMajorVersion = cudaMajorVersion;
    }

    public override string ToString() => $"cuDNN {Version} (CUDA {CudaMajorVersion}.x) at {BinPath}";
}

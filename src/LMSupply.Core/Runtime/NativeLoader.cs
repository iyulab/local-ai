using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;

namespace LMSupply.Runtime;

/// <summary>
/// Handles dynamic native library loading using AssemblyLoadContext.ResolvingUnmanagedDll.
/// This approach is cleaner than SetDllDirectory and works cross-platform.
/// Also supports Windows AddDllDirectory for native-to-native dependency resolution.
/// </summary>
public sealed class NativeLoader : IDisposable
{
    private static NativeLoader? _instance;
    private static readonly object _instanceLock = new();

    private readonly Dictionary<string, string> _libraryPaths = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, IntPtr> _loadedLibraries = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<Assembly> _registeredAssemblies = new();
    private readonly HashSet<string> _addedDllDirectories = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<IntPtr> _dllDirectoryCookies = new();
    private readonly object _lock = new();

    private bool _isRegistered;
    private bool _dllSearchPathModified;

    // Windows API for DLL search path modification
    // These are used to add directories to the DLL search path for native-to-native dependencies
    private const uint LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr AddDllDirectory(string newDirectory);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool RemoveDllDirectory(IntPtr cookie);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetDefaultDllDirectories(uint directoryFlags);

    /// <summary>
    /// Gets the singleton instance of the native loader.
    /// </summary>
    public static NativeLoader Instance
    {
        get
        {
            if (_instance is null)
            {
                lock (_instanceLock)
                {
                    _instance ??= new NativeLoader();
                }
            }
            return _instance;
        }
    }

    private NativeLoader()
    {
    }

    /// <summary>
    /// Registers a native library path for resolution.
    /// </summary>
    /// <param name="libraryName">The library name (without path, e.g., "onnxruntime").</param>
    /// <param name="libraryPath">The full path to the library file.</param>
    public void RegisterLibrary(string libraryName, string libraryPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(libraryName);
        ArgumentException.ThrowIfNullOrEmpty(libraryPath);

        lock (_lock)
        {
            // Normalize the library name
            var normalizedName = NormalizeLibraryName(libraryName);
            _libraryPaths[normalizedName] = libraryPath;

            // Also register common variations
            RegisterVariations(libraryName, libraryPath);

            EnsureRegistered();
        }
    }

    /// <summary>
    /// Registers a directory containing native libraries.
    /// All native libraries in the directory will be available for resolution.
    /// </summary>
    /// <param name="directory">The directory containing native libraries.</param>
    public void RegisterDirectory(string directory)
    {
        RegisterDirectory(directory, preload: false);
    }

    /// <summary>
    /// Registers a directory containing native libraries and optionally pre-loads them.
    /// Pre-loading ensures DLLs are available before any managed code tries to use them via DllImport.
    /// </summary>
    /// <param name="directory">The directory containing native libraries.</param>
    /// <param name="preload">If true, immediately loads all native libraries into memory.</param>
    /// <param name="primaryLibrary">Optional name of the primary library to load first (e.g., "onnxruntime").</param>
    public void RegisterDirectory(string directory, bool preload, string? primaryLibrary = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);

        if (!Directory.Exists(directory))
            return;

        lock (_lock)
        {
            // On Windows, add the directory to the DLL search path for native-to-native dependencies
            AddToWindowsDllSearchPath(directory);

            var extensions = GetNativeLibraryExtensions();
            var libraries = new List<(string name, string path)>();

            foreach (var file in Directory.EnumerateFiles(directory))
            {
                var extension = Path.GetExtension(file);
                if (extensions.Contains(extension, StringComparer.OrdinalIgnoreCase))
                {
                    var fileName = Path.GetFileName(file);
                    var libraryName = GetLibraryNameFromFileName(fileName);
                    RegisterLibrary(libraryName, file);
                    libraries.Add((libraryName, file));
                }
            }

            if (preload && libraries.Count > 0)
            {
                // Load primary library first if specified
                if (!string.IsNullOrEmpty(primaryLibrary))
                {
                    var primary = libraries.FirstOrDefault(l =>
                        l.name.Equals(primaryLibrary, StringComparison.OrdinalIgnoreCase) ||
                        l.name.Contains(primaryLibrary, StringComparison.OrdinalIgnoreCase));

                    if (primary.path is not null)
                    {
                        PreloadLibrary(primary.name, primary.path);
                    }
                }

                // Load all other libraries
                foreach (var (name, path) in libraries)
                {
                    PreloadLibrary(name, path);
                }
            }
        }
    }

    /// <summary>
    /// Adds a directory to the Windows DLL search path.
    /// This enables native DLLs to find their native dependencies in other registered directories.
    /// On non-Windows platforms, this method does nothing.
    /// </summary>
    /// <param name="directory">The directory to add to the DLL search path.</param>
    public void AddToWindowsDllSearchPath(string directory)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return;

        if (string.IsNullOrEmpty(directory) || !Directory.Exists(directory))
            return;

        var fullPath = Path.GetFullPath(directory);

        lock (_lock)
        {
            // Skip if already added
            if (_addedDllDirectories.Contains(fullPath))
                return;

            // First time: set up the search order to use application + system + user directories
            if (!_dllSearchPathModified)
            {
                try
                {
                    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
                    _dllSearchPathModified = true;
                }
                catch
                {
                    // Ignore - SetDefaultDllDirectories may not be available on older Windows
                }
            }

            // Add the directory to the DLL search path
            try
            {
                var cookie = AddDllDirectory(fullPath);
                if (cookie != IntPtr.Zero)
                {
                    _addedDllDirectories.Add(fullPath);
                    _dllDirectoryCookies.Add(cookie);
                }
            }
            catch
            {
                // Ignore - AddDllDirectory may fail in some environments
            }
        }
    }

    /// <summary>
    /// Pre-loads a native library into memory.
    /// </summary>
    private void PreloadLibrary(string libraryName, string libraryPath)
    {
        var normalizedName = NormalizeLibraryName(libraryName);

        // Skip if already loaded
        if (_loadedLibraries.ContainsKey(normalizedName))
            return;

        if (NativeLibrary.TryLoad(libraryPath, out var handle))
        {
            _loadedLibraries[normalizedName] = handle;
        }
    }

    /// <summary>
    /// Registers an assembly to use this native loader for unmanaged DLL resolution.
    /// </summary>
    /// <param name="assembly">The assembly to register.</param>
    public void RegisterAssembly(Assembly assembly)
    {
        ArgumentNullException.ThrowIfNull(assembly);

        lock (_lock)
        {
            if (_registeredAssemblies.Add(assembly))
            {
                EnsureRegistered();
            }
        }
    }

    /// <summary>
    /// Tries to load a native library by name.
    /// </summary>
    /// <param name="libraryName">The library name.</param>
    /// <param name="handle">The loaded library handle.</param>
    /// <returns>True if the library was loaded successfully.</returns>
    public bool TryLoad(string libraryName, out IntPtr handle)
    {
        handle = IntPtr.Zero;

        lock (_lock)
        {
            // Check if already loaded
            var normalizedName = NormalizeLibraryName(libraryName);
            if (_loadedLibraries.TryGetValue(normalizedName, out handle))
                return true;

            // Try to find and load the library
            if (_libraryPaths.TryGetValue(normalizedName, out var path))
            {
                if (NativeLibrary.TryLoad(path, out handle))
                {
                    _loadedLibraries[normalizedName] = handle;
                    return true;
                }
            }

            // Try variations
            foreach (var variation in GetLibraryNameVariations(libraryName))
            {
                if (_libraryPaths.TryGetValue(variation, out path))
                {
                    if (NativeLibrary.TryLoad(path, out handle))
                    {
                        _loadedLibraries[normalizedName] = handle;
                        return true;
                    }
                }
            }
        }

        return false;
    }

    /// <summary>
    /// Gets a function pointer from a loaded library.
    /// </summary>
    public bool TryGetExport(string libraryName, string functionName, out IntPtr address)
    {
        address = IntPtr.Zero;

        if (!TryLoad(libraryName, out var handle))
            return false;

        return NativeLibrary.TryGetExport(handle, functionName, out address);
    }

    /// <summary>
    /// Gets a delegate for a function in a loaded library.
    /// </summary>
    public T? GetFunction<T>(string libraryName, string functionName) where T : Delegate
    {
        if (!TryGetExport(libraryName, functionName, out var address))
            return null;

        return Marshal.GetDelegateForFunctionPointer<T>(address);
    }

    /// <summary>
    /// Checks if a library is registered.
    /// </summary>
    public bool IsRegistered(string libraryName)
    {
        lock (_lock)
        {
            var normalizedName = NormalizeLibraryName(libraryName);
            return _libraryPaths.ContainsKey(normalizedName);
        }
    }

    /// <summary>
    /// Gets all registered library names.
    /// </summary>
    public IEnumerable<string> GetRegisteredLibraries()
    {
        lock (_lock)
        {
            return _libraryPaths.Keys.ToList();
        }
    }

    private void EnsureRegistered()
    {
        if (_isRegistered)
            return;

        // Register the resolver with the default AssemblyLoadContext
        AssemblyLoadContext.Default.ResolvingUnmanagedDll += OnResolvingUnmanagedDll;
        _isRegistered = true;
    }

    private IntPtr OnResolvingUnmanagedDll(Assembly assembly, string libraryName)
    {
        // Only handle libraries we have registered
        if (TryLoad(libraryName, out var handle))
            return handle;

        // Return zero to let the default resolver handle it
        return IntPtr.Zero;
    }

    private void RegisterVariations(string libraryName, string libraryPath)
    {
        var variations = GetLibraryNameVariations(libraryName);
        foreach (var variation in variations)
        {
            _libraryPaths.TryAdd(variation, libraryPath);
        }
    }

    private static string NormalizeLibraryName(string name)
    {
        // Remove common prefixes and extensions
        var normalized = name;

        // Remove 'lib' prefix (Unix convention)
        if (normalized.StartsWith("lib", StringComparison.OrdinalIgnoreCase) && normalized.Length > 3)
        {
            normalized = normalized[3..];
        }

        // Remove extensions
        foreach (var ext in new[] { ".dll", ".so", ".dylib", ".so.1" })
        {
            if (normalized.EndsWith(ext, StringComparison.OrdinalIgnoreCase))
            {
                normalized = normalized[..^ext.Length];
                break;
            }
        }

        return normalized.ToLowerInvariant();
    }

    private static IEnumerable<string> GetLibraryNameVariations(string libraryName)
    {
        var baseName = NormalizeLibraryName(libraryName);
        yield return baseName;
        yield return $"lib{baseName}";
        yield return $"{baseName}.dll";
        yield return $"lib{baseName}.so";
        yield return $"lib{baseName}.so.1";
        yield return $"lib{baseName}.dylib";
    }

    private static string GetLibraryNameFromFileName(string fileName)
    {
        return NormalizeLibraryName(fileName);
    }

    private static string[] GetNativeLibraryExtensions()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return new[] { ".dll" };
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            return new[] { ".so", ".so.1" };
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return new[] { ".dylib" };

        return new[] { ".dll", ".so", ".dylib" };
    }

    public void Dispose()
    {
        lock (_lock)
        {
            if (_isRegistered)
            {
                AssemblyLoadContext.Default.ResolvingUnmanagedDll -= OnResolvingUnmanagedDll;
                _isRegistered = false;
            }

            // Free loaded libraries
            foreach (var handle in _loadedLibraries.Values)
            {
                if (handle != IntPtr.Zero)
                {
                    try
                    {
                        NativeLibrary.Free(handle);
                    }
                    catch
                    {
                        // Ignore unload errors
                    }
                }
            }

            // Remove added DLL directories on Windows
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                foreach (var cookie in _dllDirectoryCookies)
                {
                    if (cookie != IntPtr.Zero)
                    {
                        try
                        {
                            RemoveDllDirectory(cookie);
                        }
                        catch
                        {
                            // Ignore removal errors
                        }
                    }
                }
                _dllDirectoryCookies.Clear();
                _addedDllDirectories.Clear();
            }

            _loadedLibraries.Clear();
            _libraryPaths.Clear();
            _registeredAssemblies.Clear();
        }
    }
}

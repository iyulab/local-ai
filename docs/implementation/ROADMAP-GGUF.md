# LMSupply.Generator GGUF Support Roadmap

> llama.cpp 엔진을 통한 GGUF 형식 모델 지원 구현 계획

## Executive Summary

| Item | Decision | Rationale |
|------|----------|-----------|
| **Inference Engine** | LLamaSharp v0.25.0 (llama.cpp binding) | 가장 성숙한 .NET 바인딩, 활발한 유지보수 |
| **Native Binary Distribution** | On-demand download via RuntimeManager | LMSupply 철학: zero-bundled, lazy loading |
| **Default Models** | HuggingFace GGUF repositories | Bartowski, TheBloke 등 풍부한 양자화 모델 |
| **Quantization** | Q4_K_M (default), Q2_K ~ Q8_0 지원 | 메모리/품질 균형 |
| **GPU Support** | CUDA, Metal, Vulkan (on-demand) | 플랫폼별 네이티브 바이너리 분리 |

---

## Core Philosophy: On-Demand Everything

> **"초기 어셈블리에 포함하지 않음, 엔진/런타임/모델 모두 온디맨드 방식으로 lazy하게 구현"**

```
1. Assembly Size: 최소화 (네이티브 바이너리 0MB)
2. First Use: 런타임 바이너리 + 모델 자동 다운로드
3. Subsequent Uses: 캐시된 바이너리/모델 즉시 사용
4. Platform Detection: 자동으로 최적 백엔드 선택
```

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                      LMSupply.Generator                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    IGeneratorModel                          │ │
│  │  (공통 인터페이스: GenerateAsync, GenerateChatAsync, etc.)   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│           │                                    │                  │
│           ▼                                    ▼                  │
│  ┌─────────────────┐                 ┌─────────────────┐         │
│  │OnnxGeneratorModel│                │GgufGeneratorModel│         │
│  │  (기존 구현)     │                │   (NEW)          │         │
│  └────────┬────────┘                 └────────┬────────┘         │
│           │                                    │                  │
│           ▼                                    ▼                  │
│  ┌─────────────────┐                 ┌─────────────────┐         │
│  │ONNX Runtime GenAI│                │   LLamaSharp    │         │
│  │ (ExcludeAssets=  │                │ (ExcludeAssets= │         │
│  │    "native")     │                │    "native")    │         │
│  └────────┬────────┘                 └────────┬────────┘         │
└───────────┼────────────────────────────────────┼─────────────────┘
            │                                    │
            ▼                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                     RuntimeManager (LMSupply.Core)                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  • Platform Detection (Windows/Linux/macOS, x64/arm64)      │  │
│  │  • GPU Detection (CUDA/DirectML/Metal/Vulkan)               │  │
│  │  • Native Binary Download (GitHub Releases)                  │  │
│  │  • Version Management & Caching                              │  │
│  │  • DLL Search Path Configuration                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
            │                                    │
            ▼                                    ▼
┌─────────────────────┐              ┌─────────────────────┐
│  ONNX Runtime DLLs  │              │  llama.cpp DLLs     │
│  (Cached on disk)   │              │  (Cached on disk)   │
└─────────────────────┘              └─────────────────────┘
```

---

## Technical Research Summary

### 1. LLamaSharp Analysis

| Aspect | Finding |
|--------|---------|
| **Current Version** | v0.25.0 (August 2025) |
| **llama.cpp Commit** | `11dd5a44eb180e1d69fac24d3852b5222d66fb7f` |
| **Supported Models** | Llama 3.x, Qwen 3, Gemma 3, DeepSeek R1, Phi-4, Mistral 등 |
| **Backend Packages** | CPU, CUDA11, CUDA12, Vulkan, Metal (플랫폼별 분리) |
| **Native Config** | `NativeLibraryConfig.Instance.WithLibrary()` 지원 |

**Key APIs:**
```csharp
// 모델 로딩
using var model = await LLamaWeights.LoadFromFileAsync(new ModelParams(modelPath));

// 추론 실행기
var executor = new StatelessExecutor(model, parameters);
await foreach (var token in executor.InferAsync(prompt)) { ... }

// 또는 InteractiveExecutor for chat
var executor = new InteractiveExecutor(context);
```

### 2. llama.cpp Native Binary Distribution

| Platform | Backend | Release Asset Pattern |
|----------|---------|----------------------|
| Windows x64 | CPU | `llama-b{version}-bin-win-x64.zip` |
| Windows x64 | CUDA 12.4 | `llama-b{version}-bin-win-cuda-12.4-x64.zip` |
| Windows x64 | CUDA 13.1 | `llama-b{version}-bin-win-cuda-13.1-x64.zip` |
| Windows x64 | Vulkan | `llama-b{version}-bin-win-vulkan-x64.zip` |
| Linux x64 | CPU | `llama-b{version}-bin-linux-x64.tar.gz` |
| Linux x64 | Vulkan | `llama-b{version}-bin-linux-vulkan-x64.tar.gz` |
| macOS x64 | CPU | `llama-b{version}-bin-macos-x64.zip` |
| macOS arm64 | Metal | `llama-b{version}-bin-macos-arm64.zip` |

**CUDA Runtime:** 별도 다운로드 필요 (`cudart-llama-bin-win-cuda-{version}-x64.zip`)

### 3. GGUF Model Download Patterns

```csharp
// HuggingFace에서 단일 GGUF 파일 다운로드
// 기존 HuggingFaceDownloader 확장 필요

var modelPath = await downloader.DownloadFileAsync(
    repoId: "bartowski/Llama-3.2-3B-Instruct-GGUF",
    filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    progress: progress,
    cancellationToken: ct);
```

---

## Implementation Phases

### Phase 0: Architecture & Interface Design (Foundation)

> 기존 코드와의 호환성 보장, 인터페이스 추상화

#### Task 0.1: IGeneratorBackend 인터페이스 정의

```csharp
/// <summary>
/// Generator 백엔드 추상화 - ONNX와 GGUF 공통 인터페이스
/// </summary>
public interface IGeneratorBackend : IAsyncDisposable
{
    /// <summary>백엔드 유형 (Onnx, Gguf)</summary>
    GeneratorBackendType BackendType { get; }

    /// <summary>모델 정보</summary>
    GeneratorModelInfo ModelInfo { get; }

    /// <summary>스트리밍 텍스트 생성</summary>
    IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>모델 워밍업 (첫 추론 전 초기화)</summary>
    Task WarmupAsync(CancellationToken cancellationToken = default);
}

public enum GeneratorBackendType
{
    Onnx,   // ONNX Runtime GenAI
    Gguf    // llama.cpp via LLamaSharp
}
```

#### Task 0.2: ModelFormat 자동 감지

```csharp
public static class ModelFormatDetector
{
    public static ModelFormat Detect(string modelIdOrPath)
    {
        // 1. 파일 확장자로 판단
        if (modelIdOrPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            return ModelFormat.Gguf;

        if (modelIdOrPath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            return ModelFormat.Onnx;

        // 2. HuggingFace repo ID인 경우 API로 파일 목록 확인
        if (IsHuggingFaceRepoId(modelIdOrPath))
        {
            // "GGUF" in repo name or contains .gguf files
            return DetectFromHuggingFace(modelIdOrPath);
        }

        // 3. 로컬 디렉토리인 경우 파일 존재 확인
        if (Directory.Exists(modelIdOrPath))
        {
            if (File.Exists(Path.Combine(modelIdOrPath, "model.onnx")))
                return ModelFormat.Onnx;

            var ggufFiles = Directory.GetFiles(modelIdOrPath, "*.gguf");
            if (ggufFiles.Length > 0)
                return ModelFormat.Gguf;
        }

        // 4. 기본값: 기존 레지스트리 확인 후 ONNX
        return ModelFormat.Onnx;
    }
}
```

#### Task 0.3: LocalGenerator 팩토리 확장

```csharp
public static class LocalGenerator
{
    public static async Task<IGeneratorModel> LoadAsync(
        string modelIdOrPath,
        GeneratorOptions? options = null,
        IProgress<ModelLoadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var format = ModelFormatDetector.Detect(modelIdOrPath);

        return format switch
        {
            ModelFormat.Onnx => await LoadOnnxAsync(modelIdOrPath, options, progress, ct),
            ModelFormat.Gguf => await LoadGgufAsync(modelIdOrPath, options, progress, ct),
            _ => throw new NotSupportedException($"Unknown model format")
        };
    }
}
```

#### Deliverables Phase 0
- [ ] `IGeneratorBackend` 인터페이스 정의
- [ ] `ModelFormatDetector` 구현
- [ ] `LocalGenerator` 팩토리 확장 설계
- [ ] 기존 `OnnxGeneratorModel`을 `IGeneratorBackend` 구현으로 리팩토링

---

### Phase 1: LLamaSharp Integration & Native Binary Management

> 핵심: 네이티브 바이너리 온디맨드 다운로드

#### Task 1.1: Package Reference 설정

```xml
<!-- Directory.Packages.props -->
<PackageVersion Include="LLamaSharp" Version="0.25.0" />
<!-- 백엔드 패키지는 참조하지 않음 - RuntimeManager가 처리 -->
```

```xml
<!-- LMSupply.Generator.csproj -->
<ItemGroup>
  <!-- LLamaSharp managed wrapper only, no native binaries -->
  <PackageReference Include="LLamaSharp" ExcludeAssets="native" />
</ItemGroup>
```

#### Task 1.2: LlamaRuntimeManager 구현

```csharp
/// <summary>
/// llama.cpp 네이티브 바이너리 온디맨드 다운로드 및 로딩 관리
/// </summary>
public sealed class LlamaRuntimeManager
{
    private static readonly Lazy<LlamaRuntimeManager> _instance = new(() => new());
    public static LlamaRuntimeManager Instance => _instance.Value;

    private bool _initialized;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    /// <summary>
    /// 런타임 초기화 - 네이티브 바이너리 다운로드 및 로드
    /// </summary>
    public async Task EnsureInitializedAsync(
        ExecutionProvider provider = ExecutionProvider.Auto,
        IProgress<RuntimeDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (_initialized) return;

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (_initialized) return;

            // 1. 플랫폼 및 GPU 감지
            var platform = EnvironmentDetector.DetectPlatform();
            var gpu = EnvironmentDetector.DetectGpu();

            // 2. 최적 백엔드 결정
            var backend = DetermineBackend(provider, platform, gpu);

            // 3. 네이티브 바이너리 다운로드 (없으면)
            var binaryPath = await DownloadNativeBinaryAsync(
                backend, platform, progress, cancellationToken);

            // 4. CUDA runtime 다운로드 (필요시)
            if (backend is LlamaBackend.Cuda12 or LlamaBackend.Cuda13)
            {
                await DownloadCudaRuntimeAsync(backend, progress, cancellationToken);
            }

            // 5. NativeLibraryConfig 설정
            ConfigureNativeLibrary(binaryPath, backend);

            _initialized = true;
        }
        finally
        {
            _initLock.Release();
        }
    }

    private void ConfigureNativeLibrary(string binaryPath, LlamaBackend backend)
    {
        // LLamaSharp의 NativeLibraryConfig 사용
        NativeLibraryConfig.All.WithSearchPath(binaryPath);
        NativeLibraryConfig.All.WithAutoFallback(true);

        // 백엔드별 설정
        switch (backend)
        {
            case LlamaBackend.Cuda12:
            case LlamaBackend.Cuda13:
                NativeLibraryConfig.All.WithCuda(true);
                break;
            case LlamaBackend.Vulkan:
                NativeLibraryConfig.All.WithVulkan(true);
                break;
            case LlamaBackend.Metal:
                // macOS Metal은 기본 활성화
                break;
        }

        NativeLibraryConfig.All.WithLogs(LLamaLogLevel.Warning);
    }
}

public enum LlamaBackend
{
    Cpu,
    Cuda12,
    Cuda13,
    Vulkan,
    Metal
}
```

#### Task 1.3: GitHub Releases Downloader

```csharp
public sealed class LlamaBinaryDownloader
{
    private const string GithubApiBase = "https://api.github.com/repos/ggml-org/llama.cpp/releases";
    private const string GithubDownloadBase = "https://github.com/ggml-org/llama.cpp/releases/download";

    /// <summary>
    /// 최신 호환 버전의 llama.cpp 바이너리 다운로드
    /// </summary>
    public async Task<string> DownloadAsync(
        LlamaBackend backend,
        PlatformInfo platform,
        string? targetVersion = null,
        IProgress<RuntimeDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // 1. 타겟 버전 결정 (LLamaSharp 호환 버전)
        targetVersion ??= GetCompatibleVersion();

        // 2. 에셋 이름 결정
        var assetName = GetAssetName(backend, platform, targetVersion);

        // 3. 캐시 확인
        var cachePath = GetCachePath(targetVersion, backend, platform);
        if (Directory.Exists(cachePath) && IsValid(cachePath))
        {
            return cachePath;
        }

        // 4. 다운로드
        var downloadUrl = $"{GithubDownloadBase}/{targetVersion}/{assetName}";
        var tempPath = await DownloadAndExtractAsync(downloadUrl, progress, cancellationToken);

        // 5. 캐시로 이동
        MoveToCache(tempPath, cachePath);

        return cachePath;
    }

    private string GetAssetName(LlamaBackend backend, PlatformInfo platform, string version)
    {
        var rid = GetRuntimeIdentifier(platform);

        return backend switch
        {
            LlamaBackend.Cpu when platform.IsWindows => $"llama-{version}-bin-win-x64.zip",
            LlamaBackend.Cpu when platform.IsLinux => $"llama-{version}-bin-linux-x64.tar.gz",
            LlamaBackend.Cpu when platform.IsMacOS => $"llama-{version}-bin-macos-{GetMacArch(platform)}.zip",
            LlamaBackend.Cuda12 => $"llama-{version}-bin-win-cuda-12.4-x64.zip",
            LlamaBackend.Cuda13 => $"llama-{version}-bin-win-cuda-13.1-x64.zip",
            LlamaBackend.Vulkan when platform.IsWindows => $"llama-{version}-bin-win-vulkan-x64.zip",
            LlamaBackend.Vulkan when platform.IsLinux => $"llama-{version}-bin-linux-vulkan-x64.tar.gz",
            LlamaBackend.Metal => $"llama-{version}-bin-macos-arm64.zip",
            _ => throw new PlatformNotSupportedException($"Backend {backend} not supported on {platform}")
        };
    }

    /// <summary>
    /// LLamaSharp 0.25.0과 호환되는 llama.cpp 버전
    /// </summary>
    private string GetCompatibleVersion() => "b7766";
}
```

#### Task 1.4: CUDA Runtime Downloader

```csharp
public async Task DownloadCudaRuntimeAsync(
    LlamaBackend backend,
    IProgress<RuntimeDownloadProgress>? progress,
    CancellationToken cancellationToken)
{
    // CUDA DLL이 시스템에 없는 경우 다운로드
    var cudaVersion = backend == LlamaBackend.Cuda12 ? "12.4" : "13.1";

    if (IsCudaRuntimeAvailable(cudaVersion))
        return;

    var assetName = $"cudart-llama-bin-win-cuda-{cudaVersion}-x64.zip";
    // ... 다운로드 로직
}
```

#### Deliverables Phase 1
- [ ] LLamaSharp 패키지 참조 (ExcludeAssets="native")
- [ ] `LlamaRuntimeManager` 싱글톤 구현
- [ ] `LlamaBinaryDownloader` - GitHub releases 다운로드
- [ ] CUDA runtime 별도 다운로드 지원
- [ ] `NativeLibraryConfig` 자동 설정

---

### Phase 2: GGUF Model Download & Registry

> HuggingFace에서 GGUF 모델 다운로드 지원

#### Task 2.1: GgufModelDownloader 구현

```csharp
/// <summary>
/// HuggingFace에서 GGUF 파일 다운로드
/// </summary>
public sealed class GgufModelDownloader
{
    private readonly HuggingFaceClient _client;
    private readonly string _cacheDirectory;

    public async Task<string> DownloadAsync(
        string repoId,
        string? filename = null,
        string? quantization = null,
        IProgress<ModelLoadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // 1. 파일명 결정
        if (filename == null)
        {
            filename = await SelectBestGgufFileAsync(repoId, quantization, cancellationToken);
        }

        // 2. 캐시 경로 확인
        var cachedPath = GetCachedPath(repoId, filename);
        if (File.Exists(cachedPath))
        {
            progress?.Report(new ModelLoadProgress(1.0, "Using cached model"));
            return cachedPath;
        }

        // 3. 다운로드
        progress?.Report(new ModelLoadProgress(0, $"Downloading {filename}..."));

        await _client.DownloadFileAsync(
            repoId,
            filename,
            cachedPath,
            new Progress<double>(p => progress?.Report(new ModelLoadProgress(p, "Downloading..."))),
            cancellationToken);

        return cachedPath;
    }

    /// <summary>
    /// 양자화 선호도에 따라 최적 GGUF 파일 선택
    /// </summary>
    private async Task<string> SelectBestGgufFileAsync(
        string repoId,
        string? preferredQuantization,
        CancellationToken cancellationToken)
    {
        var files = await _client.ListFilesAsync(repoId, cancellationToken);
        var ggufFiles = files.Where(f => f.EndsWith(".gguf")).ToList();

        if (ggufFiles.Count == 0)
            throw new ModelNotFoundException($"No GGUF files found in {repoId}");

        // 양자화 우선순위: 명시적 지정 > Q4_K_M > Q4_K_S > Q5_K_M > 첫 번째
        var quantPriority = new[]
        {
            preferredQuantization ?? "Q4_K_M",
            "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S",
            "Q6_K", "Q8_0", "Q3_K_M", "Q2_K"
        };

        foreach (var quant in quantPriority)
        {
            var match = ggufFiles.FirstOrDefault(f =>
                f.Contains(quant, StringComparison.OrdinalIgnoreCase));
            if (match != null) return match;
        }

        return ggufFiles.First();
    }
}
```

#### Task 2.2: GGUF Model Registry

```csharp
public static class GgufModelRegistry
{
    private static readonly Dictionary<string, GgufModelInfo> _models = new()
    {
        // Default: 균형 잡힌 모델
        ["default"] = new GgufModelInfo
        {
            RepoId = "bartowski/Llama-3.2-3B-Instruct-GGUF",
            DefaultFile = "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            ChatFormat = "llama3",
            ContextLength = 8192,
            License = LicenseTier.Conditional
        },

        // Fast: 빠른 추론용 소형 모델
        ["fast"] = new GgufModelInfo
        {
            RepoId = "bartowski/Llama-3.2-1B-Instruct-GGUF",
            DefaultFile = "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            ChatFormat = "llama3",
            ContextLength = 8192,
            License = LicenseTier.Conditional
        },

        // Quality: 고품질 모델
        ["quality"] = new GgufModelInfo
        {
            RepoId = "bartowski/Qwen2.5-7B-Instruct-GGUF",
            DefaultFile = "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            ChatFormat = "chatml",
            ContextLength = 32768,
            License = LicenseTier.Conditional
        },

        // Multilingual: 다국어 지원
        ["multilingual"] = new GgufModelInfo
        {
            RepoId = "bartowski/gemma-2-9b-it-GGUF",
            DefaultFile = "gemma-2-9b-it-Q4_K_M.gguf",
            ChatFormat = "gemma",
            ContextLength = 8192,
            License = LicenseTier.Conditional
        },

        // Korean: 한국어 특화
        ["korean"] = new GgufModelInfo
        {
            RepoId = "bartowski/EXAONE-3.5-7.8B-Instruct-GGUF",
            DefaultFile = "EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf",
            ChatFormat = "exaone",
            ContextLength = 32768,
            License = LicenseTier.Conditional
        }
    };

    public static GgufModelInfo? Resolve(string alias)
    {
        return _models.TryGetValue(alias.ToLowerInvariant(), out var info) ? info : null;
    }
}

public sealed record GgufModelInfo
{
    public required string RepoId { get; init; }
    public required string DefaultFile { get; init; }
    public required string ChatFormat { get; init; }
    public required int ContextLength { get; init; }
    public required LicenseTier License { get; init; }
    public string? Subfolder { get; init; }
}
```

#### Deliverables Phase 2
- [ ] `GgufModelDownloader` 구현
- [ ] HuggingFace API 파일 목록 조회
- [ ] 양자화 자동 선택 로직
- [ ] `GgufModelRegistry` 별칭 등록
- [ ] 다운로드 진행률 보고

---

### Phase 3: Core GGUF Inference Engine

> GgufGeneratorModel 핵심 구현

#### Task 3.1: GgufGeneratorModel 구현

```csharp
/// <summary>
/// GGUF 모델을 위한 Generator 구현 (llama.cpp via LLamaSharp)
/// </summary>
public sealed class GgufGeneratorModel : IGeneratorModel
{
    private readonly LLamaWeights _weights;
    private readonly LLamaContext _context;
    private readonly InteractiveExecutor _executor;
    private readonly IChatFormatter _chatFormatter;
    private readonly GeneratorModelInfo _modelInfo;
    private readonly SemaphoreSlim _inferenceLock = new(1, 1);

    private bool _disposed;

    private GgufGeneratorModel(
        LLamaWeights weights,
        LLamaContext context,
        InteractiveExecutor executor,
        IChatFormatter chatFormatter,
        GeneratorModelInfo modelInfo)
    {
        _weights = weights;
        _context = context;
        _executor = executor;
        _chatFormatter = chatFormatter;
        _modelInfo = modelInfo;
    }

    /// <summary>
    /// GGUF 모델 로드 (런타임 초기화 포함)
    /// </summary>
    public static async Task<GgufGeneratorModel> LoadAsync(
        string modelPath,
        GeneratorOptions? options = null,
        IProgress<ModelLoadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new GeneratorOptions();

        // 1. llama.cpp 런타임 초기화 (온디맨드)
        progress?.Report(new ModelLoadProgress(0.1, "Initializing runtime..."));
        await LlamaRuntimeManager.Instance.EnsureInitializedAsync(
            options.Provider,
            new Progress<RuntimeDownloadProgress>(p =>
                progress?.Report(new ModelLoadProgress(0.1 + p.Progress * 0.3, p.Status))),
            cancellationToken);

        // 2. 모델 파라미터 설정
        progress?.Report(new ModelLoadProgress(0.4, "Loading model..."));
        var modelParams = new ModelParams(modelPath)
        {
            ContextSize = (uint)(options.MaxContextLength ?? 4096),
            GpuLayerCount = GetGpuLayerCount(options.Provider),
            BatchSize = 512,
            Threads = Environment.ProcessorCount
        };

        // 3. 모델 로드
        var weights = await LLamaWeights.LoadFromFileAsync(modelParams, cancellationToken);

        progress?.Report(new ModelLoadProgress(0.8, "Creating context..."));
        var context = weights.CreateContext(modelParams);
        var executor = new InteractiveExecutor(context);

        // 4. Chat formatter 결정
        var chatFormat = DetectChatFormat(modelPath, options.ChatFormat);
        var chatFormatter = ChatFormatterFactory.Create(chatFormat);

        // 5. 모델 정보 생성
        var modelInfo = new GeneratorModelInfo
        {
            ModelId = Path.GetFileNameWithoutExtension(modelPath),
            BackendType = GeneratorBackendType.Gguf,
            ContextLength = (int)modelParams.ContextSize,
            // GGUF 메타데이터에서 추가 정보 추출 가능
        };

        progress?.Report(new ModelLoadProgress(1.0, "Model loaded"));

        return new GgufGeneratorModel(weights, context, executor, chatFormatter, modelInfo);
    }

    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new GenerationOptions();

        await _inferenceLock.WaitAsync(cancellationToken);
        try
        {
            var inferenceParams = new InferenceParams
            {
                MaxTokens = options.MaxTokens,
                Temperature = options.Temperature,
                TopP = options.TopP,
                TopK = options.TopK,
                RepeatPenalty = options.RepetitionPenalty,
                AntiPrompts = options.StopSequences?.ToList()
            };

            await foreach (var token in _executor.InferAsync(prompt, inferenceParams, cancellationToken))
            {
                yield return token;
            }
        }
        finally
        {
            _inferenceLock.Release();
        }
    }

    public async IAsyncEnumerable<string> GenerateChatAsync(
        IReadOnlyList<ChatMessage> messages,
        GenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var prompt = _chatFormatter.Format(messages);

        // Stop sequences 병합
        options ??= new GenerationOptions();
        var stopSequences = (options.StopSequences?.ToList() ?? new List<string>());
        stopSequences.AddRange(_chatFormatter.GetStopSequences());
        options = options with { StopSequences = stopSequences.Distinct().ToList() };

        await foreach (var token in GenerateAsync(prompt, options, cancellationToken))
        {
            yield return token;
        }
    }

    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        // 간단한 추론으로 워밍업
        await foreach (var _ in GenerateAsync("Hello", new GenerationOptions { MaxTokens = 1 }, cancellationToken))
        {
            break;
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        _executor.Dispose();
        _context.Dispose();
        _weights.Dispose();
        _inferenceLock.Dispose();
    }

    private static int GetGpuLayerCount(ExecutionProvider provider)
    {
        return provider switch
        {
            ExecutionProvider.Cpu => 0,
            ExecutionProvider.Auto => -1,  // 모든 레이어 GPU로
            _ => -1
        };
    }
}
```

#### Task 3.2: GPU Layer 자동 계산

```csharp
public static class GpuLayerCalculator
{
    /// <summary>
    /// GPU VRAM에 따라 오프로드할 레이어 수 계산
    /// </summary>
    public static int CalculateGpuLayers(long vramBytes, long modelSizeBytes, int totalLayers)
    {
        if (vramBytes <= 0) return 0;

        // 모델 크기의 80%를 VRAM에 로드할 수 있으면 전체 레이어
        if (vramBytes > modelSizeBytes * 0.8)
            return totalLayers;

        // 비례 계산
        var ratio = (double)vramBytes / modelSizeBytes;
        var layers = (int)(totalLayers * ratio * 0.8);

        return Math.Max(0, Math.Min(layers, totalLayers));
    }
}
```

#### Deliverables Phase 3
- [ ] `GgufGeneratorModel` 구현
- [ ] 스트리밍 생성 지원
- [ ] Chat 생성 지원
- [ ] GPU 레이어 자동 계산
- [ ] 리소스 정리 (IAsyncDisposable)

---

### Phase 4: Chat Formatter Integration

> 기존 ChatFormatter 확장 및 GGUF 특화 포맷 추가

#### Task 4.1: GGUF 모델별 Chat Format 매핑

```csharp
public static class GgufChatFormatDetector
{
    /// <summary>
    /// GGUF 파일명에서 모델 패밀리 추론
    /// </summary>
    public static string DetectFromFilename(string filename)
    {
        var lower = filename.ToLowerInvariant();

        return lower switch
        {
            var f when f.Contains("llama") => "llama3",
            var f when f.Contains("qwen") => "chatml",
            var f when f.Contains("mistral") => "mistral",
            var f when f.Contains("gemma") => "gemma",
            var f when f.Contains("phi") => "phi3",
            var f when f.Contains("exaone") => "exaone",
            var f when f.Contains("deepseek") => "deepseek",
            _ => "chatml"  // 기본값
        };
    }
}
```

#### Task 4.2: EXAONE Chat Formatter (한국어 모델)

```csharp
public sealed class ExaoneChatFormatter : IChatFormatter
{
    public string ModelFamily => "exaone";

    public string Format(IReadOnlyList<ChatMessage> messages)
    {
        var sb = new StringBuilder();

        foreach (var msg in messages)
        {
            var role = msg.Role switch
            {
                ChatRole.System => "system",
                ChatRole.User => "user",
                ChatRole.Assistant => "assistant",
                _ => throw new ArgumentOutOfRangeException()
            };

            sb.Append($"[|{role}|]");
            sb.Append(msg.Content);
            sb.Append("[|endofturn|]");
        }

        sb.Append("[|assistant|]");
        return sb.ToString();
    }

    public IReadOnlyList<string> GetStopSequences() =>
        new[] { "[|endofturn|]", "[|user|]" };
}
```

#### Deliverables Phase 4
- [ ] `GgufChatFormatDetector` 구현
- [ ] EXAONE, DeepSeek 등 추가 포맷터
- [ ] 기존 `ChatFormatterFactory` 확장
- [ ] GGUF 메타데이터에서 chat_template 읽기 (가능시)

---

### Phase 5: Testing & Documentation

#### Task 5.1: 테스트 구조

```
tests/
└── LMSupply.Generator.Tests/
    ├── Gguf/
    │   ├── LlamaRuntimeManagerTests.cs
    │   ├── GgufModelDownloaderTests.cs
    │   ├── GgufGeneratorModelTests.cs
    │   └── Integration/
    │       └── GgufEndToEndTests.cs
    └── Shared/
        └── ModelFormatDetectorTests.cs
```

#### Task 5.2: Unit Tests

```csharp
public class LlamaRuntimeManagerTests
{
    [Fact]
    public async Task EnsureInitialized_DownloadsCorrectBinaryForPlatform()
    {
        // Arrange
        var manager = LlamaRuntimeManager.Instance;

        // Act
        await manager.EnsureInitializedAsync(ExecutionProvider.Cpu);

        // Assert
        // 네이티브 라이브러리 로드 가능해야 함
        Assert.True(NativeLibraryConfig.All.IsInitialized);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public async Task EnsureInitialized_Cuda_RequiresNvidiaGpu()
    {
        Skip.IfNot(GpuDetector.HasNvidiaGpu(), "NVIDIA GPU required");

        var manager = LlamaRuntimeManager.Instance;
        await manager.EnsureInitializedAsync(ExecutionProvider.Cuda);

        Assert.True(NativeLibraryConfig.All.IsInitialized);
    }
}

public class GgufGeneratorModelTests
{
    [SkippableFact]
    [Trait("Category", "Integration")]
    public async Task LoadAsync_DownloadsAndLoadsModel()
    {
        // This test downloads a real model - skip in CI
        Skip.If(Environment.GetEnvironmentVariable("CI") != null);

        await using var model = await GgufGeneratorModel.LoadAsync(
            "bartowski/Llama-3.2-1B-Instruct-GGUF");

        var response = new StringBuilder();
        await foreach (var token in model.GenerateAsync("Hello!",
            new GenerationOptions { MaxTokens = 10 }))
        {
            response.Append(token);
        }

        Assert.NotEmpty(response.ToString());
    }
}
```

#### Task 5.3: Documentation

```markdown
<!-- docs/generator-gguf.md -->

# GGUF Model Support

LMSupply.Generator supports GGUF format models via llama.cpp, enabling access to
thousands of quantized models on HuggingFace.

## Quick Start

```csharp
// Load a GGUF model - runtime is downloaded automatically
await using var model = await LocalGenerator.LoadAsync("bartowski/Llama-3.2-3B-Instruct-GGUF");

// Or use an alias
await using var model = await LocalGenerator.LoadAsync("default"); // Uses registry

// Generate text
await foreach (var token in model.GenerateAsync("Explain quantum computing:"))
{
    Console.Write(token);
}
```

## Model Aliases (GGUF)

| Alias | Model | Size | Best For |
|-------|-------|------|----------|
| `default` | Llama-3.2-3B-Instruct | ~2GB | Balanced |
| `fast` | Llama-3.2-1B-Instruct | ~700MB | Speed |
| `quality` | Qwen2.5-7B-Instruct | ~4GB | Quality |
| `korean` | EXAONE-3.5-7.8B | ~4.5GB | Korean |

## GPU Acceleration

GPU support is automatic based on your hardware:

- **NVIDIA GPU**: CUDA backend (requires CUDA toolkit or auto-downloaded DLLs)
- **AMD GPU**: Vulkan backend
- **Apple Silicon**: Metal backend (native performance)
- **CPU**: AVX2/AVX512 optimized

## Quantization Options

GGUF supports various quantization levels. Specify with `Quantization` option:

```csharp
var options = new GeneratorOptions
{
    Quantization = "Q4_K_M" // Default, good balance
    // Q2_K: Smallest, lower quality
    // Q8_0: Largest, best quality
};
```
```

#### Deliverables Phase 5
- [ ] Unit tests for all new components
- [ ] Integration tests (skippable)
- [ ] `docs/generator-gguf.md` 문서
- [ ] README.md 업데이트
- [ ] 예제 코드

---

## Task Summary

### Phase 0: Architecture ✅ COMPLETED
| # | Task | Status | Notes |
|---|------|--------|-------|
| 0.1 | `ModelFormat`/`GeneratorBackendType` enum 정의 | ✅ Done | `Models/ModelFormat.cs` |
| 0.2 | `ModelFormatDetector` 구현 | ✅ Done | 파일확장자, 디렉토리, repoId 패턴, known providers |
| 0.3 | `GeneratorModelLoader` 팩토리 확장 | ✅ Done | 형식 감지 및 GGUF/ONNX 라우팅 |
| 0.4 | Unit tests 작성 | ✅ Done | `ModelFormatDetectorTests.cs` (14+ cases) |
| - | 모든 기존 테스트 통과 | ✅ Done | 164 tests passed |

### Phase 1: Runtime Management ✅ INFRASTRUCTURE COMPLETE
| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | Package reference (LLamaSharp v0.25.0, ExcludeAssets) | ✅ Done | `Directory.Packages.props` |
| 1.2 | `LlamaRuntimeManager` 싱글톤 | ✅ Done | Backend detection, NativeLibraryConfig |
| 1.3 | `LlamaBinaryDownloader` (GitHub releases) | ✅ Done | Win/Linux/macOS, caching |
| 1.4 | `LlamaBackend` enum | ✅ Done | Cpu, Cuda12/13, Vulkan, Metal, Rocm |
| 1.5 | `NativeLibraryConfig` 자동 설정 | ✅ Done | WithSearchDirectory, WithCuda/Vulkan |
| 1.6 | 플랫폼별 테스트 | ⏳ Pending | Phase 3 통합 테스트에서 검증 |

### Phase 2: Model Download ✅ COMPLETED
| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.1 | `GgufModelInfo` 레코드 정의 | ✅ Done | 메타데이터, 라이선스 정보 포함 |
| 2.2 | `GgufModelRegistry` 별칭 등록 | ✅ Done | 8개 모델 (default, fast, quality 등) |
| 2.3 | `GgufModelDownloader` 구현 | ✅ Done | HuggingFace API, 자동 양자화 선택 |
| 2.4 | HuggingFace API 파일 목록 조회 | ✅ Done | `/api/models/{repoId}` endpoint |
| 2.5 | 다운로드 진행률 & 이력서 지원 | ✅ Done | Range header, `.part` 파일 |
| 2.6 | Unit tests | ✅ Done | `GgufModelRegistryTests.cs` (23 tests) |

### Phase 3: Inference Engine ✅ COMPLETED
| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.1 | `GgufGeneratorModel` 구현 | ✅ Done | LLamaSharp StatelessExecutor 사용 |
| 3.2 | 스트리밍 생성 | ✅ Done | IAsyncEnumerable<string> 지원 |
| 3.3 | Chat 생성 | ✅ Done | GenerateChatAsync, ChatFormatter 통합 |
| 3.4 | GPU 레이어 계산 | ✅ Done | CalculateGpuLayers helper |
| 3.5 | 리소스 정리 | ✅ Done | IAsyncDisposable 구현 |
| 3.6 | GeneratorModelLoader GGUF 라우팅 | ✅ Done | LoadGgufAsync, LoadGgufFromPathAsync |

### Phase 4: Chat Formatters ✅ COMPLETED
| # | Task | Status | Notes |
|---|------|--------|-------|
| 4.1 | `GgufChatFormatDetector` | ✅ Done | 파일명 기반 자동 감지 |
| 4.2 | EXAONE/DeepSeek 포맷터 | ✅ Done | + Gemma, Mistral 추가 |
| 4.3 | `ChatFormatterFactory` 확장 | ✅ Done | CreateByFormat에 8개 포맷 추가 |
| 4.4 | 메타데이터 chat_template 읽기 | ⏳ Deferred | 향후 GGUF 메타데이터 파싱 시 |

### Phase 5: Testing & Docs ✅ COMPLETED
| # | Task | Status | Notes |
|---|------|--------|-------|
| 5.1 | Integration tests | ✅ Done | `GgufIntegrationTests.cs`, `GgufModelFormatTests` |
| 5.2 | `docs/generator.md` GGUF 섹션 | ✅ Done | GGUF 사용법, 별칭 테이블, 설정 옵션 |
| 5.3 | README 업데이트 | ✅ Done | GGUF 예제, Generator 테이블 분리 |
| 5.4 | 예제 코드 | ✅ Done | `samples/GeneratorGgufSample` |

---

## Risk Mitigation

### 1. LLamaSharp 버전 호환성
- **Risk**: llama.cpp 빠른 업데이트로 바인딩 깨짐
- **Mitigation**: 특정 LLamaSharp 버전에 대응하는 llama.cpp 버전 고정

### 2. 네이티브 바이너리 다운로드 실패
- **Risk**: GitHub rate limit, 네트워크 오류
- **Mitigation**: 재시도 로직, 대체 미러, 오프라인 바이너리 지원

### 3. CUDA 버전 충돌
- **Risk**: 시스템 CUDA와 다운로드된 CUDA 충돌
- **Mitigation**: 격리된 DLL 디렉토리, 버전 감지 로직

### 4. 메모리 부족
- **Risk**: 대형 모델 로드 시 OOM
- **Mitigation**: GPU 레이어 자동 조절, 메모리 사전 검사

---

## Success Criteria

### Phase 0 ✅ COMPLETED
- [x] 기존 ONNX 테스트 모두 통과 (164 tests passed)
- [x] `ModelFormat` 및 `GeneratorBackendType` enum 정의 완료
- [x] `ModelFormatDetector` 구현 완료
- [x] `GeneratorModelLoader` 형식 감지 및 라우팅 추가

### Phase 1 ✅ IN PROGRESS (Infrastructure Complete)
- [x] LLamaSharp v0.25.0 패키지 참조 추가 (ExcludeAssets="native")
- [x] LlamaBackend enum 정의 (Cpu, Cuda12, Cuda13, Vulkan, Metal, Rocm)
- [x] LlamaRuntimeManager 싱글톤 구현
- [x] LlamaBinaryDownloader (GitHub releases) 구현
- [x] NativeLibraryConfig 자동 설정
- [ ] 실제 바이너리 다운로드 & 로드 통합 테스트 (Phase 3에서 검증)

### Phase 2 ✅ COMPLETED
- [x] HuggingFace GGUF 파일 다운로드
- [x] 별칭으로 모델 로드 가능
- [x] GgufModelInfo 레코드 정의
- [x] GgufModelRegistry (8개 별칭: default, fast, quality, large, multilingual, korean, code, reasoning)
- [x] GgufModelDownloader (자동 양자화 선택, 이력서 다운로드 지원)
- [x] GgufModelRegistryTests (23개 테스트 통과)

### Phase 3 ✅ COMPLETED
- [x] 텍스트 생성 스트리밍 동작
- [x] Chat 대화 동작
- [x] 리소스 정상 해제
- [x] GgufGeneratorModel 구현 완료
- [x] GeneratorModelLoader GGUF 라우팅 구현

### Phase 4 ✅ COMPLETED
- [x] 주요 모델 (Llama, Qwen, Gemma, EXAONE, DeepSeek, Mistral) chat 포맷 지원
- [x] GgufChatFormatDetector 구현
- [x] 4개 새 ChatFormatter 추가 (Gemma, EXAONE, DeepSeek, Mistral)
- [x] ChatFormatterFactory 확장

### Phase 5 ✅ COMPLETED
- [x] Integration tests 작성 (GgufIntegrationTests, GgufModelFormatTests)
- [x] docs/generator.md GGUF 섹션 추가
- [x] README.md GGUF 예제 및 테이블 추가
- [x] GeneratorGgufSample 예제 프로젝트 작성
- [x] 전체 테스트 통과 (903 tests)

---

## Recommended Start Point

**Phase 0, Task 0.1**부터 시작: `IGeneratorBackend` 인터페이스 정의

```bash
# 브랜치 확인
git checkout feature/gguf-support-planning

# 문서 검토 후 구현 브랜치 생성
git checkout -b feature/gguf-support-phase0
```

---

## Timeline Estimate

| Phase | Tasks | Dependencies |
|-------|-------|--------------|
| Phase 0 | 5 | None |
| Phase 1 | 6 | Phase 0 |
| Phase 2 | 5 | Phase 1 |
| Phase 3 | 5 | Phase 1, 2 |
| Phase 4 | 4 | Phase 3 |
| Phase 5 | 5 | All |

**Total Tasks: 30**

---

## References

- [LLamaSharp Documentation](https://scisharp.github.io/LLamaSharp/)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [GGUF Format on HuggingFace](https://huggingface.co/docs/hub/en/gguf)
- [Bartowski GGUF Models](https://huggingface.co/bartowski)

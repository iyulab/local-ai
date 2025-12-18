namespace LMSupply.Console.Host.Models.Responses;

/// <summary>
/// 시스템 상태
/// </summary>
public sealed record SystemStatus
{
    /// <summary>
    /// ONNX Runtime 준비 상태
    /// </summary>
    public bool EngineReady { get; init; }

    /// <summary>
    /// GPU 사용 가능 여부
    /// </summary>
    public bool GpuAvailable { get; init; }

    /// <summary>
    /// 실행 공급자 (CUDA, DirectML, CPU 등)
    /// </summary>
    public string? GpuProvider { get; init; }

    /// <summary>
    /// GPU 이름
    /// </summary>
    public string? GpuName { get; init; }

    /// <summary>
    /// CPU 사용률 (0-100)
    /// </summary>
    public float CpuUsage { get; init; }

    /// <summary>
    /// RAM 사용량 (MB)
    /// </summary>
    public double RamUsageMB { get; init; }

    /// <summary>
    /// RAM 전체 용량 (MB)
    /// </summary>
    public double RamTotalMB { get; init; }

    /// <summary>
    /// RAM 사용률 (0-100)
    /// </summary>
    public double RamUsagePercent { get; init; }

    /// <summary>
    /// VRAM 사용량 (MB)
    /// </summary>
    public double? VramUsageMB { get; init; }

    /// <summary>
    /// VRAM 전체 용량 (MB)
    /// </summary>
    public double? VramTotalMB { get; init; }

    /// <summary>
    /// VRAM 사용률 (0-100)
    /// </summary>
    public double? VramUsagePercent { get; init; }

    /// <summary>
    /// 현재 프로세스 메모리 (MB)
    /// </summary>
    public double ProcessMemoryMB { get; init; }

    /// <summary>
    /// 타임스탬프
    /// </summary>
    public DateTime Timestamp { get; init; }
}

/// <summary>
/// 메모리 메트릭
/// </summary>
public sealed record MemoryMetrics
{
    public double TotalMB { get; init; }
    public double UsedMB { get; init; }
    public double UsagePercent { get; init; }
}

/// <summary>
/// GPU 정보
/// </summary>
public sealed record GpuInfo
{
    public bool IsAvailable { get; init; }
    public string? Name { get; init; }
    public string? Provider { get; init; }
    public double? TotalVramMB { get; init; }
    public double? UsedVramMB { get; init; }
    public double? VramUsagePercent => TotalVramMB > 0 ? (UsedVramMB / TotalVramMB) * 100 : null;
}

/// <summary>
/// 실시간 메트릭 (스트리밍용)
/// </summary>
public sealed record SystemMetrics
{
    public float CpuUsage { get; init; }
    public double RamUsageMB { get; init; }
    public DateTime Timestamp { get; init; }
}

/// <summary>
/// 로드된 모델 정보
/// </summary>
public sealed record LoadedModelInfo
{
    public required string ModelId { get; init; }
    public required string ModelType { get; init; }
    public DateTime LoadedAt { get; init; }
    public DateTime LastUsedAt { get; init; }
}

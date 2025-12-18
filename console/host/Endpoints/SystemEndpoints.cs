using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class SystemEndpoints
{
    public static void MapSystemEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/system")
            .WithTags("System")
            .WithOpenApi();

        // 시스템 상태
        group.MapGet("/status", (SystemMonitorService monitor, ModelManagerService modelManager) =>
        {
            var status = monitor.GetStatus();
            var loadedModels = modelManager.GetLoadedModels();

            return Results.Ok(new
            {
                status,
                loadedModels = loadedModels.Count,
                models = loadedModels
            });
        })
        .WithName("GetSystemStatus")
        .WithSummary("시스템 상태 조회");

        // GPU 정보
        group.MapGet("/gpu", (SystemMonitorService monitor) =>
        {
            var gpuInfo = monitor.GetGpuInfo();
            return Results.Ok(gpuInfo);
        })
        .WithName("GetGpuInfo")
        .WithSummary("GPU 정보 조회");

        // 메모리 메트릭
        group.MapGet("/memory", (SystemMonitorService monitor) =>
        {
            var memory = monitor.GetMemoryMetrics();
            return Results.Ok(memory);
        })
        .WithName("GetMemoryMetrics")
        .WithSummary("메모리 메트릭 조회");

        // 실시간 메트릭 스트림 (SSE)
        group.MapGet("/metrics/stream", async (HttpContext context, SystemMonitorService monitor) =>
        {
            var cancellationToken = context.RequestAborted;
            var metrics = monitor.StreamMetricsAsync(cancellationToken, intervalMs: 1000);

            await SseHelper.StreamAsync(context, metrics, cancellationToken);
        })
        .WithName("StreamMetrics")
        .WithSummary("실시간 메트릭 스트림 (SSE)");
    }
}

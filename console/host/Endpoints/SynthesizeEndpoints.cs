using LMSupply.Download;
using LMSupply.Console.Host.Models.Requests;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class SynthesizeEndpoints
{
    public static void MapSynthesizeEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/synthesize")
            .WithTags("Synthesize")
            .WithOpenApi();

        // TTS (WAV 바이너리 반환)
        group.MapPost("/", async (SynthesizeRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                var synthesizer = await manager.GetSynthesizerAsync(request.ModelId, ct);

                var result = await synthesizer.SynthesizeAsync(request.Text, cancellationToken: ct);

                return Results.File(
                    result.ToWavBytes(),
                    contentType: "audio/wav",
                    fileDownloadName: "speech.wav");
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .WithName("Synthesize")
        .WithSummary("텍스트를 음성으로 합성 (WAV)");

        // TTS (Base64 JSON 반환)
        group.MapPost("/json", async (SynthesizeRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                var synthesizer = await manager.GetSynthesizerAsync(request.ModelId, ct);
                var result = await synthesizer.SynthesizeAsync(request.Text, cancellationToken: ct);

                return Results.Ok(new
                {
                    modelId = synthesizer.ModelId,
                    text = request.Text,
                    audioBase64 = Convert.ToBase64String(result.ToWavBytes()),
                    sampleRate = result.SampleRate,
                    durationSeconds = result.DurationSeconds
                });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .WithName("SynthesizeJson")
        .WithSummary("텍스트를 음성으로 합성 (JSON + Base64)");

        // 사용 가능한 Synthesizer 모델 목록
        group.MapGet("/models", (CacheService cache) =>
        {
            var models = cache.GetCachedModelsByType(ModelType.Synthesizer);
            return Results.Ok(models.Select(m => new
            {
                m.RepoId,
                m.SizeMB,
                m.LastModified
            }));
        })
        .WithName("GetSynthesizeModels")
        .WithSummary("사용 가능한 Synthesizer 모델 목록");
    }
}

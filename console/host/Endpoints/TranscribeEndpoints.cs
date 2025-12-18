using LMSupply.Download;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class TranscribeEndpoints
{
    public static void MapTranscribeEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/transcribe")
            .WithTags("Transcribe")
            .WithOpenApi();

        // 오디오 파일 트랜스크립션
        group.MapPost("/", async (HttpRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                if (!request.HasFormContentType)
                {
                    return Results.BadRequest(new { error = "Form data expected" });
                }

                var form = await request.ReadFormAsync(ct);
                var file = form.Files.GetFile("audio");

                if (file == null || file.Length == 0)
                {
                    return Results.BadRequest(new { error = "Audio file is required" });
                }

                var modelId = form["modelId"].FirstOrDefault() ?? "default";
                var language = form["language"].FirstOrDefault();

                var transcriber = await manager.GetTranscriberAsync(modelId, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);

                var result = await transcriber.TranscribeAsync(memoryStream.ToArray(), cancellationToken: ct);

                return Results.Ok(new
                {
                    modelId = transcriber.ModelId,
                    text = result.Text,
                    language = result.Language,
                    duration = result.DurationSeconds,
                    segments = result.Segments?.Select(s => new
                    {
                        start = s.Start,
                        end = s.End,
                        text = s.Text
                    })
                });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .DisableAntiforgery()
        .WithName("Transcribe")
        .WithSummary("오디오 파일 트랜스크립션");

        // 사용 가능한 Transcriber 모델 목록
        group.MapGet("/models", (CacheService cache) =>
        {
            var models = cache.GetCachedModelsByType(ModelType.Transcriber);
            return Results.Ok(models.Select(m => new
            {
                m.RepoId,
                m.SizeMB,
                m.LastModified
            }));
        })
        .WithName("GetTranscribeModels")
        .WithSummary("사용 가능한 Transcriber 모델 목록");
    }
}

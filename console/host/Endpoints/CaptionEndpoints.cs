using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class CaptionEndpoints
{
    public static void MapCaptionEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/caption")
            .WithTags("Caption")
            .WithOpenApi();

        // 이미지 캡셔닝
        group.MapPost("/", async (HttpRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                if (!request.HasFormContentType)
                {
                    return Results.BadRequest(new { error = "Form data expected" });
                }

                var form = await request.ReadFormAsync(ct);
                var file = form.Files.GetFile("image");

                if (file == null || file.Length == 0)
                {
                    return Results.BadRequest(new { error = "Image file is required" });
                }

                var modelId = form["modelId"].FirstOrDefault() ?? "default";

                var captioner = await manager.GetCaptionerAsync(modelId, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);

                var result = await captioner.CaptionAsync(memoryStream.ToArray(), ct);

                return Results.Ok(new
                {
                    modelId = captioner.ModelId,
                    caption = result.Caption,
                    confidence = result.Confidence,
                    alternatives = result.AlternativeCaptions
                });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .DisableAntiforgery()
        .WithName("CaptionImage")
        .WithSummary("이미지 캡셔닝");

        // VQA (Visual Question Answering)
        group.MapPost("/vqa", async (HttpRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                if (!request.HasFormContentType)
                {
                    return Results.BadRequest(new { error = "Form data expected" });
                }

                var form = await request.ReadFormAsync(ct);
                var file = form.Files.GetFile("image");

                if (file == null || file.Length == 0)
                {
                    return Results.BadRequest(new { error = "Image file is required" });
                }

                var question = form["question"].FirstOrDefault();
                if (string.IsNullOrWhiteSpace(question))
                {
                    return Results.BadRequest(new { error = "Question is required" });
                }

                var modelId = form["modelId"].FirstOrDefault() ?? "default";

                var captioner = await manager.GetCaptionerAsync(modelId, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);
                memoryStream.Position = 0;

                var result = await captioner.AnswerAsync(memoryStream, question, ct);

                return Results.Ok(new
                {
                    modelId = captioner.ModelId,
                    question,
                    answer = result.Answer,
                    confidence = result.Confidence
                });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .DisableAntiforgery()
        .WithName("VisualQA")
        .WithSummary("이미지 질문 응답 (VQA)");

        // 사용 가능한 Captioner 모델 목록
        group.MapGet("/models", () =>
        {
            var models = LMSupply.Captioner.LocalCaptioner.GetAvailableModels();
            return Results.Ok(models.Select(m => new { alias = m }));
        })
        .WithName("GetCaptionModels")
        .WithSummary("사용 가능한 Captioner 모델 목록");
    }
}

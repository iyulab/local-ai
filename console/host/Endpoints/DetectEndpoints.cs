using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class DetectEndpoints
{
    public static void MapDetectEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/detect")
            .WithTags("Detect")
            .WithOpenApi();

        // 객체 감지
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
                var confidenceThresholdStr = form["confidenceThreshold"].FirstOrDefault();
                var confidenceThreshold = float.TryParse(confidenceThresholdStr, out var ct2) ? ct2 : 0.5f;

                var detector = await manager.GetDetectorAsync(modelId, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);

                var allResults = await detector.DetectAsync(memoryStream.ToArray(), ct);
                // Filter by confidence threshold on client side
                var results = allResults.Where(r => r.Confidence >= confidenceThreshold).ToList();

                return Results.Ok(new
                {
                    modelId = detector.ModelId,
                    count = results.Count,
                    detections = results.Select(r => new
                    {
                        classId = r.ClassId,
                        label = r.Label,
                        confidence = r.Confidence,
                        boundingBox = new
                        {
                            x = r.Box.X1,
                            y = r.Box.Y1,
                            width = r.Box.Width,
                            height = r.Box.Height
                        }
                    })
                });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .DisableAntiforgery()
        .WithName("DetectObjects")
        .WithSummary("이미지에서 객체 감지");

        // 사용 가능한 Detector 모델 목록
        group.MapGet("/models", () =>
        {
            var models = LMSupply.Detector.LocalDetector.GetAllModels();
            return Results.Ok(models.Select(m => new
            {
                alias = m.Alias,
                id = m.Id,
                description = m.Description
            }));
        })
        .WithName("GetDetectModels")
        .WithSummary("사용 가능한 Detector 모델 목록");

        // COCO 클래스 레이블 목록
        group.MapGet("/labels", () =>
        {
            var labels = LMSupply.Detector.LocalDetector.CocoClassLabels;
            return Results.Ok(labels.Select((l, i) => new { classId = i, label = l }));
        })
        .WithName("GetCocoLabels")
        .WithSummary("COCO 클래스 레이블 목록");
    }
}

using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class OcrEndpoints
{
    public static void MapOcrEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/ocr")
            .WithTags("OCR")
            .WithOpenApi();

        // 이미지 OCR (텍스트 인식)
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

                var language = form["language"].FirstOrDefault() ?? "en";

                var ocr = await manager.GetOcrAsync(language, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);

                var result = await ocr.RecognizeAsync(memoryStream.ToArray(), ct);

                return Results.Ok(new
                {
                    detectionModelId = ocr.DetectionModelId,
                    recognitionModelId = ocr.RecognitionModelId,
                    text = result.FullText,
                    regions = result.Regions.Select(r => new
                    {
                        text = r.Text,
                        confidence = r.Confidence,
                        boundingBox = new
                        {
                            x = r.BoundingBox.X,
                            y = r.BoundingBox.Y,
                            width = r.BoundingBox.Width,
                            height = r.BoundingBox.Height
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
        .WithName("RecognizeText")
        .WithSummary("이미지에서 텍스트 인식");

        // 텍스트 영역만 감지 (인식 없이)
        group.MapPost("/detect", async (HttpRequest request, ModelManagerService manager, CancellationToken ct) =>
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

                var language = form["language"].FirstOrDefault() ?? "en";

                var ocr = await manager.GetOcrAsync(language, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);

                memoryStream.Position = 0;
                var regions = await ocr.DetectAsync(memoryStream, ct);

                return Results.Ok(new
                {
                    detectionModelId = ocr.DetectionModelId,
                    count = regions.Count,
                    regions = regions.Select(r => new
                    {
                        confidence = r.Confidence,
                        boundingBox = new
                        {
                            x = r.BoundingBox.X,
                            y = r.BoundingBox.Y,
                            width = r.BoundingBox.Width,
                            height = r.BoundingBox.Height
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
        .WithName("DetectTextRegions")
        .WithSummary("이미지에서 텍스트 영역만 감지");

        // 사용 가능한 언어 목록
        group.MapGet("/languages", () =>
        {
            var languages = LMSupply.Ocr.LocalOcr.GetSupportedLanguages();
            return Results.Ok(languages.Select(l => new { code = l }));
        })
        .WithName("GetOcrLanguages")
        .WithSummary("OCR 지원 언어 목록");

        // 사용 가능한 모델 목록
        group.MapGet("/models", () =>
        {
            var detection = LMSupply.Ocr.LocalOcr.GetAvailableDetectionModels();
            var recognition = LMSupply.Ocr.LocalOcr.GetAvailableRecognitionModels();
            return Results.Ok(new
            {
                detection = detection.Select(m => new { alias = m }),
                recognition = recognition.Select(m => new { alias = m })
            });
        })
        .WithName("GetOcrModels")
        .WithSummary("사용 가능한 OCR 모델 목록");
    }
}

using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class SegmentEndpoints
{
    public static void MapSegmentEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/segment")
            .WithTags("Segment")
            .WithOpenApi();

        // 이미지 세그멘테이션
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

                var segmenter = await manager.GetSegmenterAsync(modelId, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);

                var result = await segmenter.SegmentAsync(memoryStream.ToArray(), ct);

                // ClassMap을 요약 정보로 변환 (전체 배열은 너무 큼)
                var classHistogram = result.ClassMap
                    .GroupBy(c => c)
                    .OrderByDescending(g => g.Count())
                    .Take(10)
                    .ToDictionary(g => g.Key, g => g.Count());

                var labels = LMSupply.Segmenter.LocalSegmenter.Ade20kClassLabels;

                return Results.Ok(new
                {
                    modelId = segmenter.ModelId,
                    width = result.Width,
                    height = result.Height,
                    numClasses = result.UniqueClassCount,
                    topClasses = classHistogram.Select(kv => new
                    {
                        classId = kv.Key,
                        label = kv.Key >= 0 && kv.Key < labels.Count ? labels[kv.Key] : "unknown",
                        pixelCount = kv.Value,
                        percentage = Math.Round((double)kv.Value / result.ClassMap.Length * 100, 2)
                    })
                });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .DisableAntiforgery()
        .WithName("SegmentImage")
        .WithSummary("이미지 세그멘테이션");

        // 세그멘테이션 결과를 마스크 이미지로 반환
        group.MapPost("/mask", async (HttpRequest request, ModelManagerService manager, CancellationToken ct) =>
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

                var segmenter = await manager.GetSegmenterAsync(modelId, ct);

                using var stream = file.OpenReadStream();
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream, ct);

                var result = await segmenter.SegmentAsync(memoryStream.ToArray(), ct);

                // ClassMap을 Base64로 인코딩하여 반환
                var classMapBytes = new byte[result.ClassMap.Length];
                for (int i = 0; i < result.ClassMap.Length; i++)
                {
                    classMapBytes[i] = (byte)Math.Clamp(result.ClassMap[i], 0, 255);
                }

                return Results.Ok(new
                {
                    modelId = segmenter.ModelId,
                    width = result.Width,
                    height = result.Height,
                    numClasses = result.UniqueClassCount,
                    classMapBase64 = Convert.ToBase64String(classMapBytes)
                });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .DisableAntiforgery()
        .WithName("GetSegmentMask")
        .WithSummary("세그멘테이션 마스크 반환");

        // 사용 가능한 Segmenter 모델 목록
        group.MapGet("/models", () =>
        {
            var models = LMSupply.Segmenter.LocalSegmenter.GetAllModels();
            return Results.Ok(models.Select(m => new
            {
                alias = m.Alias,
                id = m.Id,
                description = m.Description
            }));
        })
        .WithName("GetSegmentModels")
        .WithSummary("사용 가능한 Segmenter 모델 목록");

        // ADE20K 클래스 레이블 목록
        group.MapGet("/labels", () =>
        {
            var labels = LMSupply.Segmenter.LocalSegmenter.Ade20kClassLabels;
            return Results.Ok(labels.Select((l, i) => new { classId = i, label = l }));
        })
        .WithName("GetAde20kLabels")
        .WithSummary("ADE20K 클래스 레이블 목록");
    }
}

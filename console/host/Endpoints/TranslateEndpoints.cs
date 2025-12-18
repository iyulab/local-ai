using LMSupply.Console.Host.Models.Requests;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class TranslateEndpoints
{
    public static void MapTranslateEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/translate")
            .WithTags("Translate")
            .WithOpenApi();

        // 번역
        group.MapPost("/", async (TranslateRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                var translator = await manager.GetTranslatorAsync(request.ModelId, ct);

                if (request.Texts != null && request.Texts.Count > 0)
                {
                    // 배치 번역
                    var results = await translator.TranslateBatchAsync(request.Texts, ct);
                    return Results.Ok(new
                    {
                        modelId = translator.ModelId,
                        translations = results.Select((r, i) => new
                        {
                            index = i,
                            sourceText = r.SourceText,
                            translatedText = r.TranslatedText,
                            sourceLanguage = r.SourceLanguage,
                            targetLanguage = r.TargetLanguage
                        })
                    });
                }
                else if (!string.IsNullOrEmpty(request.Text))
                {
                    // 단일 번역
                    var result = await translator.TranslateAsync(request.Text, ct);
                    return Results.Ok(new
                    {
                        modelId = translator.ModelId,
                        sourceText = result.SourceText,
                        translatedText = result.TranslatedText,
                        sourceLanguage = result.SourceLanguage,
                        targetLanguage = result.TargetLanguage
                    });
                }

                return Results.BadRequest(new { error = "Either 'text' or 'texts' must be provided" });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .WithName("Translate")
        .WithSummary("텍스트 번역");

        // 사용 가능한 Translator 모델 목록
        group.MapGet("/models", () =>
        {
            var models = LMSupply.Translator.LocalTranslator.GetAllModels();
            return Results.Ok(models.Select(m => new
            {
                alias = m.Alias,
                id = m.Id,
                sourceLanguage = m.SourceLanguage,
                targetLanguage = m.TargetLanguage
            }));
        })
        .WithName("GetTranslateModels")
        .WithSummary("사용 가능한 Translator 모델 목록");
    }
}

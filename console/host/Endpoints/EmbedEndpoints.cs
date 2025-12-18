using System.Numerics.Tensors;
using LMSupply.Download;
using LMSupply.Console.Host.Models.Requests;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class EmbedEndpoints
{
    public static void MapEmbedEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/embed")
            .WithTags("Embed")
            .WithOpenApi();

        // 임베딩 생성
        group.MapPost("/", async (EmbedRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                var embedder = await manager.GetEmbedderAsync(request.ModelId, ct);

                if (request.Texts != null && request.Texts.Count > 0)
                {
                    // 배치 임베딩
                    var embeddings = await embedder.EmbedAsync(request.Texts, ct);
                    return Results.Ok(new
                    {
                        modelId = embedder.ModelId,
                        dimensions = embedder.Dimensions,
                        embeddings = embeddings.Select((e, i) => new
                        {
                            index = i,
                            embedding = e
                        })
                    });
                }
                else if (!string.IsNullOrEmpty(request.Text))
                {
                    // 단일 임베딩
                    var embedding = await embedder.EmbedAsync(request.Text, ct);
                    return Results.Ok(new
                    {
                        modelId = embedder.ModelId,
                        dimensions = embedder.Dimensions,
                        embedding
                    });
                }

                return Results.BadRequest(new { error = "Either 'text' or 'texts' must be provided" });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .WithName("CreateEmbedding")
        .WithSummary("텍스트 임베딩 생성");

        // 유사도 계산
        group.MapPost("/similarity", async (SimilarityRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                var embedder = await manager.GetEmbedderAsync(request.ModelId, ct);

                var embeddings = await embedder.EmbedAsync([request.Text1, request.Text2], ct);
                var similarity = TensorPrimitives.CosineSimilarity(embeddings[0], embeddings[1]);

                return Results.Ok(new
                {
                    modelId = embedder.ModelId,
                    text1 = request.Text1,
                    text2 = request.Text2,
                    similarity
                });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .WithName("CalculateSimilarity")
        .WithSummary("두 텍스트 간 유사도 계산");

        // 사용 가능한 Embedder 모델 목록
        group.MapGet("/models", (CacheService cache) =>
        {
            var models = cache.GetCachedModelsByType(ModelType.Embedder);
            return Results.Ok(models.Select(m => new
            {
                m.RepoId,
                m.SizeMB,
                m.LastModified
            }));
        })
        .WithName("GetEmbedModels")
        .WithSummary("사용 가능한 Embedder 모델 목록");
    }
}

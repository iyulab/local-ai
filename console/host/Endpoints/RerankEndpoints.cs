using LMSupply.Download;
using LMSupply.Console.Host.Models.Requests;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class RerankEndpoints
{
    public static void MapRerankEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/rerank")
            .WithTags("Rerank")
            .WithOpenApi();

        // 리랭킹
        group.MapPost("/", async (RerankRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                var reranker = await manager.GetRerankerAsync(request.ModelId, ct);

                var results = await reranker.RerankAsync(
                    request.Query,
                    request.Documents,
                    request.TopK,
                    ct);

                return Results.Ok(new
                {
                    modelId = reranker.ModelId,
                    query = request.Query,
                    results = results.Select(r => new
                    {
                        index = r.OriginalIndex,
                        score = r.Score,
                        document = r.Document
                    })
                });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .WithName("Rerank")
        .WithSummary("문서 리랭킹");

        // 점수만 계산 (정렬 없음)
        group.MapPost("/score", async (RerankRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                var reranker = await manager.GetRerankerAsync(request.ModelId, ct);
                var scores = await reranker.ScoreAsync(request.Query, request.Documents, ct);

                return Results.Ok(new
                {
                    modelId = reranker.ModelId,
                    query = request.Query,
                    scores = request.Documents.Zip(scores, (doc, score) => new
                    {
                        document = doc,
                        score
                    })
                });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .WithName("Score")
        .WithSummary("관련성 점수 계산");

        // 사용 가능한 Reranker 모델 목록
        group.MapGet("/models", (CacheService cache) =>
        {
            var models = cache.GetCachedModelsByType(ModelType.Reranker);
            return Results.Ok(models.Select(m => new
            {
                m.RepoId,
                m.SizeMB,
                m.LastModified
            }));
        })
        .WithName("GetRerankModels")
        .WithSummary("사용 가능한 Reranker 모델 목록");
    }
}

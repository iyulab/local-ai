using LMSupply.Download;
using LMSupply.Generator.Models;
using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.Requests;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class ChatEndpoints
{
    public static void MapChatEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/chat")
            .WithTags("Chat")
            .WithOpenApi();

        // 채팅 (SSE 스트리밍)
        group.MapPost("/", async (ChatRequest request, HttpContext context, ModelManagerService manager) =>
        {
            try
            {
                var cancellationToken = context.RequestAborted;
                var generator = await manager.GetGeneratorAsync(request.ModelId, cancellationToken);

                var messages = request.Messages.Select(m =>
                    new ChatMessage(Enum.Parse<ChatRole>(m.Role, ignoreCase: true), m.Content));

                var options = request.Options != null
                    ? new GenerationOptions
                    {
                        MaxTokens = request.Options.MaxTokens,
                        Temperature = request.Options.Temperature,
                        TopP = request.Options.TopP,
                        TopK = request.Options.TopK,
                        RepetitionPenalty = request.Options.RepetitionPenalty,
                        StopSequences = request.Options.StopSequences?.ToList()
                    }
                    : null;

                var tokens = generator.GenerateChatAsync(messages, options, cancellationToken);
                await SseHelper.StreamTokensAsync(context, tokens, cancellationToken);
            }
            catch (Exception ex)
            {
                context.Response.StatusCode = 500;
                await context.Response.WriteAsJsonAsync(new { error = ex.Message });
            }
        })
        .WithName("ChatStream")
        .WithSummary("채팅 (SSE 스트리밍)");

        // 채팅 (비스트리밍)
        group.MapPost("/complete", async (ChatRequest request, ModelManagerService manager, CancellationToken ct) =>
        {
            try
            {
                var generator = await manager.GetGeneratorAsync(request.ModelId, ct);

                var messages = request.Messages.Select(m =>
                    new ChatMessage(Enum.Parse<ChatRole>(m.Role, ignoreCase: true), m.Content));

                var options = request.Options != null
                    ? new GenerationOptions
                    {
                        MaxTokens = request.Options.MaxTokens,
                        Temperature = request.Options.Temperature,
                        TopP = request.Options.TopP,
                        TopK = request.Options.TopK,
                        RepetitionPenalty = request.Options.RepetitionPenalty,
                        StopSequences = request.Options.StopSequences?.ToList()
                    }
                    : null;

                var response = await generator.GenerateChatCompleteAsync(messages, options, ct);

                return Results.Ok(new
                {
                    modelId = generator.ModelId,
                    response,
                    usage = new
                    {
                        maxContextLength = generator.MaxContextLength
                    }
                });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message);
            }
        })
        .WithName("ChatComplete")
        .WithSummary("채팅 (비스트리밍)");

        // 사용 가능한 Generator 모델 목록
        group.MapGet("/models", (CacheService cache) =>
        {
            var models = cache.GetCachedModelsByType(ModelType.Generator);
            return Results.Ok(models.Select(m => new
            {
                m.RepoId,
                m.SizeMB,
                m.LastModified
            }));
        })
        .WithName("GetChatModels")
        .WithSummary("사용 가능한 Generator 모델 목록");
    }
}

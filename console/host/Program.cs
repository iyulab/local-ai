using System.Text.Json;
using System.Text.Json.Serialization;
using LMSupply.Console.Host.Endpoints;
using LMSupply.Console.Host.Services;

var builder = WebApplication.CreateBuilder(args);

// JSON 직렬화 설정
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
    options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
    options.SerializerOptions.Converters.Add(new JsonStringEnumConverter(JsonNamingPolicy.CamelCase));
});

// CORS 설정 (개발용)
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.WithOrigins("http://localhost:5173", "http://localhost:3000")
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials();
    });
});

// OpenAPI/Swagger
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new() { Title = "LMSupply Console API", Version = "v1" });
});

// 서비스 등록
builder.Services.AddSingleton<CacheService>();
builder.Services.AddSingleton<SystemMonitorService>();
builder.Services.AddSingleton<ModelManagerService>();
builder.Services.AddSingleton<DownloadService>();

var app = builder.Build();

// 개발 환경에서 Swagger UI
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseCors();

// API 엔드포인트 매핑
app.MapModelsEndpoints();
app.MapSystemEndpoints();
app.MapChatEndpoints();
app.MapEmbedEndpoints();
app.MapRerankEndpoints();
app.MapTranscribeEndpoints();
app.MapSynthesizeEndpoints();
app.MapCaptionEndpoints();
app.MapOcrEndpoints();
app.MapDetectEndpoints();
app.MapSegmentEndpoints();
app.MapTranslateEndpoints();

// Health check
app.MapGet("/health", () => Results.Ok(new { status = "healthy", timestamp = DateTime.UtcNow }));

app.Run();

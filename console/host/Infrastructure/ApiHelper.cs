using LMSupply.Console.Host.Models.OpenAI;

namespace LMSupply.Console.Host.Infrastructure;

/// <summary>
/// API helper utilities
/// </summary>
public static class ApiHelper
{
    /// <summary>
    /// Generate a unique ID with prefix
    /// </summary>
    public static string GenerateId(string prefix) => $"{prefix}-{Guid.NewGuid():N}";

    /// <summary>
    /// Create an error response
    /// </summary>
    public static IResult Error(string message, string type = "invalid_request_error", int statusCode = 400)
    {
        var error = new ErrorResponse
        {
            Error = new ErrorDetail
            {
                Message = message,
                Type = type
            }
        };

        return Results.Json(error, statusCode: statusCode);
    }

    /// <summary>
    /// Create an internal error response
    /// </summary>
    public static IResult InternalError(Exception ex)
    {
        var error = new ErrorResponse
        {
            Error = new ErrorDetail
            {
                Message = ex.Message,
                Type = "internal_error"
            }
        };

        return Results.Json(error, statusCode: 500);
    }

    /// <summary>
    /// Parse input that can be string or array of strings
    /// </summary>
    public static List<string> ParseInput(System.Text.Json.JsonElement input)
    {
        if (input.ValueKind == System.Text.Json.JsonValueKind.String)
        {
            return [input.GetString()!];
        }

        if (input.ValueKind == System.Text.Json.JsonValueKind.Array)
        {
            return input.EnumerateArray()
                .Select(e => e.GetString()!)
                .ToList();
        }

        throw new ArgumentException("Input must be a string or array of strings");
    }
}

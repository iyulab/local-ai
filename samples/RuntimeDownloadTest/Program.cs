// Runtime Download Test - Tests that GenAI runtime binaries are downloaded from NuGet
// This is a minimal test that doesn't require model download

using LMSupply;
using LMSupply.Runtime;

Console.WriteLine("=== ONNX Runtime GenAI Download Test ===\n");

// Progress reporter
var progress = new Progress<DownloadProgress>(p =>
{
    if (p.TotalBytes > 0)
    {
        var percent = (double)p.BytesDownloaded / p.TotalBytes;
        Console.Write($"\r[{percent:P0}] {p.FileName}                    ");
    }
    else if (!string.IsNullOrEmpty(p.FileName))
    {
        Console.WriteLine($"  {p.FileName}");
    }
});

try
{
    // Step 1: Initialize RuntimeManager
    Console.WriteLine("[Step 1] Initializing RuntimeManager...");
    await RuntimeManager.Instance.InitializeAsync();
    Console.WriteLine($"  Platform: {RuntimeManager.Instance.Platform.RuntimeIdentifier}");
    Console.WriteLine($"  GPU: {RuntimeManager.Instance.Gpu}");
    Console.WriteLine($"  Recommended: {RuntimeManager.Instance.RecommendedProvider}\n");

    // Step 2: Download base ONNX Runtime
    Console.WriteLine("[Step 2] Downloading ONNX Runtime (base)...");
    var onnxPath = await RuntimeManager.Instance.EnsureRuntimeAsync(
        "onnxruntime",
        provider: "cpu",
        progress: progress);
    Console.WriteLine($"\n  Downloaded to: {onnxPath}\n");

    // Step 3: Download ONNX Runtime GenAI
    Console.WriteLine("[Step 3] Downloading ONNX Runtime GenAI...");
    var genaiPath = await RuntimeManager.Instance.EnsureRuntimeAsync(
        "onnxruntime-genai",
        provider: "cpu",
        progress: progress);
    Console.WriteLine($"\n  Downloaded to: {genaiPath}\n");

    // Step 4: Verify files exist
    Console.WriteLine("[Step 4] Verifying downloaded files...");
    var onnxFiles = Directory.GetFiles(onnxPath, "*.*", SearchOption.TopDirectoryOnly);
    var genaiFiles = Directory.GetFiles(genaiPath, "*.*", SearchOption.TopDirectoryOnly);

    Console.WriteLine($"  ONNX Runtime files: {onnxFiles.Length}");
    foreach (var f in onnxFiles.Take(5))
        Console.WriteLine($"    - {Path.GetFileName(f)}");

    Console.WriteLine($"  ONNX Runtime GenAI files: {genaiFiles.Length}");
    foreach (var f in genaiFiles.Take(5))
        Console.WriteLine($"    - {Path.GetFileName(f)}");

    Console.WriteLine("\n" + new string('=', 50));
    Console.WriteLine("[SUCCESS] Runtime download test passed!");
    Console.WriteLine("The new NuGet-based download system works correctly.");
    Console.WriteLine(new string('=', 50));
}
catch (Exception ex)
{
    Console.WriteLine($"\n[ERROR] {ex.GetType().Name}: {ex.Message}");
    if (ex.InnerException != null)
    {
        Console.WriteLine($"  Inner: {ex.InnerException.Message}");
    }
    Console.WriteLine($"\nStack trace:\n{ex.StackTrace}");
    Environment.Exit(1);
}

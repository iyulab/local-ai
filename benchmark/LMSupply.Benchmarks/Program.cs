using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Exporters.Csv;
using BenchmarkDotNet.Running;

namespace LMSupply.Benchmarks;

public static class Program
{
    public static void Main(string[] args)
    {
        // Create custom config with report exporters
        var config = DefaultConfig.Instance
            .AddExporter(HtmlExporter.Default)
            .AddExporter(MarkdownExporter.GitHub)
            .AddExporter(CsvExporter.Default)
            .WithArtifactsPath(Path.Combine(Directory.GetCurrentDirectory(), "BenchmarkResults"));

        // Run with command line args support
        // Usage:
        //   dotnet run -c Release                    # Run all benchmarks
        //   dotnet run -c Release -- --filter *Embedder*  # Run only embedder benchmarks
        //   dotnet run -c Release -- --list flat     # List all benchmarks
        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);
    }
}

using System.Runtime.InteropServices;
using FluentAssertions;
using LMSupply.Llama;
using LMSupply.Runtime;

namespace LMSupply.Llama.Tests;

public class LlamaRuntimeManagerTests
{
    [Fact]
    public void Instance_ShouldReturnSameInstance()
    {
        var instance1 = LlamaRuntimeManager.Instance;
        var instance2 = LlamaRuntimeManager.Instance;

        instance1.Should().BeSameAs(instance2);
    }

    [Fact]
    public void Instance_ShouldNotBeNull()
    {
        LlamaRuntimeManager.Instance.Should().NotBeNull();
    }

    [Fact]
    public void GetBackendFallbackChain_WindowsNvidia_ShouldIncludeCudaAndCpu()
    {
        // Arrange
        var platform = new PlatformInfo
        {
            OS = OSPlatform.Windows,
            Architecture = Architecture.X64,
            RuntimeIdentifier = "win-x64"
        };

        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "NVIDIA GeForce RTX 4090",
            CudaDriverVersionMajor = 12,
            TotalMemoryBytes = 24576L * 1024 * 1024
        };

        // Act
        var chain = LlamaRuntimeManager.GetBackendFallbackChain(platform, gpu);

        // Assert
        chain.Should().Contain(LlamaBackend.Cuda12);
        chain.Should().Contain(LlamaBackend.Cpu);
        chain.Last().Should().Be(LlamaBackend.Cpu, "CPU should always be the final fallback");
    }

    [Fact]
    public void GetBackendFallbackChain_WindowsNvidiaWithCuda13_ShouldIncludeBothCudaVersions()
    {
        // Arrange
        var platform = new PlatformInfo
        {
            OS = OSPlatform.Windows,
            Architecture = Architecture.X64,
            RuntimeIdentifier = "win-x64"
        };

        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "NVIDIA GeForce RTX 5090",
            CudaDriverVersionMajor = 13,
            TotalMemoryBytes = 32768L * 1024 * 1024
        };

        // Act
        var chain = LlamaRuntimeManager.GetBackendFallbackChain(platform, gpu);

        // Assert
        chain.Should().Contain(LlamaBackend.Cuda13);
        chain.Should().Contain(LlamaBackend.Cuda12);
        chain.Should().Contain(LlamaBackend.Cpu);

        var chainList = chain.ToList();
        chainList.IndexOf(LlamaBackend.Cuda13).Should().BeLessThan(chainList.IndexOf(LlamaBackend.Cuda12),
            "CUDA 13 should come before CUDA 12");
    }

    [Fact]
    public void GetBackendFallbackChain_MacOSArm64_ShouldStartWithMetal()
    {
        // Arrange
        var platform = new PlatformInfo
        {
            OS = OSPlatform.OSX,
            Architecture = Architecture.Arm64,
            RuntimeIdentifier = "osx-arm64"
        };

        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Apple,
            DeviceName = "Apple M2"
        };

        // Act
        var chain = LlamaRuntimeManager.GetBackendFallbackChain(platform, gpu);

        // Assert
        chain.First().Should().Be(LlamaBackend.Metal, "macOS ARM64 should start with Metal");
        chain.Last().Should().Be(LlamaBackend.Cpu);
    }

    [Fact]
    public void GetBackendFallbackChain_LinuxAmd_ShouldIncludeVulkan()
    {
        // Arrange
        var platform = new PlatformInfo
        {
            OS = OSPlatform.Linux,
            Architecture = Architecture.X64,
            RuntimeIdentifier = "linux-x64"
        };

        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Amd,
            DeviceName = "AMD Radeon RX 7900"
        };

        // Act
        var chain = LlamaRuntimeManager.GetBackendFallbackChain(platform, gpu);

        // Assert
        chain.Should().Contain(LlamaBackend.Vulkan);
        chain.Should().Contain(LlamaBackend.Cpu);
    }

    [Fact]
    public void GetBackendFallbackChain_WindowsIntel_ShouldIncludeVulkanAndCpu()
    {
        // Arrange
        var platform = new PlatformInfo
        {
            OS = OSPlatform.Windows,
            Architecture = Architecture.X64,
            RuntimeIdentifier = "win-x64"
        };

        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Intel,
            DeviceName = "Intel Arc A770"
        };

        // Act
        var chain = LlamaRuntimeManager.GetBackendFallbackChain(platform, gpu);

        // Assert
        chain.Should().Contain(LlamaBackend.Vulkan);
        chain.Should().Contain(LlamaBackend.Cpu);
    }

    [Fact]
    public void GetBackendFallbackChain_CpuOnly_ShouldOnlyIncludeCpu()
    {
        // Arrange
        var platform = new PlatformInfo
        {
            OS = OSPlatform.Windows,
            Architecture = Architecture.X64,
            RuntimeIdentifier = "win-x64"
        };

        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Unknown,
            DeviceName = null
        };

        // Act
        var chain = LlamaRuntimeManager.GetBackendFallbackChain(platform, gpu);

        // Assert
        chain.Should().Contain(LlamaBackend.Cpu);
        chain.Should().NotContain(LlamaBackend.Cuda12);
        chain.Should().NotContain(LlamaBackend.Cuda13);
        chain.Should().NotContain(LlamaBackend.Metal);
    }

    [Fact]
    public void GetBackendFallbackChain_ShouldAlwaysEndWithCpu()
    {
        // Test multiple scenarios to ensure CPU is always the last option
        var scenarios = new (PlatformInfo platform, GpuInfo gpu)[]
        {
            (new PlatformInfo { OS = OSPlatform.Windows, Architecture = Architecture.X64, RuntimeIdentifier = "win-x64" },
             new GpuInfo { Vendor = GpuVendor.Nvidia, CudaDriverVersionMajor = 12 }),
            (new PlatformInfo { OS = OSPlatform.OSX, Architecture = Architecture.Arm64, RuntimeIdentifier = "osx-arm64" },
             new GpuInfo { Vendor = GpuVendor.Apple }),
            (new PlatformInfo { OS = OSPlatform.Linux, Architecture = Architecture.X64, RuntimeIdentifier = "linux-x64" },
             new GpuInfo { Vendor = GpuVendor.Amd }),
            (new PlatformInfo { OS = OSPlatform.Windows, Architecture = Architecture.X64, RuntimeIdentifier = "win-x64" },
             new GpuInfo { Vendor = GpuVendor.Unknown })
        };

        foreach (var (platform, gpu) in scenarios)
        {
            var chain = LlamaRuntimeManager.GetBackendFallbackChain(platform, gpu);
            chain.Should().NotBeEmpty();
            chain.Last().Should().Be(LlamaBackend.Cpu,
                $"CPU should be last for {platform.RuntimeIdentifier} with {gpu.Vendor}");
        }
    }
}

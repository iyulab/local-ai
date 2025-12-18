import { useEffect } from 'react';
import { useSystemStore } from '../stores/systemStore';
import { formatBytes } from '../lib/utils';
import { Cpu, MemoryStick, Gauge, HardDrive, CheckCircle2, XCircle } from 'lucide-react';

function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  color = 'text-primary'
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ElementType;
  color?: string;
}) {
  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-muted-foreground">{title}</p>
          <p className={`text-2xl font-bold ${color}`}>{value}</p>
          {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
        </div>
        <Icon className={`w-8 h-8 ${color} opacity-50`} />
      </div>
    </div>
  );
}

function StatusIndicator({ ready, label }: { ready: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2">
      {ready ? (
        <CheckCircle2 className="w-5 h-5 text-green-500" />
      ) : (
        <XCircle className="w-5 h-5 text-red-500" />
      )}
      <span className="text-sm">{label}</span>
    </div>
  );
}

export function Dashboard() {
  const { status, cachedModels, loadedModels, fetchStatus, fetchModels, fetchLoadedModels } =
    useSystemStore();

  useEffect(() => {
    fetchStatus();
    fetchModels();
    fetchLoadedModels();

    // Poll status every 5 seconds
    const interval = setInterval(() => {
      fetchStatus();
      fetchLoadedModels();
    }, 5000);

    return () => clearInterval(interval);
  }, [fetchStatus, fetchModels, fetchLoadedModels]);

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>

      {/* Status Indicators */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-3">System Status</h2>
        <div className="flex gap-6">
          <StatusIndicator ready={status?.engineReady ?? false} label="ONNX Runtime" />
          <StatusIndicator ready={status?.gpuAvailable ?? false} label="GPU Available" />
          {status?.gpuName && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span className="font-medium">{status.gpuProvider}:</span>
              <span>{status.gpuName}</span>
            </div>
          )}
        </div>
      </div>

      {/* Resource Usage */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="CPU Usage"
          value={`${status?.cpuUsage?.toFixed(1) ?? 0}%`}
          icon={Cpu}
        />
        <StatCard
          title="RAM Usage"
          value={`${status?.ramUsagePercent?.toFixed(0) ?? 0}%`}
          subtitle={`${formatBytes((status?.ramUsageMB ?? 0) * 1024 * 1024)} / ${formatBytes((status?.ramTotalMB ?? 0) * 1024 * 1024)}`}
          icon={MemoryStick}
        />
        {status?.vramTotalMB && (
          <StatCard
            title="VRAM Usage"
            value={`${status.vramUsagePercent?.toFixed(0) ?? 0}%`}
            subtitle={`${formatBytes((status.vramUsageMB ?? 0) * 1024 * 1024)} / ${formatBytes(status.vramTotalMB * 1024 * 1024)}`}
            icon={Gauge}
            color="text-purple-500"
          />
        )}
        <StatCard
          title="Process Memory"
          value={formatBytes((status?.processMemoryMB ?? 0) * 1024 * 1024)}
          icon={HardDrive}
        />
      </div>

      {/* Models Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cached Models */}
        <div className="bg-card border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-3">
            Cached Models ({cachedModels.length})
          </h2>
          <div className="space-y-2 max-h-64 overflow-auto">
            {cachedModels.length === 0 ? (
              <p className="text-muted-foreground text-sm">No models cached</p>
            ) : (
              cachedModels.map((model) => (
                <div
                  key={model.repoId}
                  className="flex items-center justify-between p-2 bg-muted rounded"
                >
                  <div>
                    <p className="font-medium text-sm">{model.repoId}</p>
                    <p className="text-xs text-muted-foreground">
                      {model.detectedType} - {formatBytes(model.sizeBytes)}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Loaded Models */}
        <div className="bg-card border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-3">
            Loaded Models ({loadedModels.length})
          </h2>
          <div className="space-y-2 max-h-64 overflow-auto">
            {loadedModels.length === 0 ? (
              <p className="text-muted-foreground text-sm">No models loaded</p>
            ) : (
              loadedModels.map((model) => (
                <div
                  key={model.modelId}
                  className="flex items-center justify-between p-2 bg-muted rounded"
                >
                  <div>
                    <p className="font-medium text-sm">{model.modelId}</p>
                    <p className="text-xs text-muted-foreground">
                      {model.modelType} - Last used:{' '}
                      {new Date(model.lastUsedAt).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

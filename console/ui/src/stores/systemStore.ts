import { create } from 'zustand';
import type { SystemStatus, CachedModelInfo, LoadedModelInfo } from '../api/types';
import { api } from '../api/client';

interface SystemState {
  status: SystemStatus | null;
  cachedModels: CachedModelInfo[];
  loadedModels: LoadedModelInfo[];
  isLoading: boolean;
  error: string | null;

  fetchStatus: () => Promise<void>;
  fetchModels: () => Promise<void>;
  fetchLoadedModels: () => Promise<void>;
  deleteModel: (repoId: string) => Promise<boolean>;
  unloadModel: (key: string) => Promise<boolean>;
}

export const useSystemStore = create<SystemState>((set, get) => ({
  status: null,
  cachedModels: [],
  loadedModels: [],
  isLoading: false,
  error: null,

  fetchStatus: async () => {
    try {
      const response = await api.getSystemStatus();
      set({ status: response.status, error: null });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  fetchModels: async () => {
    set({ isLoading: true });
    try {
      const response = await api.getCachedModels();
      set({ cachedModels: response.models, isLoading: false, error: null });
    } catch (e) {
      set({ isLoading: false, error: (e as Error).message });
    }
  },

  fetchLoadedModels: async () => {
    try {
      const loadedModels = await api.getLoadedModels();
      set({ loadedModels, error: null });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  deleteModel: async (repoId: string) => {
    try {
      const response = await api.deleteModel(repoId);
      if (response.ok) {
        await get().fetchModels();
        return true;
      }
      return false;
    } catch {
      return false;
    }
  },

  unloadModel: async (key: string) => {
    try {
      const response = await api.unloadModel(key);
      if (response.ok) {
        await get().fetchLoadedModels();
        return true;
      }
      return false;
    } catch {
      return false;
    }
  },
}));

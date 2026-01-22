# LMSupply Roadmap

> **Philosophy**: "No model management. Just use it."
>
> 모든 개선은 이 철학을 강화하는 방향으로 진행됩니다.

---

## ✅ Version 0.10.0 (Released)

**Theme**: Local Performance Maximization & Developer Experience

### Highlights

- **HardwareProfile & PerformanceTier**: 통합 하드웨어 감지 시스템
- **"auto" Model Selection**: 하드웨어 기반 최적 모델 자동 선택
- **Runtime Diagnostics**: 모든 도메인에 `IsGpuActive`, `ActiveProviders`, `EstimatedMemoryBytes` 추가
- **IModelInfoBase**: 통합 모델 정보 인터페이스
- **Documentation**: MODEL_LIFECYCLE.md, GPU_PROVIDERS.md, MEMORY_REQUIREMENTS.md, TROUBLESHOOTING.md

### Completed Tasks

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Core Infrastructure (HardwareProfile, IModelRuntimeInfo, ThreadCount) | ✅ |
| **Phase 2** | Runtime Diagnostics (IsGpuActive, ModelInfo 통일) | ✅ |
| **Phase 3** | Adaptive Model Selection ("auto" mode) | ✅ |
| **Phase 4** | Advanced Features (EstimatedMemoryBytes, HTTP Resume) | ✅ |
| **Phase 5** | Documentation | ✅ |

---

## 🔮 Version 0.11.0 (Planning)

**Theme**: TBD

*다음 버전 계획은 커뮤니티 피드백을 기반으로 수립됩니다.*

### Potential Features

- [ ] Batched inference optimization
- [ ] Model quantization utilities
- [ ] Extended multi-modal support
- [ ] Performance benchmarking tools

---

## Version History

| Version | Theme | Status |
|---------|-------|--------|
| 0.9.2 | ONNX Runtime Management | Released |
| 0.10.0 | Local Performance Max & DX | **Released** |
| 0.11.0 | TBD | Planning |

---

## Related Issues

- `claudedocs/issues/ISSUE-20260122-adaptive-model-selection.md` (Completed)
- `claudedocs/issues/ISSUE-20260122-cachedmodelinfo-metadata-extension.md` (Completed)

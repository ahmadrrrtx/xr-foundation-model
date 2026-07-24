# Changelog

## [v0.1.0] — 2026-07-24

### Added
- Project foundation (`xr-foundation-model`)
- Professional open-source repository structure
- ConfigLoader with YAML validation and dot-notation access
- Config-driven architecture supporting 10M → 1B+ without rewrites
- Git workflow with semantic version tags
- Professional documentation standards (LICENSE, CONTRIBUTING, ROADMAP)

### Design Decisions
- Pure PyTorch (no TensorFlow, no external LLM libraries for core model)
- Config-driven hyperparameters (no hard-coded values)
- Original implementation with documented conceptual references
- Clean architecture: no circular dependencies, dependency injection preferred

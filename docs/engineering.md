---
title: Engineering
nav_order: 7
has_children: true
description: Implementation notes, numerical benchmarks and regression tests.
---

# Engineering

- [Engineering notes](engineering-notes.md): numerical methods and pipeline behavior.
- [Compute worker lifecycle](mission-control-worker-lifecycle.md): process guarantees and tests.
- [CUDA kernel benchmark](benchmarks/fused_kernel_evaluation.md): measurements and backend decisions.
- [GPU regression testing](gpu-regression/README.md): procedure and recorded runs.
- [Reference tolerance audit](reference-tolerance-audit.md): EOP pinning, comparison budgets and provenance.
- [E1–E4 acceptance review](acceptance-e1-e4.md): tested revision, resolved findings and evidence limits.

## Workflow triggers

| Change | Automatic workflows |
|---|---|
| English Markdown in `docs/` | Pages build and translation drafts. |
| README | Translation drafts. |
| `translations/` drafts | No compute tests or repeat translation. |
| Runtime code, tests or compute configuration | CPU CI; GPU CI still requires its PR label or a manual run. |

CI skips documentation-only pushes and PRs. Benchmark JSON under
`docs/benchmarks/` remains tested because some suites read it as regression
input. Pages also rebuilds when its source/API inputs change; translation
reruns when its scripts, model requirements or workflow change. Manual runs
remain available regardless of changed paths.

Pages builds and uploads in one job, then deploys in a dependent job after
checking the exact uploaded artifact ID. Metadata checks make up to six
attempts, five seconds apart; missing or expired artifacts fail before deployment.

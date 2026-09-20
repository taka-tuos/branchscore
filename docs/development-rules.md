# Development rules

- **Graph-first:** express tensor work as ggml graphs; do not move tensor
  computations to host C++ without a reason.
- **Backend-aware:** track tensor placement, ownership, copy paths, and
  lifetimes explicitly.
- **Sequential-first:** validate `1 model / 1 backend / 1 request / N options`
  before separation, batching, workers, or scheduling.
- **Observable:** measure Vision, Prefill, Logit, copy, and synchronization
  times independently when those stages exist.
- **No premature abstraction:** choose the smallest structure that supports
  the current phase. Do not introduce a general framework for hypothetical
  distributed execution.
- **Reference, do not wrap:** use llama.cpp to learn tensor layouts, model
  handling, graph construction, KV/cache behavior, and backend usage; keep
  this project independent.
- **Research traceability:** record inspected upstream revisions/commit hashes
  and concrete findings in the appropriate research and phase notes.

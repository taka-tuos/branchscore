# Discarded placement pilot

These E4B F32 reference runs used n_gpu_layers=99 and the CUDA device without
an explicit tensor buffer override. The logs show 43/43 layer offload but
2,208 MiB of CPU-mapped embedding weights. They are excluded from report.json.
The last run was interrupted while correcting this placement mismatch.
The top-level reference sources add the explicit `.*` CUDA buffer override;
all reported reference results use the corrected build.

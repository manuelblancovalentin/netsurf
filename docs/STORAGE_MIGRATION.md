# Storage migration record — 2026-10-22

The migration used same-filesystem atomic renames for repository datasets and
generated collections. Directory device/inode pairs were identical before and
after the moves, so no large collection was copied and no existing shared
dataset was overwritten.

| Original repository path | External destination | Pre/post device:inode | Size observed |
|---|---|---:|---:|
| `datasets/` | `/ml/datasets/netsurf-legacy/repository-datasets-2026-10-22/` | `2304:481525882` | 263 GB |
| `benchmarks/` | `/ml/experiments/netsurf/legacy/benchmarks/` | `2304:481525889` | 8.0 GB |
| `benchmarks_old/` | `/ml/experiments/netsurf/legacy/benchmarks_old/` | `2304:481521761` | 23 GB |
| `syn/` | `/ml/experiments/netsurf/hardware/syn-legacy/` | `2304:481513605` | 5.4 GB |
| `v2_runs/` | `/ml/experiments/netsurf/runs/` | `2304:15875090` | 15 MB |

Ignored root-level symlinks retain convenient historical path access. They are
not a library API and are not committed.

## Dataset conflicts

Existing `/ml/datasets/{AutoMPG,SmartPixel,downloads,speech_commands,toyadmos}`
entries were left untouched. Same-name repository entries had different inode
identities. In particular, repository `toyadmos` reported 252 GB while the
shared namesake reported 64 GB with an unreadable subtree, so equivalence was
not claimed. The entire repository collection remains separately namespaced
until provenance is resolved.

The 11.5 MB Keras MNIST cache was copied to `/ml/datasets/mnist/mnist.npz` and
verified before retaining the original cache. Both hashes are
`731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.

## Key artifact verification

- MNIST-lite checkpoint: `ba48aee8adbe7d04092d74f41958ec99112ae3e5f88ed56552ea9d1c4dba3006`.
- Definitive scientific payload hash recorded by the finalized milestone:
  `07f4e88d7a36663fb120fafb97c28b80bab5dd3be13b91700755a4784b30bc1b`.

Technology libraries and historical hardware outputs remain inside the private
external hardware collection. No proprietary file was added to Git.

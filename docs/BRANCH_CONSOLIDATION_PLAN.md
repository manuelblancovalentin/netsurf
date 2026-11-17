# Branch and worktree consolidation plan

Status: **proposal, waiting for the owner's written approval.** Nothing in this
plan has been run. No branch, tag, worktree, remote or file outside this
document was changed while writing it.

Audit date: 2026-11-17, repository `/projects/netsurf/manuelbv` on `v2` at
`db5e2f7`.

## 1. Findings that change the plan

1. **Run outputs live inside the worktrees.** Most of the 37 worktrees under
   `workspace/implementation-worktrees/` hold ignored run outputs, about 340 GB
   in total (largest: `w8a8-ber-stress` 138 GB, `paper1-runK-retrained-engine`
   46 GB, `transformer-integration` 41 GB). The Paper 1 run registry
   (`paper1-tecs-revision-v1/workspace/run_registry/`, 88 entries), the
   provenance file and the manuscript sources name these folders by absolute
   path (for example 72 references to `m97_ber`, 38 to `m12-fkeras-multibench`,
   26 to `reliability-db`). **`git worktree remove` must not be used**: it
   deletes ignored files, which would delete those outputs and break the paths.
   This plan retires a worktree by detaching it in place (files and paths stay
   exactly where they are) and deleting only the branch name.
2. **One cited commit survives only in the reflog.** `workspace/runs/numpy_pgcs_exact/…/manifest.json`
   names `cccdca8461ffc7526ac5899866213b8f0ebcb8c5` ("Complete schema 4 search
   diagnostics"). It was amended away on `v2` on 2026-10-25 and no branch,
   tag or remote ref contains it. Git keeps unreachable reflog entries for 30
   days by default, so after about **2026-11-24** any `git gc` (including an
   automatic one) can delete it. Phase 0 tags it; this is the one urgent step.
   Seven other reflog-only commits exist (`095f39a`, `10f4ef2`, `403c62b`,
   `a7a1384`, `b189eb8`, `bd39644`, `f6a7922`); no run output or delivery
   names them, so they are left to expire.
3. **All branches are already on GitHub.** Between 2026-11-17 00:28 and 00:35
   this repository pushed `v2` to `origin` (`manuelblancovalentin/netsurf`) and
   every local branch, plus `v2` as `main`, to the remote `kona`, which is
   `https://github.com/Memik-Lab/netsurf.git` (41 remote branches). The local
   `paper1-tecs-revision-v1` is one commit ahead of `kona/paper1-tecs-revision-v1`
   (`bb74dbc` vs `76e9270`). This plan does not push and does not delete
   anything on GitHub; remote cleanup is a separate decision (section 6).
4. **`~/netsurf` is not a separate clone.** It is a symlink to
   `/projects/netsurf/manuelbv` (created 2026-11-17 00:33). There is only one
   repository; `kona` is the GitHub remote, not a path to this folder.
5. **`v2` has no tracked modifications.** `git status` on `v2` shows only the
   untracked `repo/`. The dirty handoff state recorded in
   `docs/LEGACY_MAPPING.md` (SHA-256 `769f4980…`) is already represented in the
   canonical migration commit. Nothing here touches the `v2` working tree until
   Phase 3, and Phase 3 only fast-forwards it after review.
6. **Every non-`v2` branch forks from the same point**, `e3209f7` (the
   `m97-integration` merge, 2026-10-31). `v2` is that commit plus 11 commits of
   task-log files and configs, so all branches are 11 commits behind `v2` and
   none conflicts with `v2` itself.
7. **Live work right now.** `paper1-tecs-revision-v1` has 8 uncommitted figure
   files and about 20 processes with their working folder inside it (another
   session is editing the manuscript). `w8a8-ber-stress` has an idle tmux
   server `netsurf-w8a8-high-ber` (sessions `pilot`, `pilot-resume`, created
   2026-11-09). Task `netsurf-manuelbv-0044` (language models to 16 draws) is
   still `doing`.

## 2. The `repo/` folder

`/projects/netsurf/manuelbv/repo` appeared on 2026-11-17 at 00:27. It is an
empty clone of `git@github.com:Memik-Lab/netsurf.git`: branch `main`, no
commits, only a `.git` folder (last modified 01:21). It was cloned one minute
before the pushes in finding 3, so it does not contain them.

Effect on the task log: `/usr/local/bin/kona-task` (`/opt/kona-project/kona_tasks.py`,
lines 271-272) uses `<your folder>/repo/.kona/tasks` whenever `repo/.git`
exists and `repo` is not a symlink, and falls back to `<your folder>/.kona/tasks`
otherwise. The website compiler uses the same rule (line 595-596: "a git work
tree keeps its tasks inside the repository"). Because `repo/.git` now exists,
`kona-task list` in this folder prints `0 tasks, 0 problems`: none of the 50
committed tasks in `.kona/tasks/` (on `v2`) are seen, and any new
`kona-task new` would write into `repo/.kona/tasks`, outside the netsurf
repository. The website presumably shows no tasks for this folder either (same
code; not checked in the browser).

`repo/` is also not ignored by `.gitignore`, so it shows as `?? repo/` on `v2`.

Options (owner's choice; nothing done):

- **A. Remove the empty clone** (it holds no commits and no files):
  `rm -rf /projects/netsurf/manuelbv/repo`. `kona-task` then reads
  `.kona/tasks` again. Recommended if the clone was an accident or a test.
- **B. Adopt the kona layout** (repository at `<your folder>/repo`). This moves
  the whole repository and its 37 worktrees, which breaks the absolute paths
  in run manifests, the run registry and the manuscript sources (finding 1).
  Not recommended before Paper 1 is submitted.

The task for this consolidation should be created with `kona-task new` only
after this is settled, so that it lands in `.kona/tasks/`.

## 3. Inventory

Columns: commits on the branch that `v2` lacks ("ahead"); first and last date
of those commits; whether `git branch --merged v2` lists it; which branch it
grew from (from `git merge-base` and ancestry); worktree folder under
`workspace/implementation-worktrees/`; uncommitted tracked/untracked changes
(`git status --porcelain`, ignored files excluded); size of the worktree's
ignored `workspace/` folder and number of run folders in it; task in
`.kona/tasks/`.

"Named by" lists outputs that depend on the branch's code: commits named in
run manifests, provenance or registry JSON files (scan of 1,534 such files
under `workspace/`, including every worktree), Paper 1 delivery READMEs, and
absolute worktree paths in the Paper 1 registry and manuscript sources.

### 3.1 Already merged into `v2`

All forked from the previous `v2` milestone tip; 0 ahead.

| Branch | Last commit | Worktree | Changes | Outputs in worktree | Named by | Task |
|---|---|---|---|---|---|---|
| `m93-multibenchmark` | 2026-10-29 | `m93` | 0 | 36 KB, 0 runs | — | 0013 |
| `toycar-conformance-mechanisms` | 2026-10-29 | `toycar-audit` | 0 | 2.0 GB, 1 run | its own runs (`defaf99`, `e65a5f2`) | 0014 |
| `m95-conditional-damage` | 2026-10-30 | `m95` | 0 | 256 KB, 1 | own runs | 0015 |
| `m953-same-distribution` | 2026-10-30 | `m953` | 0 | 2.0 MB, 1 | own runs | 0016 |
| `m954-protection-context` | 2026-10-30 | `m954` | 0 | 11 MB, 1 | own runs | 0016 |
| `m955-paired-swap` | 2026-10-30 | `m955` | 0 | 14 MB, 1 | own runs | 0016 |
| `m956-publication-preflight` | 2026-10-30 | `m956` | 0 | 198 MB, 4 | own runs | 0016 |
| `m957-frozen-mask-replication` | 2026-10-30 | `m957` | 0 | 231 MB, 2 | own runs | 0016 |
| `m96-cross-pool` | 2026-10-30 | `m96` | 0 | 856 MB, 8 | manuscript sources (24 paths) | 0017 |
| `m97-ber-campaign` | 2026-10-31 | `m97_ber` | 0 | 1.4 GB, 6 | manuscript sources (72 paths) | 0018 |
| `m97-fkeras-adapters` | 2026-10-31 | `m97` | 0 | 1.5 GB, 13 | own runs | 0018 |
| `m97-integration` | 2026-10-31 | `m97-integration` | 0 | 8 KB, 0 | fork point of every other branch | 0018 |
| `master` | 2025-04-21 | (none) | — | — | historical base (`docs/LEGACY_MAPPING.md`), GitHub default branch | — |

### 3.2 Contained in a later branch (superseded)

All forked from `e3209f7`.

| Branch | Ahead | First – last | Contained in | Worktree | Changes | Outputs in worktree | Named by | Task |
|---|---|---|---|---|---|---|---|---|
| `m98-conditional-adapters` | 2 | 10-31 – 10-31 | `m99` ⊂ `m910` ⊂ `m11-conditional-scaling` | `m98` | 0 | 407 MB, 8 | runs in `m98`, `m99`; manuscript (8 paths) | 0019 |
| `m99-toycar-conditional` | 4 | 10-31 – 11-01 | `m910` ⊂ `m11` | `m99-toycar-conditional` | 0 | 932 MB, 8 | own runs; manuscript (8) | 0020 |
| `m910-interaction-attribution` | 6 | 10-31 – 11-01 | `m11-conditional-scaling` | `m910-interaction-attribution` | 0 | 257 MB, 8 | own runs, `m11` runs; manuscript (1) | 0021 |
| `kws-ber-transition-pilot` | 2 | 11-09 – 11-10 | `m12-kws-transition-validation` | `kws-ber-transition-pilot` | 0 | 4.1 GB, 2 | own runs (`5491d8a`) | 0033 |
| `codex/transformer-integration` | 35 | 11-02 – 11-04 | `codex/w8a8-fkeras-feasibility`, `m12-w8a8-*`, `m12-pythia-*` | `transformer-integration` | 0 | 41 GB, 53 | own runs; registry (1), manuscript (5) | 0024 |
| `m12-w8a8-ber-stress` | 41 | 11-02 – 11-09 | `m12-w8a8-followup` ⊂ `m12-pythia-wide-confirm`; also `m12-pythia-recovery` | `w8a8-ber-stress` | 0 | 138 GB, 8 | registry (3), manuscript (19); idle tmux server | 0026 |
| `m12-w8a8-followup` | 43 | 11-02 – 11-10 | `m12-pythia-wide-confirm` | `w8a8-followup` | 0 | 27 GB, 5 | manuscript (4) | 0027 |
| `m12-fkeras-multibench` | 4 | 11-04 – 11-04 | `m12-priorifi-crossbench` and the whole `paper1-run*` chain | `m12-fkeras-multibench` | 0 | 16 GB, 8 | own runs (`03033ca`, `c27c346`); registry (5), manuscript (38) | 0031 |
| `paper1-engineering-reconstruction` | 2 | 11-02 – 11-02 | `paper1-expanded-benchmarks` | `paper1-engineering` | 0 | 32 MB, 4 | export `NetSuRF_Paper1_Software_Review_2026-11-02` (`589aaa6`) | 0034 |
| `paper1-complete-review-v1` | 5 | 11-10 – 11-10 | `paper1-tecs-revision-v1` | `paper1-complete-review-v1` | 0 | 31 MB, 0 | deliveries v1–v4 (`70c5fc8`, `fb36037`, `ee26501`, `3c517dc`); `delivery_v4` must stay untouched | 0036 |
| `paper1-runA-tflite-confirm` | 8 | 11-04 – 11-11 | `runK` ⊂ `runP` ⊂ `runR-rate-aware` ⊂ `runR-v2-validated` | `paper1-runA-tflite-confirm` | 0 | 14 GB, 6 | Run A/A2/I bundles (`1d11df2`, `c21b354`, `d368969`); registry (11), manuscript (6) | 0037 |
| `paper1-runK-retrained-engine` | 11 | 11-04 – 11-11 | `runP` ⊂ … ⊂ `runR-v2-validated` | `paper1-runK-retrained-engine` | 0 | 46 GB, 9 | Run K bundles (`9b57733`); registry (24), manuscript (16) | 0038 |
| `paper1-runP-mean-variance` | 26 | 11-04 – 11-12 | `runR-rate-aware` ⊂ `runR-v2-validated` | `paper1-runP-mean-variance` | 0 | 8 KB, 0 | Paper 1 data (`feb8db1`, `c442d09`) | 0039 |
| `paper1-runR-rate-aware` | 38 | 11-04 – 11-12 | `runR-v2-validated` | `paper1-runR-rate-aware` | 0 | 12 KB, 0 | manuscript (2) | 0040 |

### 3.3 Live (Paper 1)

| Branch | Ahead | First – last | Grew from | Worktree | Changes | Outputs | Named by | Tasks |
|---|---|---|---|---|---|---|---|---|
| `paper1-tecs-revision-v1` | 152 | 11-10 – 11-17 | `paper1-complete-review-v1` | `paper1-tecs-revision-v1` | **8** (figures, in progress) | 13 GB, 157 runs, run registry | deliveries v5–v10 (in the worktree) and v11–v22 (`workspace/paper1_review/`), e.g. v22 = `bb74dbc`; Paper 1 data tables | 0042, 0045, 0048–0050 |
| `paper1-runR-v2-validated` | 90 | 11-04 – 11-16 | `paper1-runR-rate-aware` | `paper1-runR-v2-validated` | 0 | 24 KB (runs are written into the TECS worktree) | 24 of its commits are named in run outputs or the registry of the TECS worktree | 0041, 0043, 0044 |

### 3.4 Unmerged tips with finished work (not contained in any other branch)

| Branch | Ahead (own) | First – last | Grew from / contains | Worktree | Changes | Outputs in worktree | Named by | Task |
|---|---|---|---|---|---|---|---|---|
| `m11-conditional-scaling` | 9 (3) | 10-31 – 11-02 | contains `m910`, `m99`, `m98` | `m11-conditional-scaling` | 0 | 1.5 GB, 44 | own runs (`10cb2f2`, `65bb127`, `78459d8`, `a255c3d`); manuscript (2) | 0022 |
| `m12-reliability-database` | 21 | 11-03 – 11-08 | `e3209f7` | `reliability-db` | 0 | 9.2 GB, 3 | own runs (`bb33fc3`, `d964c64`); registry (2), manuscript (26) | 0030 |
| `m12-pythia-wide-confirm` | 45 (4) | 11-02 – 11-10 | contains `m12-w8a8-followup` | `pythia-wide-confirm` | 0 | 23 GB, 2 | registry (1), manuscript (9) | 0029 |
| `m12-pythia-recovery` | 44 (3) | 11-02 – 11-10 | contains `m12-w8a8-ber-stress`; shares 41 commits with `wide-confirm` | `pythia-recovery` | 0 | 90 MB, 9 | — | 0028 |
| `codex/w8a8-fkeras-feasibility` | 36 (1) | 11-02 – 11-04 | contains `codex/transformer-integration` | `w8a8-fkeras-feasibility` | 0 | 104 KB, 0 | — | 0025 |
| `pythia-bf16-storage` | 10 (1) | 11-02 – 11-02 | forks from `codex/transformer-integration` at `2ca15ac` (9 shared) | `pythia-bf16-storage` | 0 | 87 MB, 0 | — | 0023 |
| `m12-kws-transition-validation` | 3 | 11-09 – 11-10 | contains `kws-ber-transition-pilot` | `kws-transition-validation` | 0 | 2.2 GB, 4 | own runs (`7e5dd44`); registry (1), manuscript (7) | 0033 |
| `m12-priorifi-crossbench` | 6 | 11-04 – 11-09 | contains `m12-fkeras-multibench` | `m12-priorifi-crossbench` | 0 | 264 KB, 9 | — | 0032 |
| `m12-priorifi-probing` | 1 | 11-08 – 11-08 | `e3209f7` | `m12-priorifi` | 0 | 120 KB, 2 | — | 0032 |
| `paper1-expanded-benchmarks` | 10 | 11-02 – 11-03 | contains `paper1-engineering-reconstruction` | `paper1-expanded` | 0 | 123 MB, 2 | five exports of 2026-11-02/03 (`d7adfb4`, `c8b2f77`, `2512bb6`, `371fe62`, `64fc519`) | 0035 |

Commits named by outputs that are not in this repository: `85b00d6` and
`4c865dc` (`workspace/imported-ml-experiments`, from the deleted
`codex/netsurf-v2` branch, already recorded as lost in `docs/LEGACY_MAPPING.md`);
the `revision`/`commit` fields in `config/benchmark_expansion_registry.json`,
`config/transformer_runtime_registry.json` and `workspace/quest_transfer_v1/*.lock.json`
name upstream repositories (MLCommons Tiny, Hugging Face model revisions), not
NetSuRF commits. Every other named commit is reachable from `v2` or a local
branch, except `cccdca8` (finding 2).

### 3.5 Merge test

Tested without changing any ref (`git merge-tree --write-tree`; the test left
unreferenced objects in `.git/objects` that a later `git gc` removes):

- Every tip merges into `v2` on its own without conflicts.
- In the order of Phase 3 below, items 1–6 merge without conflicts, and
  `paper1-tecs-revision-v1` and `paper1-expanded-benchmarks` also merge cleanly
  on top of them.
- Conflicts (all in source files, all pairwise with `m11-conditional-scaling`
  or between the two PrioriFI branches):
  - `m12-kws-transition-validation` × `m11`: `netsurf/cross_pool.py`, `netsurf/cross_pool_analysis.py`
  - `m12-priorifi-crossbench` × `m11`: `netsurf/benchmarks/capabilities.py`
  - `paper1-runR-v2-validated` × `m11`: `netsurf/benchmarks/capabilities.py`
  - `m12-priorifi-crossbench` × `m12-priorifi-probing`: `netsurf/priorifi.py`, `tests/test_priorifi.py`

## 4. Classification

| Class | Branches | Action |
|---|---|---|
| Merged, safe to delete | the 12 in 3.1 except `master` | tag, detach worktree in place, delete branch |
| Superseded | the 14 in 3.2 | tag, detach worktree in place, delete branch (the containing tip keeps every commit reachable; the tag keeps the name) |
| Live | `paper1-tecs-revision-v1`, `paper1-runR-v2-validated` | keep branches and worktrees; snapshot tag now; merge after the TECS revision is submitted (Phase 4) |
| Finished, to merge | `m11-conditional-scaling`, `m12-reliability-database`, `m12-pythia-wide-confirm`, `m12-pythia-recovery`, `codex/w8a8-fkeras-feasibility`, `pythia-bf16-storage`, `m12-kws-transition-validation`, `m12-priorifi-crossbench` | merge into `v2` through an integration branch (Phase 3), then retire like the merged ones |
| Needs a decision | `m12-priorifi-probing`, `paper1-expanded-benchmarks`, `master`, remote branches on GitHub, `repo/` | see section 6 |

## 5. Target layout

After all phases:

- **Branches:** `v2` (main line, all finished milestone code), `master`
  (historical base, untouched), and until Paper 1 is submitted
  `paper1-tecs-revision-v1` and `paper1-runR-v2-validated`.
- **Tags (local, annotated, never pushed by this plan):** `archive/<branch>` for
  every retired branch at its exact tip (36 in Phase 0, 38 after Phase 4, so the names in the task log,
  manifests and notes still resolve); `provenance/numpy-pgcs-exact-cccdca8`;
  `snapshot/<branch>/2026-11-17` for the two live branches; later
  `paper1/delivery-vNN` for each delivery.
- **Worktrees:** the two live ones stay registered. The other 35 stay on disk
  at the same paths, detached at their tag, so every path in the registry,
  manifests and manuscript still works. Optionally (Phase 5) they are
  unregistered from git but kept as plain folders.
- **Backup:** one git bundle of all refs and tags before anything is deleted.

### Phase 0 — protect (no deletion; reversible with `git tag -d`)

Run from `/projects/netsurf/manuelbv`. Do this before 2026-11-24 because of
`cccdca8`.

```bash
cd /projects/netsurf/manuelbv

# 0.1 The reflog-only commit named by workspace/runs/numpy_pgcs_exact
git tag -a provenance/numpy-pgcs-exact-cccdca8 cccdca8461ffc7526ac5899866213b8f0ebcb8c5 \
  -m "Commit named by workspace/runs/numpy_pgcs_exact/20261025T124334.157894Z-cc2cbb50/manifest.json; amended away on v2 on 2026-10-25 and otherwise reachable only from the reflog. Tagged per docs/BRANCH_CONSOLIDATION_PLAN.md."

# 0.2 One annotated tag per branch tip
t() {  # t <branch> <task ids> <worktree folder>
  git tag -a "archive/$1" "refs/heads/$1" -m "Branch $1 at $(git rev-parse "refs/heads/$1"), last commit $(git log -1 --format=%cs "refs/heads/$1"). Tasks: $2. Worktree: workspace/implementation-worktrees/$3. Kept for provenance: run manifests, registries and deliveries name commits on this line. Tagged per docs/BRANCH_CONSOLIDATION_PLAN.md."
}
t m93-multibenchmark                 netsurf-manuelbv-0013 m93
t toycar-conformance-mechanisms      netsurf-manuelbv-0014 toycar-audit
t m95-conditional-damage             netsurf-manuelbv-0015 m95
t m953-same-distribution             netsurf-manuelbv-0016 m953
t m954-protection-context            netsurf-manuelbv-0016 m954
t m955-paired-swap                   netsurf-manuelbv-0016 m955
t m956-publication-preflight         netsurf-manuelbv-0016 m956
t m957-frozen-mask-replication       netsurf-manuelbv-0016 m957
t m96-cross-pool                     netsurf-manuelbv-0017 m96
t m97-ber-campaign                   netsurf-manuelbv-0018 m97_ber
t m97-fkeras-adapters                netsurf-manuelbv-0018 m97
t m97-integration                    netsurf-manuelbv-0018 m97-integration
t m98-conditional-adapters           netsurf-manuelbv-0019 m98
t m99-toycar-conditional             netsurf-manuelbv-0020 m99-toycar-conditional
t m910-interaction-attribution       netsurf-manuelbv-0021 m910-interaction-attribution
t m11-conditional-scaling            netsurf-manuelbv-0022 m11-conditional-scaling
t pythia-bf16-storage                netsurf-manuelbv-0023 pythia-bf16-storage
t codex/transformer-integration      netsurf-manuelbv-0024 transformer-integration
t codex/w8a8-fkeras-feasibility      netsurf-manuelbv-0025 w8a8-fkeras-feasibility
t m12-w8a8-ber-stress                netsurf-manuelbv-0026 w8a8-ber-stress
t m12-w8a8-followup                  netsurf-manuelbv-0027 w8a8-followup
t m12-pythia-recovery                netsurf-manuelbv-0028 pythia-recovery
t m12-pythia-wide-confirm            netsurf-manuelbv-0029 pythia-wide-confirm
t m12-reliability-database           netsurf-manuelbv-0030 reliability-db
t m12-fkeras-multibench              netsurf-manuelbv-0031 m12-fkeras-multibench
t m12-priorifi-probing               netsurf-manuelbv-0032 m12-priorifi
t m12-priorifi-crossbench            netsurf-manuelbv-0032 m12-priorifi-crossbench
t kws-ber-transition-pilot           netsurf-manuelbv-0033 kws-ber-transition-pilot
t m12-kws-transition-validation      netsurf-manuelbv-0033 kws-transition-validation
t paper1-engineering-reconstruction  netsurf-manuelbv-0034 paper1-engineering
t paper1-expanded-benchmarks         netsurf-manuelbv-0035 paper1-expanded
t paper1-complete-review-v1          netsurf-manuelbv-0036 paper1-complete-review-v1
t paper1-runA-tflite-confirm         netsurf-manuelbv-0037 paper1-runA-tflite-confirm
t paper1-runK-retrained-engine       netsurf-manuelbv-0038 paper1-runK-retrained-engine
t paper1-runP-mean-variance          netsurf-manuelbv-0039 paper1-runP-mean-variance
t paper1-runR-rate-aware             netsurf-manuelbv-0040 paper1-runR-rate-aware

# 0.3 Snapshots of the two live branches (they keep moving; these mark today's tips)
git tag -a snapshot/paper1-runR-v2-validated/2026-11-17 refs/heads/paper1-runR-v2-validated \
  -m "Snapshot of the live Run R v2 branch on 2026-11-17 (tasks 0041, 0043, 0044)."
git tag -a snapshot/paper1-tecs-revision-v1/2026-11-17 refs/heads/paper1-tecs-revision-v1 \
  -m "Snapshot of the live TECS revision branch on 2026-11-17; contains the commits named by deliveries v1-v22."

# 0.4 Backup bundle of every branch and tag (git objects total under 100 MB; outside workspace/runs)
mkdir -p /projects/netsurf/manuelbv/workspace/git_backups
git bundle create /projects/netsurf/manuelbv/workspace/git_backups/netsurf-refs-2026-11-17.bundle --branches --tags
git bundle verify /projects/netsurf/manuelbv/workspace/git_backups/netsurf-refs-2026-11-17.bundle
sha256sum /projects/netsurf/manuelbv/workspace/git_backups/netsurf-refs-2026-11-17.bundle
```

### Phase 1 — check (read-only)

```bash
cd /projects/netsurf/manuelbv
# every branch tip has an identical tag (prints nothing when all is well)
for b in $(git for-each-ref --format='%(refname:short)' refs/heads | grep -v -x -e v2 -e master -e branch-consolidation-plan); do
  case $b in paper1-tecs-revision-v1|paper1-runR-v2-validated) continue;; esac
  [ "$(git rev-parse "archive/$b^{commit}")" = "$(git rev-parse "refs/heads/$b")" ] || echo "MISMATCH $b"
done
git merge-base --is-ancestor cccdca8461ffc7526ac5899866213b8f0ebcb8c5 provenance/numpy-pgcs-exact-cccdca8 && echo "cccdca8 protected"
git tag -l 'archive/*' | wc -l   # expect 36
```

### Phase 2 — retire merged and superseded branches

Files, ignored run outputs and paths stay exactly where they are; only the
branch name goes away (its tag remains). Each call refuses if the tag is
missing or different, if the worktree has changes, or if the branch is not
contained in `v2` or in the named containing branch.

```bash
cd /projects/netsurf/manuelbv
WT=/projects/netsurf/manuelbv/workspace/implementation-worktrees
retire() {  # retire <branch> <worktree folder> <branch or ref that must contain it>
  b=$1; w=$WT/$2; keep=$3
  [ "$(git rev-parse -q --verify "archive/$b^{commit}")" = "$(git rev-parse "refs/heads/$b")" ] || { echo "SKIP $b: tag missing or different"; return 1; }
  [ -z "$(git -C "$w" status --porcelain)" ] || { echo "SKIP $b: $w has changes"; return 1; }
  git merge-base --is-ancestor "refs/heads/$b" "$keep" || { echo "SKIP $b: not contained in $keep"; return 1; }
  git -C "$w" switch --detach "archive/$b" && git branch -D "$b"
}
# 2a. merged into v2
retire m93-multibenchmark            m93              v2
retire toycar-conformance-mechanisms toycar-audit     v2
retire m95-conditional-damage        m95              v2
retire m953-same-distribution        m953             v2
retire m954-protection-context       m954             v2
retire m955-paired-swap              m955             v2
retire m956-publication-preflight    m956             v2
retire m957-frozen-mask-replication  m957             v2
retire m96-cross-pool                m96              v2
retire m97-ber-campaign              m97_ber          v2
retire m97-fkeras-adapters           m97              v2
retire m97-integration               m97-integration  v2
# 2b. superseded by a later branch
retire m98-conditional-adapters          m98                          m11-conditional-scaling
retire m99-toycar-conditional            m99-toycar-conditional       m11-conditional-scaling
retire m910-interaction-attribution      m910-interaction-attribution m11-conditional-scaling
retire kws-ber-transition-pilot          kws-ber-transition-pilot     m12-kws-transition-validation
retire codex/transformer-integration     transformer-integration      m12-pythia-wide-confirm
retire m12-w8a8-ber-stress               w8a8-ber-stress              m12-pythia-wide-confirm
retire m12-w8a8-followup                 w8a8-followup                m12-pythia-wide-confirm
retire m12-fkeras-multibench             m12-fkeras-multibench        paper1-runR-v2-validated
retire paper1-engineering-reconstruction paper1-engineering           paper1-expanded-benchmarks
retire paper1-complete-review-v1         paper1-complete-review-v1    paper1-tecs-revision-v1
retire paper1-runA-tflite-confirm        paper1-runA-tflite-confirm   paper1-runR-v2-validated
retire paper1-runK-retrained-engine      paper1-runK-retrained-engine paper1-runR-v2-validated
retire paper1-runP-mean-variance         paper1-runP-mean-variance    paper1-runR-v2-validated
retire paper1-runR-rate-aware            paper1-runR-rate-aware       paper1-runR-v2-validated
```

Detaching does not change any file, so the idle tmux sessions in
`w8a8-ber-stress` are not affected. Closing them
(`tmux -L netsurf-w8a8-high-ber kill-server`) is the owner's call.

Undo for one branch: `git branch <branch> archive/<branch>` and
`git -C <worktree> switch <branch>`.

### Phase 3 — merge the finished milestone work into `v2`

Done on a new integration branch in its own worktree, tested, reviewed, and only
then fast-forwarded into `v2`. Merge commits (`--no-ff`) keep every original
commit hash, so every manifest stays valid.

```bash
cd /projects/netsurf/manuelbv
git worktree add -b integration/m11-m12 workspace/implementation-worktrees/integration-m11-m12 v2
cd workspace/implementation-worktrees/integration-m11-m12

git merge --no-ff m11-conditional-scaling        -m "Merge M9.8-M11 conditional scaling (task 0022)"
git merge --no-ff m12-reliability-database       -m "Merge reliability database v3 builder (task 0030)"
git merge --no-ff m12-pythia-wide-confirm        -m "Merge transformer integration, W8A8 stress/follow-up and Pythia wide-BER confirmation (tasks 0024, 0026, 0027, 0029)"
git merge --no-ff m12-pythia-recovery            -m "Merge Pythia recovery (task 0028)"
git merge --no-ff codex/w8a8-fkeras-feasibility  -m "Merge W8A8 FKeras feasibility gate (task 0025)"
git merge --no-ff pythia-bf16-storage            -m "Merge exact BF16 storage faults (task 0023)"
# expected conflict: netsurf/cross_pool.py, netsurf/cross_pool_analysis.py (with m11)
git merge --no-ff m12-kws-transition-validation  -m "Merge KWS high-BER transition pilot and validation plan (task 0033)"
# expected conflict: netsurf/benchmarks/capabilities.py (with m11)
git merge --no-ff m12-priorifi-crossbench        -m "Merge matched FKeras plans and PrioriFI cross-benchmark gate (tasks 0031, 0032)"

python -m pytest -q          # in the activated netsurf environment (proj netsurf)
```

Each conflict needs a person or a reviewed agent session to resolve it so that
both sides' behavior is kept, with the tests of both branches passing; this
plan does not prescribe the resolution. After the owner reviews the
integration branch:

```bash
cd /projects/netsurf/manuelbv               # the v2 checkout; it has no tracked changes
git merge --ff-only integration/m11-m12
# the merged tips are now contained in v2; retire them with Phase 2's retire (same shell, or define WT and retire again)
retire m11-conditional-scaling        m11-conditional-scaling   v2
retire m12-reliability-database       reliability-db            v2
retire m12-pythia-wide-confirm        pythia-wide-confirm       v2
retire m12-pythia-recovery            pythia-recovery           v2
retire codex/w8a8-fkeras-feasibility  w8a8-fkeras-feasibility   v2
retire pythia-bf16-storage            pythia-bf16-storage       v2
retire m12-kws-transition-validation  kws-transition-validation v2
retire m12-priorifi-crossbench        m12-priorifi-crossbench   v2
git worktree remove workspace/implementation-worktrees/integration-m11-m12   # holds no run outputs
git branch -d integration/m11-m12
```

### Phase 4 — Paper 1, after the TECS revision is submitted

Not before: another session is editing `paper1-tecs-revision-v1` and task 0044
is open.

```bash
cd /projects/netsurf/manuelbv
# tag each delivery at the commit its README names (v1-v22 so far), e.g.
git tag -a paper1/delivery-v22 bb74dbc -m "Paper 1 delivery v22 (workspace/paper1_review/delivery_v22_lm_final)"
git tag -a paper1/tecs-submitted refs/heads/paper1-tecs-revision-v1 -m "Paper 1 TECS revision as submitted"
git tag -a archive/paper1-runR-v2-validated refs/heads/paper1-runR-v2-validated -m "Run R v2 branch at Paper 1 submission (tasks 0041, 0043, 0044)"
git tag -a archive/paper1-tecs-revision-v1 refs/heads/paper1-tecs-revision-v1 -m "TECS revision branch at submission (tasks 0042, 0045, 0048-0050)"
```

Then, in a new integration worktree from `v2` as in Phase 3: merge
`paper1-runR-v2-validated` (expected conflict in
`netsurf/benchmarks/capabilities.py`) and `paper1-tecs-revision-v1` (merges
cleanly in the test; adds 441 files under `publications/paper1_review/`), run
the tests, review, fast-forward `v2`, and retire both branches with Phase 2's
`retire` (worktree folder stays: it holds 13 GB of runs and the run registry).

### Phase 5 — optional: unregister retired worktrees, keep the folders

Not recommended for any folder from which code may still be run: after
unregistering, git commands inside the folder reach the parent repository, so
a script that records `git rev-parse HEAD` would record `v2`'s commit instead
of the folder's. Detached worktrees (end of Phase 2) do not have this problem.

`git worktree list` would then show only live worktrees. The folders, their
checked-out files and run outputs stay; `workspace/*` is ignored by `v2`, so
they do not show up as untracked files. This is not needed for correctness.

```bash
cd /projects/netsurf/manuelbv
for w in m93 toycar-audit m95 m953 m954 m955 m956 m957 m96 m97_ber m97 m97-integration \
         m98 m99-toycar-conditional m910-interaction-attribution kws-ber-transition-pilot \
         transformer-integration w8a8-ber-stress w8a8-followup m12-fkeras-multibench \
         paper1-engineering paper1-complete-review-v1 paper1-runA-tflite-confirm \
         paper1-runK-retrained-engine paper1-runP-mean-variance paper1-runR-rate-aware; do   # add Phase 3 folders once retired
  d=workspace/implementation-worktrees/$w
  git -C "$d" rev-parse HEAD > "$d/.archived-worktree-commit" && rm "$d/.git"
done
git worktree prune -v
```

## 6. Decisions for the owner

1. **`repo/`**: remove the empty clone (option A) or plan a move to the kona
   layout (option B). Until then `kona-task` sees 0 tasks here.
2. **`m12-priorifi-probing`** (1 commit, "bounded PrioriFI probe scheduling
   reference") conflicts with `m12-priorifi-crossbench` in `netsurf/priorifi.py`
   and its test. Recommendation: keep it as tag `archive/m12-priorifi-probing`
   only, unless its scheduler is wanted in `v2`, then merge it after
   crossbench and reconcile the two versions.
3. **`paper1-expanded-benchmarks`** (older manuscript line, 2026-11-02/03,
   53 files under `publications/tecs_software_review_2026-11-02` plus
   `workflows/run_paper1_*.sh` and `paper1_evidence_audit.py`). The TECS
   revision rebuilt the manuscript from `paper1-complete-review-v1` and does not
   contain it. Recommendation: tag only (`archive/paper1-expanded-benchmarks`),
   retire with `retire paper1-expanded-benchmarks paper1-expanded archive/paper1-expanded-benchmarks`,
   and cherry-pick any script that is still used.
4. **Main line name.** `v2` stays the main line in this plan. GitHub
   `Memik-Lab/netsurf` already has `main` = `v2`; `origin`'s default is
   `master`. Renaming is optional and separate.
5. **GitHub remote branches.** All 41 branches pushed on 2026-11-17 remain on
   `Memik-Lab/netsurf`. Deleting them there is a push to a public repository
   and is not part of this plan; if wanted, the exact commands can be written
   once the local layout is final. `git fetch --prune` would also drop the local
   `kona/*` remote-tracking refs, which is why Phase 0 tags are the local
   guarantee.
6. **Paper 1 timing.** Phase 4 waits for the submission; confirm that this is
   the trigger.

## 7. Method

Commands used (all read-only, except unreferenced test-merge objects in
`.git/objects`): `git branch -vv`, `git worktree list --porcelain`,
`git branch --merged v2`, `git rev-list --count`, `git merge-base`
(`--is-ancestor` for every pair of branches), `git status --porcelain` per
worktree, `git reflog`, `git merge-tree --write-tree`, `du` on each worktree's
`workspace/`, a scan of every JSON file under `workspace/` whose name contains
manifest/provenance/registry/lock/metadata/run_info (1,534 files, up to 5 MB,
fields whose key contains commit/git/head/revision/sha), the README of each
Paper 1 delivery, a text search for absolute worktree paths in the Paper 1
registry and manuscript sources, `/proc/*/cwd` for running processes, and
`/opt/kona-project/kona_tasks.py`. Run-output dependency is therefore a lower
bound: an output that names its commit in a file not covered by that scan is
not listed, but such a commit is still protected because Phase 0 tags every
branch tip.

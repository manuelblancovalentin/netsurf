# Bibliography verification and publication status

The manuscript uses the full-method audit completed in
`docs/PAPER1_NOVELTY_AUDIT.md`, not abstracts alone. No connected Zotero library
was changed: its local API was unavailable. The supplied FCCM/EISFINN PDFs and
the pinned literature/source records provide the primary evidence below.

| BibTeX key | Status and verification source | Claim supported |
|---|---|---|
| `guo2025inpixel` | Published FCCM 2025 invited paper; supplied complete seven-page PDF, DOI 10.1109/FCCM62733.2025.00014, pp. 261–267. PDF SHA-256 `80b08d098e1f3adc9fbd347c67b745f597bf0e4bee13d560dafd7500f11b5cec`. | Own prior flow already described susceptibility, gradient/delta ranking and selective allocation. Not new v2 measurements. |
| `weng2024fkeras` | ACM JATS 1(3), article 15, DOI 10.1145/3665334; full author-hosted paper, §§3.1–3.2 / Eq. (3). | Hessian projection and tier-first selective protection. |
| `fkerasSource` | Public software snapshot `68e775c77670c9a1aab4e1b5295b523347d97013`, `fkeras/metrics/hessian.py`. | Effective additional eigenvalue and magnitude-before-sum; not a second peer-reviewed paper. |
| `weng2026priorifi` | ASPLOS 2026, DOI 10.1145/3779212.3790204; author-hosted full method and Algorithm 1. | Hessian-informed initial ordering followed by observed-degradation-driven injection prioritization. |
| `colucci2024eisfinn` | IOLTS 2024, DOI 10.1109/IOLTS60994.2024.10616075; supplied complete three-page PDF, SHA-256 `996060e82bc8e2d60cd9b20909b4bdd8dc923eb809a792a080c3edb6dbdd406f`. | Attribution-guided neuron-output/bit sampling; no claimed matched Bernoulli mask-risk experiment. |
| `rakin2019bfa` | ICCV 2019 CVF open-access paper, §3. | Quantized encoding-aware progressive bit search predates NetSuRF. |
| `catalan2025redundancy` | Journal of Supercomputing 81, article 183; online 2024, volume year 2025; DOI 10.1007/s11227-024-06693-7, full §§3–5. | Existing representation-dependent selective recovery; not free ideal mask suppression. |
| `alobaidi2026selectiveeccdataset` | Zenodo companion dataset/software, DOI 10.5281/zenodo.23054820; source archive checksum in first-gate audit. | Prior learned/rule-based allocation and contextual layer analysis. Peer-reviewed acceptance of the accompanying manuscript is **not established**. |
| `banbury2021mlperf` | NeurIPS 2021 Datasets and Benchmarks proceedings paper, linked primary PDF. | Benchmark suite/tasks, not this manuscript's cohort-specific scores. |
| `mlcommons2026tiny` | MLCommons public source snapshot `4addd0fa08d216e20637637874e084895f289da4`, matched to asset contracts. | Source protocol identity, not certification of a NetSuRF official submission. |
| `odonnell2014analysis` | Cambridge University Press 2014 book, DOI 10.1017/CBO9781139814782. | Standard Boolean/multilinear foundations, credited rather than claimed novel. |
| `rauzy2022reliability` | AltaRica Association 2022 book, ISBN 978-82-692273-2-1, author-hosted full text. | Established reliability/conditional-importance background. |

| `su2025rl` | Peer-reviewed Sensors 25(13), 4196 (2025), DOI 10.3390/s25134196; full method and experiments read at https://pmc.ncbi.nlm.nih.gov/articles/PMC12252199/, §§3–4. | Layer-bit DQN allocation and reward-transfer prior art; no matched local efficacy comparison. |

All 13 entries are cited and resolve in the compiled bibliography. The current
rejected TCAD manuscript is a historical source, not cited as a published
validation result. The source audit retains its equations and discrepancies.
No physical implementation outcome from another paper is transferred into a
NetSuRF v2 result. The special ECC dataset's long title is intentional to avoid
misidentifying the release as a published journal article.

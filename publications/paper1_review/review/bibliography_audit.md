# Bibliography audit

Audit date: 10 November 2026. Input: the newly supplied Overleaf `references.bib`. DOI metadata was checked against Crossref for the central competing and prior-author publications; substantive methods were read from the cited author PDFs, supplied PDFs, and public source archive. This document does not certify every unused reference in the inherited bibliography.

| Citation | Metadata/method verification | Action |
|---|---|---|
| `weng2024fkeras` | Crossref confirms authors, JATS 1(3), 1–27, 23 July 2024, DOI 10.1145/3665334. Author PDF Eq. 3 and tier ordering inspected. | Retain. The author PDF has 26 manuscript pages; use published 27-page metadata. State the NetSuRF continuous-companion adaptation explicitly. |
| `weng2026priorifi` | Crossref confirms ASPLOS 2026 Volume 2, pp. 1461–1475, DOI 10.1145/3779212.3790204. Author PDF Sec. 3 and Algorithm 1 inspected. | Add “Volume 2” and pages if desired; current title/authors/DOI correct. Do not call its published procedure a fixed-budget protection optimizer. |
| `libano2019selective` | Crossref confirms TNS 66(1), 216–222, DOI 10.1109/TNS.2018.2884460. | Retain as selective-hardening precedent. No new quantitative comparison is certified by this metadata check. |
| `catalan2025redundancy` | Crossref confirms title/authors, Journal of Supercomputing 81(1), article 183; online publication 15 November 2024. | Existing 2025 volume citation plus online-2024 note is reasonable; use `articleno={183}` rather than a page interpretation if the style permits. |
| FCCM original design flow | Missing from supplied bibliography. Supplied full PDF and Crossref agree on authors, 2025, pp. 261–267, DOI 10.1109/FCCM62733.2025.00014. | Mandatory addition `guo2025inpixel`; cite in introduction, related work and scope distinction. |
| EISFINN | Missing. Full supplied three-page PDF read. Crossref confirms Colucci, Steininger, Shafique; IOLTS 2024; DOI 10.1109/IOLTS60994.2024.10616075. | Add `colucci2024eisfinn`; metadata uses pp. 1–3, not the proceedings-TOC offset 64–66. |
| Selective-ECC companion release | Missing. [Zenodo record](https://zenodo.org/records/23054820) confirms five creators, v1, release date 30 September 2026, dataset DOI 10.5281/zenodo.23054820. Its code snapshot was inspected. | Add `alobaidi2026selectiveecc` as dataset; do not assign journal publication. The source README's short author list differs from the five-person deposit metadata, so use deposit creators for dataset citation. |

The separate `references_additions.bib` contains only these three verified missing entries. Existing main bibliography files were not modified by this audit.

## Required citation locations

- Introduction: cite the published FCCM flow when explaining the origin of NetSuRF's resource-aware selective-protection objective. Make the new software scope explicit.
- Related work: give FKeras, PrioriFI, EISFINN and the selective-ECC dataset separate objectives, protection granularity, fault domains and evaluation endpoints. Avoid suggesting the first three or the dataset ignore multiple faults/interactions.
- Estimator definitions: cite FKeras Eq. 3 for the published projection; document that NetSuRF uses its magnitude within strict significance tiers and a numerically validated companion, not the integer program's Hessian.
- Conditional formulation: cite the existing Boolean-function reference for established foundations. Do not claim novelty for a multilinear expectation or ordinary conditional marginal.
- Discussion: cite the selective-ECC dataset for existing layer-context diagnostics; distinguish its granularity and ECC/float-forward execution. Do not treat its reported numerical outcomes as independently reproduced here.

## Inherited bibliography risks

The supplied file contains numerous peripheral pruning, verification, security and hardware references. Keep only those actually needed to support the revised scope. Several 2026 arXiv references were not method-audited in this turn and should not support a substantive priority claim without opening their source papers. A valid DOI or arXiv ID establishes a record, not that its method supports our sentence. No blanket statement that the entire inherited bibliography is “verified” is warranted.

Do not cite the rejected TCAD manuscript as published work. It is the historical source of the engineering motivation and voice, not an independently established experimental claim. Do not carry its hardware percentages into the new software paper.

## URLs and source versions

- FKeras paper: https://kastner.ucsd.edu/wp-content/uploads/2024/05/admin/jats24-fkeras.pdf
- FKeras source version used by the reproduction: `68e775c77670c9a1aab4e1b5295b523347d97013`; correspondence recorded in `docs/M97_FKERAS_REPRODUCTION.md`.
- PrioriFI paper: https://kastner.ucsd.edu/wp-content/uploads/2026/05/admin/asplos2026-priorifi.pdf
- PrioriFI source version used by existing NetSuRF engineering: `6e7637b92315e5df75bb6ce258972bdf1131bc20`; scheduling/adaptation boundary recorded in the separate `m12-priorifi` worktree's `docs/PRIORIFI_NUMPY_ENGINEERING.md`.
- Selective-ECC source snapshot: https://zenodo.org/records/23054820/files/code__eccmarl-2ea13f9.tar.gz?download=1 ; SHA-256 `82eed19a02e3242afc908ac249e549778cc0174b9b87ddfc1bfe5f4e72f7e000`.

Citation checks were read-only; no literature code was executed and no protected model/data artifacts were changed.

## TECS revision additions (11 November 2026)

| Citation | Verification in this revision | Note |
|---|---|---|
| `weng2024fkeras` | Author PDF re-read in full: Eq. 3 projection, MSB-to-LSB tier ordering, statement that tier-first ordering "is not ideal", four-ranking comparison against single-bit oracles, 960-flip multi-bit protection experiment on ECON-T. | The passage ranks by "higher score"; it does not state an absolute value. The repository contract uses the magnitude; keep that stated as the NetSuRF adaptation. |
| `weng2026priorifi` | ASPLOS author PDF read: single-bit model, 15% intra-parameter and 80% inter-parameter monotonicity violations, Hessian-initialised campaign ordering. | Fault-injection ordering, not a protection allocator. |
| `alobaidi2026selectiveecc` | Zenodo record page read: 13 allocators including QMIX and search baselines, matched cost, held-out injection, layer-interaction diagnostics. | Publication status of the manuscript not stated on the record. |
| `rausand2004system` (new) | Chapter 5 of the author's lecture version read: Birnbaum's measure as a partial derivative; improvement potential equals Birnbaum's measure times the component failure probability. | Book metadata (2nd ed., Wiley, 2004) from that source. |
| `birnbaum1969importance` (new) | Not inspected. Title, editor, publisher and pages taken from secondary citations. | Confirm before submission. |
| `stutz2021biterror` (new) | MLSys proceedings abstract read. | Cited only for the existence of BER-based robustness training on quantized weights. |
| `nazari2024fortune`, `colucci2024eisfinn` | Not re-read; descriptions rely on the earlier audit above. | Bibliographic records re-confirmed only. |

## Added 11 November 2026 (exposure and related mechanisms)

| Key | Checked against | Status |
|---|---|---|
| `li2017understanding` | Full text (NVIDIA Research PDF), title page, Table 8, Sec. 4.7 and 5.2.1 | Read. Numbers quoted from Table 8. |
| `deutsch2024delayavf` | Title page of the public preprint (par.nsf.gov) | Title and authors read; venue (MICRO 2024) from a search result. |
| `mukherjee2003avf` | Search results citing the MICRO-36 paper | Not read; cited for the definition only. |
| `cummings2023econd` | Slides (text and the plot on slide 14) | Read. Conference slides, not peer reviewed. |
| `diguglielmo2021reconfigurable` | arXiv listing (title, authors, journal reference) | Listing read; the statement about weight registers is from a search summary and needs the co-author's confirmation. |
| `hllhc2017tdr` | Search results (report number, editors, DOI, luminosity targets) | Not read; cited for 3000 fb^-1 and 5e34 only. |
| `alyamani2001abft` | PDF supplied by the researcher, pages 1-3 | Read. |

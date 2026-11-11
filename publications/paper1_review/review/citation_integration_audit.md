# Citation integration audit

Date: 2026-11-10. Scope: the actual `source/**/*.tex` citation commands and `source/references.bib` in this review package.

## Changes

- Merged ten verified missing entries without duplicate keys: FCCM predecessor, EISFINN, the public selective-ECC companion release, DistilBERT, Pythia, Qwen3, SST-2, WikiText, Speech Commands and ToyADMOS.
- Added FORTUNE's published page range (1–6).
- Added PrioriFI's published ASPLOS volume (2) and page range (1461–1475).
- Replaced only the first two paragraphs in the physical-mitigation related-work subsection with precise FORTUNE, Ranger, FT-CNN and selective-TMR distinctions. The companion-release and earlier-NetSuRF discussion was preserved.
- Included the independent novelty/venue audit, metadata audit, related-work addendum and supplemental BibTeX files in this package's `review/` directory.

## Structural check

At this integration snapshot: **38 bibliography entries, 14 distinct cited keys, zero missing citation keys and zero duplicate keys**. Braces are balanced. The scan excluded LaTeX comments and checked `cite`, `citep`, `citet`, `citeauthor`, `citeyear`, `citeyearpar` and `nocite`, including optional arguments and comma-separated keys.

The seven newly added model/dataset references were not yet cited at this snapshot because the experimental-setup section is being updated separately. They should be cited in the benchmark table or its introductory paragraph. Uncited entries do not appear in a normal BibTeX bibliography and were preserved rather than deleted from the researcher's supplied source.

This audit establishes citation-key completeness, not independent validation of every scientific assertion. The accompanying source-specific review documents state what methods were inspected and where access or publication-status limitations remain. Final TeX/BibTeX build validation belongs to the manuscript build receipt.

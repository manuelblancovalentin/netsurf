# Draft v2: rate-aware protection (12 November 2026)

A separate LaTeX project. It does not replace `../source/` (v14), which stays the fallback.

- Claim: which bits to protect depends on the fault rate; a mean-and-variance expression of the expected loss explains
  why, predicts where the preferred ranking switches, and gives a rate-aware assignment that uses the residual error
  curve of each protection mechanism.
- Every Run R result is a placeholder. `\TBD{...}` marks a missing number; `\Branch{R1}{...}{...}` prints the text for
  "rule holds" and "rule fails" side by side. After a result, uncomment `\setbranch{R1}{holds}` (or `{fails}`) in
  `main.tex` and only that version is printed. Rules: R1 (Stage 1), R2 and R2c (Stage 2), R3a and R3b (Stage 3), as
  defined in `../review/RUN_R_RATE_AWARE_PLAN_2026-11-12.md`.
- Static-ranking results of v14 are condensed in Section 7.1; the detail moved to Appendix "Static Rankings in Detail".
- `figures/residual_curves.pdf` is analytic (`../scripts/build_residual_curves_figure.py`); other figures come from
  `../figures/`.
- Build: `latexmk -pdf main.tex`. Length: the main text ends on page 16 with both branches printed and placeholder
  boxes; it must be cut to 14 once the results pick the branches.

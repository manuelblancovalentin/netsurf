# Review of draft v22 against the rejected TCAD paper

Review date: 17 November 2026.

## Overall assessment

Draft v22 is materially stronger than the rejected manuscript. It replaces a weakly justified claim of a superior ranking with a broader experimental comparison and a testable statistical model of simultaneous stored-weight faults. TECS remains a credible target. The manuscript should receive a substantive scientific revision before submission, particularly around the Gaussian approximation, the interpretation of the gradient score, and the scope of the mechanism comparisons.

This is a review of the supplied manuscripts, including the v22 appendices and the reviewer correspondence embedded in the old proof. Reported numerical results below are manuscript claims, not an independent reanalysis of finalized experimental archives. No manuscript, source result, checkpoint, database or campaign was changed or executed for this review.

## What has improved

| Aspect | Rejected TCAD manuscript | Draft v22 |
| --- | --- | --- |
| Central contribution | qPolarGrad ranking plus HLS-backed partial protection | Statistical description of multi-fault behavior, ranking comparisons, and mechanism tradeoffs |
| Interaction argument | Claims that a clean gradient-based expression implicitly captures interference | Explicit distinction between isolated damage and conditional protection value; bounded counterexamples and measured reversals |
| Experimental breadth | Three small benchmark families in the main results | 24 checkpoints across nine architectures, including DistilBERT, Pythia and Qwen; repeated and modified ResNet-8 checkpoints |
| Protection | Selected bits treated as immune in fault-injection comparisons | Main study includes faulty TMR copies and code/checksum storage; explanatory studies explicitly retain ideal suppression |
| Statistical evidence | Repeated injections and error bands with limited inferential detail | Paired fields, declared effect thresholds, multiplicity correction, calibration checks and protocol disclosures |
| Prediction | Broad qualitative claims about cancellation and susceptibility | Noise-plus-events accuracy predictions on 18 checkpoints excluded from model development; prediction of ranking switches |
| Practical scope | Claims about optimal coverage and real hardware deployment outstrip some supporting evidence | Explicit software-only scope and storage-bit costs; still needs narrower wording in several places |

The old criticism about small and insufficiently described models is substantially addressed. The new draft also responds to the objection that rankings should be judged by actual protection quality, rather than by their similarity to a single-bit oracle. Correcting the FKeras description from a diagonal-Hessian approximation to the published eigenvector projection is another material improvement.

The loss of the old synthesis-centered contribution is a tradeoff, not automatically a weakness: v22 is a reliability-modeling and evaluation paper, not a demonstrated physical implementation. Its practical claims must follow that identity.

## The most valuable results

### Predicting behavior rather than only plotting it

Sections 5.2 and 7.2, Figure 5 and Appendix E are the strongest potential methodological contribution. The draft reports 0.08–0.50 accuracy-point mean absolute error per checkpoint on 18 checkpoints excluded from model development, with replay controls and prediction of the sign of practically meaningful mask differences. Predicting the preference between masks is more persuasive than a high correlation between their absolute accuracies alone.

Keep the scope explicit: these are 18 TFLite checkpoints of four architectures, not 18 independent architecture families and not validation of this approximation on the three transformers. Several checkpoints are snapshots of the same training trajectory. The worst individual condition differs by 5.5 accuracy points, so the headline accuracy range must always be described as a per-checkpoint average, not a uniform error bound.

### Useful rankings exist without a universal guarantee

Figures 3–4 and Tables 4–5 provide substantially stronger evidence than the earlier flat curves. For example, the reported absolute-gradient advantage over sign-bit triplication includes 4.5 accuracy points on DistilBERT, with a 95% field interval of 3.1–5.8, and 0.35 nats/token on Pythia, with an interval of 0.32–0.37. These are conditional experimental effects, not merely overlapping curves.

The evidence supports absolute gradient as a strong default in the tested domain. It does not support the earlier aspiration that static ranking is inherently ineffective or that a dynamic allocator must outperform it. A useful default and a rate-dependent exception can both be true.

### Reversals occur while decisions still matter

The ResNet-8 fault-rate study includes multiple training states and calibration replications, and focuses on usable conditions. The reported sign agreement of 515/519 conditions outside the transition neighborhood, and the contrasting absence of a switch on other architectures, provide an explanatory result. The shortcut's false switch prediction on DS-CNN should remain visible: it distinguishes the approximate curvature calculation from the full network-based surrogate.

### Mechanism choice is a separate design decision

Figure 7 shows a useful tradeoff: at the stated storage budgets, ranked triplication has a higher measured tolerated grid rate on eight checkpoints, SEC-DED on two, and fourteen are tied at the grid's resolution. A tie on a coarse grid is not equivalence of continuous failure thresholds. The small family of training trajectories also means these counts are not population frequencies for arbitrary neural networks.

## Issues to resolve before submission

### 1 The Gaussian model is an approximation with exact moments

Locations: abstract; Section 5.2, page 7; Figure 2; Section 7.2, page 13; conclusion.

The mean and variance of independent bit-flip displacements are exact. Their Gaussian distribution is not. Each weight has only six low-order Bernoulli contributions, with geometrically unequal magnitudes. At a BER of 0.001, a weight with all six low bits eligible has no low-bit fault with probability `(1 - 0.001)^6`, approximately 99.4%. Its displacement distribution therefore has a large atom at zero, unlike a Gaussian.

A central-limit argument may motivate approximating sums at a preactivation when no contribution dominates; it does not by itself prove that replacing every weight's noise by a Gaussian preserves a nonlinear network's response. The measured performance of that replacement is valuable empirical evidence. Describe it as a moment-matched surrogate and state the relevant fan-in, dominance and nonlinear-propagation limitations. Do not turn a successful approximation into an exact distributional claim.

The schematic also calls bits 6–7 “a few” events. They are rare per word but may be numerous over an entire large network. Specify which population the adjective describes.

### 2 Absolute-gradient ranking does not guarantee zero drift

Locations: Figure 2(c) and caption, page 8; Section 5.3; page 13, lines 647–650; discussion, page 17.

Ignoring the sign when sorting scores does not ensure that the remaining signed effects cancel. The residual weight drift and its projection on the loss gradient can be nonzero. The DistilBERT measurements support near cancellation in that case; they do not establish a general property of the ranking.

Similarly, ranking by `E[|g Delta|]` is not identical to minimizing a variance term involving `E[g^2] Delta^2`. Explain it as a plausible magnitude-based proxy, with the observed outcomes establishing its usefulness. Replace assertions such as “spread only” and “leaves no systematic drift” with empirical or conditional statements.

### 3 Equation 4 substitutes a curvature proxy for a Hessian

Location: Section 5.3, page 7, and Appendix F.

A second-order expansion around the mean displaced weights involves the loss Hessian at those weights: approximately `R(theta + mu) + 0.5 tr(H_R(theta + mu) Sigma)`. With independent weight displacements, the trace uses the Hessian diagonal. The manuscript instead uses `E_x[g_w(x)^2]`, an empirical-Fisher-type quantity. These are not generally equal, nor are gradients at clean weights automatically curvature at displaced weights.

Label this substitution explicitly as a surrogate assumption, state how and where it is evaluated, and distinguish its explanatory successes from the full noise-plus-events simulation. The DS-CNN counterexample already demonstrates why that distinction matters. This correction does not invalidate the measured ranking comparisons.

### 4 Mechanism comparisons need narrower engineering claims

Locations: Section 3, pages 4–5; Figure 7; discussion recommendation 5, page 17.

Including failure of redundant storage is a major improvement. Nevertheless, “each scheme fails as it would in hardware” is too broad. SEC-DED is represented by an error-count correction abstraction with higher-order miscorrections omitted; ABFT is a particular per-channel single-fault repair abstraction. Voters, decoder faults, recovery after detection, scrubbing, latency and physical layout are not modeled. Figure 7 compares 25% extra storage for TMR with 21.9% for the 32-bit code and model-dependent costs of 1–25% for ABFT. It is a useful storage tradeoff, not an exactly equal-cost full-system comparison.

The draft correctly separates the ideal-suppression model-check study from the failing-protection main study in Section 6.2. Preserve that distinction in the discussion. The statistical surrogate has not thereby been validated for all the mechanisms. Substituting a residual BER is reasonable for independent TMR groups under the stated assumptions; codeword correction can introduce dependence between residual errors, which marginal BERs alone do not capture. Do not claim that the existing Gaussian validation already establishes a cheap mechanism selector.

FORTUNE should be described here as its sign-bit-triplication policy adapted to the frozen INT8 setting, not as a reproduction of every benefit of its quantization-and-protection workflow.

### 5 Main illustrations select operating points using the favored method

Locations: Figures 3–4, pages 11–12.

Figure 3 selects the densest grid BER at which the absolute-gradient mask at 12.5% remains usable, then annotates the largest advantage across budgets. Figure 4 inherits that BER. This is disclosed, but it is still outcome-dependent selection favoring a particular narrative. It does not erase the prespecified full-grid evidence, which is much more important.

Treat these figures as illustrations and support the main inference with the complete fixed comparison family. Add a method-neutral operating-point or full-grid summary, and show that conclusions survive it. Also clarify whether Holm correction includes the entire predefined family or only cells passing an observed-data usability filter; the latter needs a justification or a conservative sensitivity analysis.

“Never confirmed worse” means that no inferiority test passed the stated decision rule. It is not a proof of equivalence or noninferiority. The disclosed expansion from 4/8 to 16 language-model fields also needs accurate description rather than a blanket assertion that all sample sizes were fixed before any outcome. The appendix is commendably transparent about this.

### 6 Establish what the surrogate saves

Locations: Section 7.2; Appendix C; discussion and conclusion.

The draft reports eight surrogate draws per condition and roughly 64,000 passes including replay controls, but does not establish a matched end-to-end speed/accuracy advantage over direct fault injection. A network evaluation is still required, and the hybrid still samples top-bit events. The phrase “no bit-level fault sampling” is therefore misleading as written.

Report setup cost, wall time, memory and error for a fixed number of candidate masks, including direct injection, all-eight-bit Gaussian noise and the hybrid. If there is no computational advantage, position the contribution as an explanatory statistical approximation, not a validated cheap optimization engine. The all-eight-bit Gaussian ablation already has reported mean errors below the one-point criterion; quantify the hybrid's incremental decision accuracy or tail-error benefit instead of claiming that both pieces are indispensable merely because the hybrid has a lower mean error.

### 7 Related work needs to match the new contribution

The new surrogate changes which prior work is closest. DeepVigor+ was cited in the rejected paper but is absent from the v22 bibliography. Its semi-analytical reliability analysis should be discussed, with the actual distinction made clear: its stated mathematical model concerns single-bit faults and vulnerability factors, whereas v22 concerns simultaneous random faults under a protection mask. This is a useful distinction, not grounds to dismiss prior work. See the [primary DeepVigor+ paper](https://arxiv.org/html/2410.15742v2).

The statement that Ranger's protection is already implied by integer arithmetic also needs correction. Ranger derives activation bounds from functions or profiled data and propagates restrictions to selected operations; representable INT8 limits are not automatically the same bounds. Either demonstrate redundancy for the specific executable or treat Ranger as a complementary mechanism outside the matched evaluation. See [Ranger Section III-C](https://arxiv.org/html/2003.13874v4).

The opening claim that all one-score-per-bit methods assume additive multi-fault damage is too sweeping. Scores can incorporate interactions or act as heuristics without making that exact assumption. Critique the lack of a general guarantee for isolated-bit rankings instead.

## Activations and MC-guided selection

The revised response to activation faults is much more defensible than the old paper's assertion that short-lived inputs do not merit protection. A paper may study persistent weight storage without solving whole-accelerator reliability. Keep that scope explicit, and change “every stored bit” to “every eligible weight bit and its redundancy” wherever otherwise ambiguous. Do not claim universal priority of weights over activations. The additional radiation discussion motivates the problem but does not calibrate the logical BER to arbitrary chips or missions.

Appendices F and J now explain the limited outcome of the MC-guided and learned searches. They do not establish general inferiority of those algorithm classes, and the manuscript appropriately says so. Their role is supporting evidence that extra search complexity does not automatically buy held-out improvement. The positive contribution of this paper need not depend on an MC allocator winning.

## TECS recommendation

TECS is a reasonable target for a careful software-level modeling and evaluation contribution. ACM explicitly includes embedded system modeling, fault tolerance, reliability, edge AI and TinyML in the journal's scope: [official ACM scope description](https://www.sigplan.org/announce/2025-10-16-tecs-eic/). Scope fit does not settle novelty, correctness or acceptance.

The defensible paper identity is: a validated moment-matched model for simultaneous stored-weight faults, together with controlled evidence that protection decisions depend on the checkpoint, fault rate, budget and mechanism, and that an inexpensive magnitude-gradient ranking is a strong default in the tested domain.

The current draft is not ready for an unconditional acceptance recommendation. The likely recommendation on scientific content as written is major revision, particularly for the theoretical statements and mechanism extrapolation. The right next step is not another indiscriminate campaign or a retreat to a weaker venue. First correct the inference and scope, strengthen novelty positioning, and run only the small targeted checks needed for any retained predictive-efficiency or mechanism-selection claims.

The supplied PDF has 34 single-column review pages, with the main text through page 18 and references and appendices thereafter. It should not be compared directly with the rejected paper's two-column page count, nor treated as confirmation of a particular TECS page allowance.

## Version identification

| Document | SHA-256 |
| --- | --- |
| NetSuRF_draft_v22.pdf | `c862b7bee4661b0f125bc8575acec60b098df223d942b36c1d8c71204ee3b92f` |
| TCAD-2026-0117.R1_Proof_hi.pdf | `41b60dae3654e0ec855746e48bebfbeeacc0316e5cd929b73dd2625bed3a26c0` |

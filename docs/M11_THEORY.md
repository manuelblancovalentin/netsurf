# M11 — Outside-conditioned responses and restricted protection regret

Protocol `outside-conditioned-forced-clean/v1`. The mathematical results below
concern a fixed checkpoint, input cohort, candidate pool C, and outside protection
context P_out. They do not imply physical reliability, a whole-network optimum,
architectural linearity, or a new general optimization algorithm.

## 1. Five different objects

Write h_G(A)=R(G union A), where G contains only coordinates outside C and P_out.
All eligible candidate codes are exact stored INT8 XOR transitions. Candidate
faults have independent Bernoulli probability p, independently of G. Protection
forces a coordinate clean; this coupling is justified for this product law, not
arbitrary fixed-count/correlated fault processes.

1. Clean isolated damage: d0_b=R({b})−R(empty), on the same input cohort.
2. Outside-conditioned response: d_b(G)=h_G({b})−h_G(empty). Every OTHER candidate
   is forced clean, regardless of whether its proposed protection mask covers it.
3. Full conditional vulnerability: E_G,A[h_G(A union {b})−h_G(A)], where A draws
   the other unprotected candidates. This is a derivative of the multilinear
   expected-risk extension with respect to b's fault probability. It generally
   differs from E_G d_b(G). For h(A)=1[{1,2,3} subset A], the latter is zero and
   the full derivative is p². Signed vulnerability is not an absolute influence.
4. Complete-subcube risk: J_p(Q)=E_G sum_{A subset C\Q}
   p^|A|(1−p)^(m−|A|)h_G(A), m=|C\Q|.
5. Every finite outside-field average estimates one of these quantities,
   conditional on the frozen cohort. Exactly summing candidate states removes
   candidate Monte Carlo variance; it does NOT remove outside-field or input
   sampling uncertainty.

## 2. Exact representation, retaining every interaction order

Let r_G(A)=h_G(A)−h_G(empty)−sum_{b in A}d_b(G). Linearity of expectation gives

    J_p(Q) = E_G h_G(empty) + p sum_{b in C\Q} E_G d_b(G) + E_ge2(Q),
    E_ge2(Q) = E_G,A r_G(A).

r_G vanishes on the empty state and singletons. Equivalently, with anchored
Möbius coefficients μ_G(S)=sum_{T subset S}(−1)^(|S|−|T|)h_G(T),

    E_ge2(Q) = sum_{S subset C\Q, |S|>=2} p^|S| E_G μ_G(S).

This is an identity, not a truncation of the fault engine. The K+1 measurement
baseline measures only the intercept and first-order anchored coefficients;
the oracle integrates all orders. Clean and outside-conditioned ranking can
differ because the entire outside field changes the function being measured.

## 3. Sharp bounded-risk inequality

**Proposition.** If every state risk lies in [0,L], and candidates are independent
Bernoulli(p), then, for m>=2,

    |E_ge2(Q)| <= L m p [1−(1−p)^(m−1)].

For m=0 or1 the bound and remainder are zero. No monotonicity, nonnegative
damage, smoothness, independence of responses across bits, or neural-network
approximation is assumed. G can have any fixed distribution independent of the
candidate draw, provided its law is the same for all compared Q.

**Proof.** Conditional on A of size t>=2,

    r_G(A)=h_G(A)+(t−1)h_G(empty)−sum_{b in A}h_G({b}).

The positive and negative groups each lie in [0,tL], so |r_G(A)|<=tL.
For T~Binomial(m,p),

    |E r| <= L E[T 1(T>=2)]
           = L (mp−mp(1−p)^(m−1)).

The construction h(empty)=L, h(singleton)=0, h(A)=L for |A|>=2 attains the
positive bound simultaneously for every state. Thus the bound cannot in general
be replaced by L Pr(T>=2): subtracting the singleton linearization can overshoot
the [0,L] range. Approximate risks are NOT clipped to hide that fact. At p=0 the
remainder vanishes; at p=1 the bound is mL, also sharp. At fixed small mp it is
approximately L m(m−1)p². Small mp is not intrinsic network additivity.

For heterogeneous independent probabilities the same proof gives
L sum_b p_b [1−product_{j!=b}(1−p_j)]. For correlated candidates, use the actual
E[T 1(T>=2)] and actual first moments; the binomial expression is not justified.
Protection also must not change the outside-field law without changing J.

ToyCar has L=1 for authoritative complete-file 1−ROC-AUC, including half-credit
ties. The existing classification metric floors each true-class softmax
probability at 1e−12, giving L=−log(1e−12). This is a bound for that recorded
clipped metric, not unbounded logit cross-entropy. The codec/inference is unchanged.

## 4. Restricted-optimum regret and finite-sample separation

Let A(Q) be the outside-additive prediction on a fixed equal-budget family F.
If sup_F |A−J|<=ε, Q_A minimizes A, and Q_* minimizes J, then

    J(Q_A)−J(Q_*)
       <= [J(Q_A)−A(Q_A)] + [A(Q_*)−J(Q_*)] <= 2ε.

Cap by L, since both true risks lie in [0,L]. This concerns F only, not all
whole-network masks. For an estimated additive function with uniform error η,
the same argument gives 2ε+2η. Deterministic bias and sampling error are distinct.

**A tighter equal-budget sampling term.** For n iid outside fields,
d_b in [−L,L]. Hoeffding and a K-coordinate union bound imply, with probability
at least 1−α, simultaneous |mean_n d_b−E d_b| <=

    t = L sqrt(2 log(2K/α)/n).

Any two masks of cardinality r differ in at most 2 min(r,K−r) coordinates.
The shared random intercept cancels. Therefore the data-selected additive mask
has restricted regret at most

    min(L, 2ε + 2p min(r,K−r)t).

This certificate needs independent fields, fixed pool/cohort, and a valid L.
Reuse of the same field across bits is allowed and desirable. No independence
between candidate responses is asserted. For simultaneous guarantees over
several pools/BERs, allocate α across the complete prespecified family. Empirical
standard errors and Student-t intervals are descriptive approximations, not
substitutes for this finite-sample bounded-risk guarantee. Zero observed variance
does not prove equivalence. Fixed-cohort certificates do not imply input-population
or candidate-population guarantees, nor a guarantee over independently trained models.

## 5. Curvature bounds and what an audit can certify

Suppose |h_G(S+i+j)−h_G(S+i)−h_G(S+j)+h_G(S)|<=κ_ij for EVERY allowed G,S
excluding i,j. Add coordinates successively and telescope each incremental
effect back to its empty-context singleton. Each pair appears at most once:

    |r_G(A)| <= sum_{i<j in A} κ_ij,
    |E_ge2(Q)| <= p² sum_{i<j in C\Q} κ_ij.

The minimum of this bound and Section 3 is valid when κ is globally certified.
Exhaustive K<=12 enumeration yields exact maxima over candidate contexts for
each OBSERVED G. It bounds the empirical-field risk only. The maximum across
sampled outside fields is not an upper confidence bound on unseen curvature.
Pairwise differences at the empty candidate state alone do not bound higher-order
interactions: a pure third-order function has zero such differences and nonzero
third-order damage. M11 retains this falsifier.

## 6. Failure regimes and comparisons

A purely third-order response on candidate indices3,4,5 has all singleton scores
zero. At K=6,r=2,p=.5 stable additive ties protect0,1, leaving risk .125, while
protecting3 and any other candidate has risk0. This is a valid bounded-risk
counterexample to universal ranking optimality. Negative marginals/beneficial
faults are retained; ranking absolute damage would define a different policy.

For K<=12, measure all legal masks on the SAME observed fields. Distinguish
absolute-risk prediction errors, centered decision contrasts, rank reversals,
near-ties, and finite-selection-field regret. Choose masks on selection data
only. At K>12, there is no enumerated optimum: independent paired final fields
measure actual frozen-mask risk, not regret against an unknown optimum.

Equal BER and equal λ=(N−B)p are separate prespecified controls. They do not
equalize physical exposure, architecture, perturbation magnitude or task units.
Do not numerically pool CE and AUC risks. Saturated/null/adverse pools stay in
the analysis. Probability-sampled pools and structural enrichment are distinct
estimands. Joint protection quality is not an unbiased bit-population statistic
merely because individual candidates have known inclusion probabilities.

## 7. Prior work and claim boundary

The product-measure multilinear extension, discrete differences and influence
are established mathematics, not NetSuRF inventions. Cite O'Donnell's
*Analysis of Boolean Functions* (biased measures/discrete derivatives) and
Rauzy's *Model-Based Reliability Engineering*, Section6.8 (marginal/Birnbaum
importance). Nonmonotone signed risk contrasts must not be confused with a
Boolean pivotal probability without the relevant assumptions.

FKeras's continuous Hessian projection is a static parameter score; its
bit-tier ordering is not an estimator of J_p(Q) conditional on an outside fault
field. PrioriFI uses initial Hessian information and observed fault outcomes
to prioritize further injections. Neither comparison justifies saying that a
Hessian contains no interactions. M11 does not reproduce PrioriFI's adaptive
algorithm. See `M97_FKERAS_REPRODUCTION.md` for pinned paper/source differences.

EISFINN's verified institutional abstract describes neuron-sensitivity-guided
importance-sampled fault scenarios. The full three-page methodology was not
accessible in this audit; its likelihood-weighting and target-estimand details
remain unverified. No claim that M11 supersedes it is justified. The same authors'
ISimDL preprint is related context, not a substitute for the EISFINN paper.

Catalán et al. exploit invariant bit patterns within groups of stored weights,
with consecutive/nonconsecutive recovery masks and hardware overhead. This is
an implemented recovery mechanism, not an ideal individual-bit suppression
policy or a conditional-risk representation.

The [September2026 Zenodo release](https://doi.org/10.5281/zenodo.23054820) is a
public companion dataset; peer-reviewed status of *Learned or Ranked?* is not
established. We inspected source snapshot `eccmarl-2ea13f9` (archive SHA256
`82eed19a02e3242afc908ac249e549778cc0174b9b87ddfc1bfe5f4e72f7e000`). Its
`interaction.py` measures paired layer-level pair effects, subset additivity
and context-dependent layer rankings. `faultmodels.py` includes ideal suppression
and word-based approximate ECC decoding, burst/interleaving variants and explicit
uncorrectable-word handling. `reward.py` combines clipped accuracy degradation
and redundancy cost. These are directly relevant precedents: neither context
ranking nor matched-cost learned-versus-static comparisons can be claimed new.
They are not exact INT8-coordinate Möbius coefficients under NetSuRF's fixed
Bernoulli law. Report both overlap and the differing interventions/estimands.

Potential contribution, still contingent on wider evidence: an exact-execution,
multi-task validation of candidate-forced-clean conditional responses, with
explicit restricted-decision error/regret bounds, measured scaling/failure
regimes and separated uncertainty. The proof alone and a small engineering
smoke do not establish publishable empirical benefit. No priority/novelty claim
is made for the bound without a broader theorem-level literature review.

Suggested manuscript citation locations: mathematical preliminaries (Boolean /
reliability); related work and static baselines (FKeras/PrioriFI); FI sampling
(EISFINN); mitigation semantics (Catalán); comparative scope and interactions
(Zenodo dataset). No unrelated manuscript text is changed.

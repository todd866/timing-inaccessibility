# Changelog

The version of record is the published paper (BioSystems 258, 105632; DOI in the
README). This repository tracks ongoing development and corrections; the published
text remains the reference.

## 2026-06-11 — corrections

- **Appendix A (protein-folding degeneracy estimate).** The displayed equation adds a
  resolution term `(D_eff/κ)·log10(τc/Δt_fine)` (≈ 12–120) to a state-ratio term
  `log10(N_micro/N_meso)` (≈ 42–94), but the stated result (~42–94) reflects only the
  second term. The resolution term double-counts and should be omitted — consistent
  with the neural subsection, which explicitly drops its analog — leaving ~42–94. The
  order-of-magnitude conclusion (degeneracy ≫ 1) is unaffected.
- **Appendix C citation.** Inter-site phase coherence (ISPC) is attributed to Stringer
  et al. 2019; ISPC is standard phase-connectivity terminology (Lachaux et al. 1999;
  Cohen 2014). Stringer 2019 is correctly cited elsewhere for participation-ratio
  dimensionality.

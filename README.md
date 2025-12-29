# Timing Inaccessibility and the Projection Bound

**Resolving Maxwell's Demon for Continuous Biological Substrates**

**Published in BioSystems 258, 105632 (2025)**

[![DOI](https://img.shields.io/badge/DOI-10.1016/j.biosystems.2025.105632-blue)](https://doi.org/10.1016/j.biosystems.2025.105632)

## Overview

Maxwell's demon was resolved by Landauer and Bennett: the demon must erase information to reuse its memory, paying ≥ kT ln 2 per bit. But this resolution assumed discrete memory states. This paper extends the resolution to continuous biological substrates, showing that the thermodynamic advantage of living systems arises from **timing inaccessibility**.

**Key result:** Irreversibly registering fine temporal order requires ≥ kT ln 2 per stabilized order bit. When couplings are sub-Landauer, extracting such records requires measurement resources that dominate the coupling energies (back-action). This creates massive path degeneracy—exponentially many micro-trajectories map to the same observable outcome.

## Version History

This paper follows a **living document** approach, with periodic upgrades that extend the theoretical framework while preserving backward compatibility with the published version.

### v1.0 (November 2025)
- Published in BioSystems
- Core arguments: timing inaccessibility, path degeneracy, Projection Bound
- Camera-engine duality for biological demons
- DOI: [10.1016/j.biosystems.2025.105632](https://doi.org/10.1016/j.biosystems.2025.105632)

### v2.0 (December 2025)
- **Maxwell's demon lineage**: Explicit trace from Maxwell → Szilard → Landauer → Bennett → this paper
- **Two-stage model**: Stage 1 (reversible correlation) vs Stage 2 (stabilization/registration)
- **Framework dependence of timing**: What counts as "simultaneous" vs "sequential" depends on measurement framework
- **The "when" is created, not revealed**: Temporal order emerges at projection
- **AI substrate constraints**: Digital architectures and projection boundary placement
- **Reviewer-proofing**: TRB scope conditions, PB anchored in standard thermo, metric dependence
- Available in: `v2.0/` folder

**Why version papers?** As AI tools improve, we revisit past publications to increase the rigor of the analysis. This could be perceived as undermining peer review—but the underlying result hasn't changed. The v1.0 claims stand; v2.0 simply makes them harder to misunderstand and harder to attack. Each version is self-contained and citable. The published version remains the canonical reference for formal citation; upgraded versions are available here for those who want the strongest form of the argument.

## Repository Structure

```
timing-inaccessibility/
├── demon.tex                   # v1.0 manuscript (published)
├── v2.0/                       # Version 2.0 (December 2025)
│   ├── timing_inaccessibility_v2.tex   # Upgraded manuscript
│   ├── timing_inaccessibility_v2.pdf   # Compiled PDF
│   └── figures/                # Figures + generation scripts
└── README.md
```

## Key Results

### v1.0 (Published)
1. **Temporal Registration Bound (TRB)**: Ordering M temporal bins requires log₂(M!) bits
2. **Projection Bound (PB)**: Quasistatic dimensional collapse dissipates ≥ kT_eff ln(N_pre/N_post)
3. **Path degeneracy**: 10^42–10^94 (proteins), 10^50–10^100 (neural) as upper bounds
4. **Camera-engine duality**: Biological demons sense via weak coupling, steer via coordinated back-coupling, pay only at collapse

### v2.0 (Extended)
5. **Two-stage model**: Reversible coupling (Stage 1, free) vs stabilization/registration (Stage 2, Landauer cost)
6. **Framework dependence**: Temporal resolution determines causal visibility; regress terminates at Landauer threshold
7. **Projection boundary placement**: The key variable for efficiency is not "digital vs analog" but where stabilization occurs
8. **Three claims ladder**: Thermodynamic (strong), operational-access (plausible), interpretive (optional)

## Citation

For the published version:
```bibtex
@article{todd2025timing,
  title={Timing inaccessibility and the projection bound: Resolving Maxwell's demon for continuous biological substrates},
  author={Todd, Ian},
  journal={BioSystems},
  volume={258},
  pages={105632},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.biosystems.2025.105632}
}
```

For the extended version (v2.0):
```bibtex
@misc{todd2025timingv2,
  title={Timing Inaccessibility and the Projection Bound: Resolving Maxwell's Demon for Continuous Biological Substrates (Version 2.0)},
  author={Todd, Ian},
  year={2025},
  note={Extended version available at: https://github.com/todd866/timing-inaccessibility}
}
```

## Author

Ian Todd
Sydney Medical School, University of Sydney
itod2305@uni.sydney.edu.au
ORCID: [0009-0002-6994-0917](https://orcid.org/0009-0002-6994-0917)

## License

MIT

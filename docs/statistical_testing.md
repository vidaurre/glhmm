# Statistical testing

Once you have trained a model and extracted state measures (such as fractional occupancy, state lifetimes, or transition probabilities), a natural next step is to assess whether observed differences are statistically significant.

The toolbox supports both parametric and non-parametric tests. The non-parametric approach builds a null distribution by resampling the data: depending on the design, this means either permutation testing (randomly shuffling labels across observations) or Monte Carlo resampling (used for longitudinal designs such as across-visits testing). Setting `Nnull_samples=0` switches to a standard parametric test instead.

Non-parametric resampling makes few assumptions about the data distribution and works well with the small-to-medium sample sizes common in neuroimaging.

---

## Choosing the right test

The right test depends on how your data are structured:

| Situation | Notebook |
|---|---|
| One measurement per subject, testing across subjects | [Testing across subjects](notebooks/Testing_across_subjects.ipynb) |
| Multiple sessions per subject, testing within subject | [Testing across sessions](notebooks/Testing_across_sessions_within_subject.ipynb) |
| Multiple trials within a session | [Testing across trials](notebooks/Testing_across_trials_within_session.ipynb) |
| Multiple measurements on the same subject during scanning (e.g., comparing brain state measures with a simultaneous physiological signal such as heart rate, pupil size, or skin conductance) | [Testing across visits](notebooks/Testing_across_visits.ipynb) |
| HCP data with family structure | [HCP: testing across subjects](notebooks/HCP_Testing_across_subjects.ipynb) |
| HCP data requiring block permutation | [HCP: multi-level permutation](notebooks/HCP_multi_level_block_permutation.ipynb) |

---

## Further reading

The statistical testing framework is described in full in:

> Larsen, N.Y., Paulsen, L.B., Ahrends, C., Winkler, A.M. & Vidaurre, D. (2026). *A comprehensive framework for statistical testing of brain dynamics.* Nature Protocols, 21, 3148–3179. [https://doi.org/10.1038/s41596-025-01300-2](https://www.nature.com/articles/s41596-025-01300-2)

A code-free GUI for running all tests is available at [github.com/Nick7900/glhmm_protocols](https://github.com/Nick7900/glhmm_protocols).

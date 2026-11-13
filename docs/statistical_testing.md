# Statistical testing

After training, you can test whether the states relate to something else, like behaviour, a condition or a physiological signal. glhmm does this with permutation tests. It shuffles the variable many times and checks how often a result as strong as yours shows up by chance. This works without assuming that your data are normally distributed.

Every test takes two inputs: `D_data`, which describes the states (e.g. the fractional occupancy of each state), and `R_data`, the variable you want to relate them to.

---

## Which test?

That depends on how your data were collected:

| Your data | Test | Example | Full protocol |
|---|---|---|---|
| One value per person, e.g. the fractional occupancy of each state, and a variable per person, like age or a test score | `test_across_subjects` | [Testing across subjects](notebooks/Testing_across_subjects.ipynb) | [Procedure 1](https://github.com/Nick7900/glhmm_protocols/blob/main/Procedures/Procedure_1_Across_subject_testing.ipynb) |
| A group of people with several sessions each, or people who are related, like siblings | `test_across_subjects` with `indices_blocks` | *Block permutations* below | |
| Trials within sessions. Do the states differ between trial types, e.g. conditions? | `test_across_trials` | [Testing across trials](notebooks/Testing_across_trials_within_session.ipynb) | [Procedure 2](https://github.com/Nick7900/glhmm_protocols/blob/main/Procedures/Procedure_2_Across_trials_testing.ipynb) |
| Several sessions from one person. Does the relation between the states and a variable change from session to session? | `test_across_sessions_within_subject` | [Testing across sessions](notebooks/Testing_across_sessions_within_subject.ipynb) | [Procedure 3](https://github.com/Nick7900/glhmm_protocols/blob/main/Procedures/Procedure_3_Across_session_testing.ipynb) |
| A signal recorded during the scan, like pupil size or heart rate. Is it different when a state is active? | `test_across_state_visits` | [Testing across visits](notebooks/Testing_across_visits.ipynb) | [Procedure 4](https://github.com/Nick7900/glhmm_protocols/blob/main/Procedures/Procedure_4_Across_visits_testing.ipynb) |

The examples are short and use simulated data. The protocols go through each test in full on real data, and are described in the [Nature Protocols paper](https://www.nature.com/articles/s41596-025-01300-2).

---

## Choosing the options

The tests share most of their options. These are the ones you'll use most. The examples and protocols above show them in use, and the [statistics](statistics.md) API page lists all options of each test.

| If you want to … | Set |
|---|---|
| get one p-value per variable, testing all states together | `method="multivariate"` (the default) |
| get one p-value per state and variable | `method="univariate"` |
| get one p-value for the link between all states and all variables | `method="cca"` |
| test a variable with categories, like groups or conditions | `detect_categorical=True`, which uses a t-test or ANOVA instead of a regression |
| remove the effect of e.g. age or sex | `confounds=...`, which is regressed out of both inputs |
| correct for testing many states and variables | `FWER_correction=True`, or `statistics.pval_correction(result_dic=result)` afterwards for FDR |
| correct across timepoints | `statistics.pval_cluster_based_correction(result)` |
| combine all p-values into one | `combine_tests=True` |
| take several sessions per person, or related people, into account | `indices_blocks` with `permute_within_blocks` and/or `permute_between_blocks`, see *Block permutations* below |
| run a parametric test instead of permutations | `Nnull_samples=0` |
| make it faster | `n_jobs=4`, see *Speeding it up* below |

The test across visits also has two methods of its own: `method="osr"` compares one state with all the others, and `method="osa"` compares pairs of states. See its example.

`statistics.display_test_summary(result)` prints a table of the results.

---

## Block permutations: several sessions per person, or families

`test_across_subjects` treats every row as a different, unrelated person. If you have several sessions per person, the sessions of one person are more alike than sessions of different people. If you ignore that, the test finds effects that aren't there, or misses effects that are. So give each row the ID of the person it belongs to with `indices_blocks`, and choose how to shuffle. (The other tests do this for you. The test across trials only shuffles trials within each session, and the test across sessions swaps whole sessions.)

Here's what the shuffling does, with three people who each have three sessions, and their reaction time on the task in each session. The states stay where they are, and only the reaction times are moved around. The colours show whose reaction time each number is:

<!-- Edit this table as HTML, since Markdown tables can't colour cells. Each number is coloured by whose reaction time it is (Pastel1):
     Anna #fbb4ae, Ben #b3cde3, Carl #ccebc5. color:#222 keeps the text readable in dark mode. -->
<div class="table-wrapper docutils container">
<table class="docutils align-default">
<thead><tr><th>Person</th><th>Session</th><th>Your data</th><th>Shuffled within people</th><th>Shuffled between people</th><th>Both</th></tr></thead>
<tbody>
<tr><td style="background:#fbb4ae;color:#222">Anna</td><td>1</td><td style="background:#fbb4ae;color:#222">410</td><td style="background:#fbb4ae;color:#222">450</td><td style="background:#ccebc5;color:#222">380</td><td style="background:#ccebc5;color:#222">390</td></tr>
<tr><td style="background:#fbb4ae;color:#222">Anna</td><td>2</td><td style="background:#fbb4ae;color:#222">450</td><td style="background:#fbb4ae;color:#222">430</td><td style="background:#ccebc5;color:#222">390</td><td style="background:#ccebc5;color:#222">370</td></tr>
<tr><td style="background:#fbb4ae;color:#222">Anna</td><td>3</td><td style="background:#fbb4ae;color:#222">430</td><td style="background:#fbb4ae;color:#222">410</td><td style="background:#ccebc5;color:#222">370</td><td style="background:#ccebc5;color:#222">380</td></tr>
<tr><td style="background:#b3cde3;color:#222">Ben</td><td>1</td><td style="background:#b3cde3;color:#222">520</td><td style="background:#b3cde3;color:#222">500</td><td style="background:#fbb4ae;color:#222">410</td><td style="background:#fbb4ae;color:#222">430</td></tr>
<tr><td style="background:#b3cde3;color:#222">Ben</td><td>2</td><td style="background:#b3cde3;color:#222">560</td><td style="background:#b3cde3;color:#222">520</td><td style="background:#fbb4ae;color:#222">450</td><td style="background:#fbb4ae;color:#222">410</td></tr>
<tr><td style="background:#b3cde3;color:#222">Ben</td><td>3</td><td style="background:#b3cde3;color:#222">500</td><td style="background:#b3cde3;color:#222">560</td><td style="background:#fbb4ae;color:#222">430</td><td style="background:#fbb4ae;color:#222">450</td></tr>
<tr><td style="background:#ccebc5;color:#222">Carl</td><td>1</td><td style="background:#ccebc5;color:#222">380</td><td style="background:#ccebc5;color:#222">370</td><td style="background:#b3cde3;color:#222">520</td><td style="background:#b3cde3;color:#222">560</td></tr>
<tr><td style="background:#ccebc5;color:#222">Carl</td><td>2</td><td style="background:#ccebc5;color:#222">390</td><td style="background:#ccebc5;color:#222">380</td><td style="background:#b3cde3;color:#222">560</td><td style="background:#b3cde3;color:#222">500</td></tr>
<tr><td style="background:#ccebc5;color:#222">Carl</td><td>3</td><td style="background:#ccebc5;color:#222">370</td><td style="background:#ccebc5;color:#222">390</td><td style="background:#b3cde3;color:#222">500</td><td style="background:#b3cde3;color:#222">520</td></tr>
</tbody>
</table>
</div>

```{only} latex
The coloured example table is only shown in the [online documentation](https://glhmm.readthedocs.io/en/latest/statistical_testing.html).
```

- **Within people**, everyone keeps their own reaction times, just in a different order. Ben is still the slowest. So the test only finds something if a person's states change with their reaction time from session to session.
- **Between people**, everyone gets someone else's reaction times, in the same order. Anna now has Carl's. So the test compares people with each other.
- **Both** does both. Everyone gets someone else's reaction times, in a new order.

The test repeats the shuffling thousands of times and checks how often the link between the reaction times and the states is as strong as in your data. Which shuffling to use depends on your variable:

| Your variable … | For example | Set | If it's significant, you can say that … |
|---|---|---|---|
| changes between the sessions of a person | session 1, 2 and 3, or a condition | `permute_within_blocks=True` | a person's states change when the variable changes |
| is the same in all sessions of a person | age, or a test score | `permute_between_blocks=True` | people with different values have different states |
| is measured in every session, and differs between sessions and between people | reaction time on the task in each session | both: `permute_within_blocks=True, permute_between_blocks=True` | the variable is linked to the states across all sessions, as in a test without blocks, but with a correct p-value |

For a variable like reaction time, you can also shuffle only within people, if you just want to know whether a person's states change with their reaction time from session to session. That's more sensitive to changes within a person, but doesn't see differences between people.

Use one row per session. The rows can be in any order. For example, to test whether the states differ between sessions 1, 2 and 3:

```python
from glhmm import statistics, utils

FO = utils.get_FO(Gamma, indices)   # one row per session, as in indices
result = statistics.test_across_subjects(FO, session, method="univariate",
                                         indices_blocks=subject, permute_within_blocks=True,
                                         detect_categorical=True)
```

`session` and `subject` have one value per session, in the same order as the rows of `indices`. `detect_categorical=True` tests whether the sessions differ at all. Without it, the session numbers are treated as numbers, and the test only looks for a steady increase or decrease.

For age, the call is the same, but with `permute_between_blocks=True`:

```python
result = statistics.test_across_subjects(FO, age, method="univariate",
                                         indices_blocks=subject, permute_between_blocks=True)
```

Here `age` is repeated for each session of a person. People are only swapped with people who have the same number of sessions. If the numbers differ, compute one value per person first, and test those without blocks:

```python
FO_person = utils.get_FO(Gamma, indices, session_of_segment=subject)   # one row per person, in the order of the sorted IDs
result = statistics.test_across_subjects(FO_person, age_person, method="univariate")
```

If you choose a setting that can't change your variable, e.g. shuffling within people for age, every shuffle gives back your own data, and all p-values are 1. glhmm warns you when this happens. glhmm also prints how many different shuffles your design allows. If that number is small, the p-values can't get small either. With 5 people, for example, shuffling between people allows only 120 orders, so p can't go below 1/120.

### Related people, like siblings

Siblings are more alike than strangers, just as sessions of one person are more alike than sessions of different people. So the same idea applies, with the family as the block and each person as a row. Again, which shuffling to use depends on your variable:

- a value per person that differs between siblings and between families, like a test score: shuffle both within and between families. This is the usual choice.
- a value that's the same for the whole family, like household income: shuffle between families only.
- to check whether a link also holds between siblings, and doesn't just come from something families share: shuffle within families only. This only uses the differences between siblings.

```python
result = statistics.test_across_subjects(FO_person, score_person, method="univariate",
                                         indices_blocks=family,
                                         permute_between_blocks=True, permute_within_blocks=True)
```

`FO_person` and `score_person` have one row per person, and `family` gives the family of each person. Families are only swapped with families of the same size, so shuffling between families works best when many families have the same number of members. If it's significant, it means the same as with unrelated people, and the p-value takes into account that siblings are alike. For data with twins, where siblings aren't interchangeable, use `dict_family`, as in the [HCP example](notebooks/HCP_Testing_across_subjects.ipynb).

---

## Speeding it up

All tests take an `n_jobs` argument (default `1`). With `n_jobs > 1`, the permutations run on that many CPU cores. The statistics are computed in the same way as with `n_jobs=1`, and for tests across subjects and across trials the results are identical for the same random seed.

For testing across visits, the random Viterbi paths are made inside the parallel jobs, so `np.random.seed` doesn't make a parallel run reproducible. To get the same results as with `n_jobs=1`, make them first with `statistics.surrogate_state_time_matrix` and pass them to `test_across_state_visits` as `vpath_surrogates`.

---

## Further reading

The tests are described in full in:

> Larsen, N.Y., Paulsen, L.B., Ahrends, C., Winkler, A.M. & Vidaurre, D. (2026). *A comprehensive framework for statistical testing of brain dynamics.* Nature Protocols, 21, 3148–3179. [https://doi.org/10.1038/s41596-025-01300-2](https://www.nature.com/articles/s41596-025-01300-2)

A code-free GUI for running all tests is available at [github.com/Nick7900/glhmm_protocols](https://github.com/Nick7900/glhmm_protocols).

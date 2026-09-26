# Issues affecting the published results

Findings from reviewing and re-running this pipeline, 2026-09-20 to 2026-09-26. Each entry says
what the issue is, the evidence for it, and how much it changes the result — separating what has
been measured from what has not.

Reference points used throughout: Rescorla-Wagner scores **0.4685** mean test NLL per free choice
over the nine experiments, reproduced here to four decimals per experiment and row-for-row in the
per-trial files, so the scoring path itself is sound. Chance is `log(num_options)`: 0.693 for the
two-option tasks, 1.386 for DriftingBandit, 1.099 for Maggie's Farm.

---

## 1. Parameters were barely fitted, and the selection rule discarded good models

**What.** The search scored each candidate after roughly 23 objective evaluations of the optimiser.
That is not enough to fit even a handful of free parameters.

**Evidence.** Rescorla-Wagner itself scores **0.476** under that budget and **0.636** when fitted
properly. So the published selection rule, applied to the baseline it is compared against, would
have rejected it.

**Effect.** This is the largest single issue, because it corrupts the comparison in both
directions: candidates with more parameters are penalised arbitrarily (they need more evaluations
to reach their optimum), and the reported "evolved models lose to Rescorla-Wagner" conclusion is
partly a statement about the optimiser budget rather than about the models.

**After fixing** (real budget, multi-start, JAX gradients where the program permits): a 4000-iteration
base-model search produces a program scoring **0.4768** over the nine tasks against RW's 0.4685 —
beating RW on **6 of 9**, including the held-out ChangingBandit (0.3834 vs 0.4146). It loses badly
only on Maggie's Farm (0.6274 vs 0.4961), the only three-option task and one never seen during the
search. Excluding Maggie's Farm: **0.4580 vs 0.4651**, i.e. the evolved model wins.

---

## 2. Selection on training-set NLL, which overfits

**What.** Candidates were ranked by the NLL on the same participants used to fit their parameters.

**Evidence.** The best program under that rule scored 0.6293 on the training split and **0.5095** on
test — worse than RW (0.4685) — with per-task generalisation gaps up to **+0.19**.

**Effect.** The search optimises the gap rather than the model. Selecting on a held-out validation
split instead brings gaps down to about **0.03**.

**Note.** This interacts with issue 1: a short fitting budget suppresses overfitting by accident, so
fixing the budget without also fixing the selection rule makes generalisation *worse*, not better.
Both have to move together.

---

## 3. Programs branch on `game` where they mean the experiment

**What.** Both published best programs test the game number when their own comments say they mean
the experiment:

```python
if game in [1, 2]:   # 2-option experiments
elif game == 3:      # 4-option experiment
```

Experiment indices are 1=TwoBandit1, 2=TwoBandit2, 3=DriftingBandit (the four-option task),
4=HorizonSomer, 5=HorizonWaltz. But `game` is the game number *within a participant*, so
`game == 3` selects everyone's third game in every experiment. The experiment index is never passed
to `predict`, so what the code meant was not expressible in the interface it was given.

**Evidence.** Refitting both published programs as published, and again with `game` replaced by the
experiment index, under an identical budget (Powell, `maxfev=300`). Test NLL:

| experiment | Base as published | Base corrected | FT as published | FT corrected |
|---|---|---|---|---|
| TwoBandit 1 | 0.4420 | 0.4422 | 0.4489 | 0.4489 |
| TwoBandit 2 | 0.3711 | 0.3767 | 0.3686 | 0.3665 |
| DriftingBandit | 0.8983 | 0.8983 | 0.8334 | 0.8332 |
| HorizonSomer | 0.4342 | 0.4320 | 0.4322 | **0.5094** |
| HorizonWaltz | 0.3620 | 0.3591 | 0.3234 | **0.4006** |
| HorizonSade *(held out)* | 0.6175 | 0.6179 | 0.6046 | 0.6024 |
| HorizonFeng *(held out)* | 0.3622 | 0.3610 | 0.3629 | 0.3611 |
| ChangingBandit *(held out)* | 0.4443 | 0.4438 | 0.3879 | 0.3872 |
| Maggie's Farm *(held out)* | 0.6603 | 0.6592 | *pending* | *pending* |
| **mean over the first 8** | **0.4914** | **0.4914** | **0.4702** | **0.4886** |

**Effect, base program:** none. Mean change −0.0001, nothing above 0.006. The bug is real in the
code and inert in the results.

**Effect, fine-tuned program:** correcting the bug *costs* 0.077 on HorizonSomer and HorizonWaltz and
does nothing on the held-out horizon tasks. Under the intended reading, the horizon-specific branch
fires on every game of those two tasks; under the bug it fires only on games 4–5, which mostly
switches that machinery off. So the LLM's horizon-specific logic is actively harmful, and the bug
accidentally suppressed it. The published program is better than the program it was trying to write.

**Consequence for the headline comparison.** The fine-tuned program beats the base program by 0.021
as published (0.4702 vs 0.4914). Corrected, they are level (0.4886 vs 0.4914). **Most of the
apparent fine-tuning advantage is an artefact of this bug**, not evidence that fine-tuning found a
better cognitive model.

*Caveat: this table uses a lightweight Powell harness at `maxfev=300`, not the full JAX pipeline, so
treat absolute values as approximate. The differences are the meaningful part, since both columns
use an identical budget.*

---

## 4. The fine-tuned model was barely fine-tuned, on the wrong objective

Two independent defects, either of which alone would undermine the base-versus-fine-tuned
comparison.

### 4a. ~90% of the adapter never trained

**What.** Every one of the 96 expert `lora_B` tensors in the published adapter is exactly zero.
`peft` 0.18 attaches adapters to the fused MoE parameters through `target_parameters` and does not
propagate gradients to them; `lora_B` starts at zero, so it stays there.

**Scale.** With 512 experts at `effective_r=1` over 48 layers, the expert adapters account for about
**139M of the adapter's 155.5M parameters**. Only ~16M — attention, linear attention and the shared
MLPs — were ever trained. The "fine-tuned" model was therefore nearly the base model.

**Evidence.** Direct inspection of the adapter's safetensors. Also unrecoverable after the fact: no
merge can apply a delta that was never learned, which is why `LLMFineTuning/mergeExpertDeltas.py`
(written to patch the deltas in by hand) cannot help.

**Fixed.** `peft` 0.19.1 corrected the layout; verified on 0.21.0, where a retrained adapter has
96/96 expert tensors non-zero both at step 5 and in the saved file.

### 4b. The loss covered the whole transcript, not the participants' choices

**What.** The code builds a `completion_mask` marking the choice tokens, with the comment
"-100 is don't compute nll over it". trl folds that mask into the labels **only** when
`completion_only_loss` is true, and that defaults to whether the dataset has `prompt` and
`completion` columns — this one has `input_ids` and `completion_mask`. The mask was silently
dropped and every token contributed to the loss.

**Evidence.** The published run logs **0.856 token accuracy and 0.408 loss at step 1**. The same
base model on the same data scores **0.8544** validation loss on the choices alone. A model cannot
be at 0.41 on human bandit choices when chance is 0.69 and the untrained value is 0.85; 88% token
accuracy is what predicting templated text like "You chose X and received Y points" looks like.
`num_tokens` at step 1 is identical in both runs (139,625), so the data is the same and only the
loss coverage differs.

**Effect.** The model was trained to reproduce Psych-101 transcript text rather than to predict
choices — a different objective from the one the code describes.

### After retraining (2026-09-26)

3 epochs, 63 steps, choice-only loss: validation **0.8544 → 0.5022**, decreasing monotonically and
flat at the end (0.5029 → 0.5024 → 0.5022), so no earlier checkpoint is preferable. 96/96 expert
tensors live. This is the first adapter for which "base versus fine-tuned" is a real contrast.

**What this does not establish.** A 41% cut in choice-prediction loss shows the model learned human
choice behaviour. It does not show it writes better cognitive models — a different capability, and
training 139M parameters on non-code text can erode code ability. The direction is open, and the
prior evidence for a fine-tuning benefit is weak anyway (see issue 3).

---

## 5. Failed programs were silently dropped from the search

**What.** The evaluator's error paths returned metrics without `model_complexity`, a MAP-Elites
feature dimension. OpenEvolve discards programs missing a feature dimension, logging
`Feature dimension 'model_complexity' not found`.

**Effect.** Failures never entered the database, so the search could not learn from them and the
island populations were smaller than the iteration count suggests. This changes search dynamics
rather than any reported number.

---

## 6. The improvement-suggestion channel was inert

**What.** The prompt template referenced `{improvement_suggestion}`, which OpenEvolve substitutes
only in *user* templates. With no `templates/` directory, the placeholder was never filled.

**Effect.** The intended mechanism for feeding evaluator feedback back to the LLM did nothing; the
model saw the literal placeholder or nothing at all.

---

## 7. `np.seterr(all='raise')` rejected sound programs

**What.** The evaluator escalated all floating-point warnings to exceptions. Underflow in
`logsumexp` is harmless and routine — and becomes *more* frequent as a model fits better, because
confident predictions produce very small probabilities.

**Effect.** Good candidates could be scored as failures, with a bias against the better ones.

---

## 8. Data pipeline defects

- **`loadAndSplitData.py` did not run as published**: a missing comma made it a `SyntaxError`. So
  the data preparation step could not have been executed in the form committed.
- **Choices were stored as floats.** The `.npy` files were float, so programs used floating-point
  values as array indices. Now written as `int64`.
- **Text export produces empty files against the current Psych-101.** The dataset stores
  `participant` as a string (`'0'`, `'1'`, `'10'`), the per-experiment parquet as `int64`, so
  `isin` matched nothing and every `*_text_*.csv` was written with a header and no rows — silently,
  since exporting an empty frame succeeds. *This blocks reproduction today; whether it affected the
  original run depends on the dataset version then, and the published run clearly did have text, so
  this is most likely a change in the dataset rather than a defect in the original results.*

---

## Reproduction hazards (not defects in the published results)

Recorded because they cost significant time and will recur.

- **Model name mismatch → silent no-op search.** If `--primary-model` does not match what vLLM
  serves, every request 404s. A 600-iteration run produced only the initial program. The 404s go to
  `errors_*.txt`, not the run log, so the search looks alive. `OpenEvolve/summariseRun.py` now reads
  both.
- **Context-length overruns.** 19,755 context-length errors before setting `--max-model-len 65536`
  and quoting fewer programs in the prompt.
- **JAX with forked workers deadlocks.** Requires `max_tasks_per_child` to force spawn.
- **One output directory per run.** OpenEvolve does not auto-resume, but stale program files
  accumulate and contaminate a later summary.

---

## Open questions

1. **Does fine-tuning actually help the search?** Untested with a valid adapter. Needs a fine-tuned
   run *and* a base run under identical current settings — comparing against the older `Base4000`
   would confound the adapter with the search changes.
2. **Maggie's Farm.** Both conditions do poorly on the only three-option task, which never appears
   in the search. Whether this is a generalisation failure or an interface artefact (`num_options`
   varying) is untested; the prompt does not mention that it varies.
3. **Whether the original `completion_mask` was honoured by the trl version used then.** The
   evidence above says the published loss was full-sequence, but the exact trl version is not
   recorded, and older versions consumed mask columns in the collator. The conclusion rests on the
   loss and accuracy values, not on version archaeology.

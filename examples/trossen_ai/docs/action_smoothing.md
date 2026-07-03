# Action Smoothing — Temporal Ensemble & CogACT

How overlapping action-chunk predictions get blended into one commanded action,
and what the **CogACT** method's three modes (`cogact` / `latest` / `hybrid`) do.

Source of truth: [`ensemble/__init__.py`](../ensemble/__init__.py). This doc explains
the *why*; the code is the *what*.

---

## 1. The problem: chunks overlap

A single policy inference returns a **chunk** of `T` future actions (e.g. 16
steps). The control loop does **not** run a whole chunk before asking again — it
re-queries every `rate_of_inference` steps (or every step in async mode). So at
any given control step, several recently predicted chunks all contain a
prediction for *that same step*:

```
            step:   t   t+1  t+2  t+3  t+4
 chunk @ t        a[t,0] a[t,1] a[t,2] a[t,3] a[t,4]
 chunk @ t+1             a[t+1,0] a[t+1,1] a[t+1,2] a[t+1,3]
 chunk @ t+2                      a[t+2,0] a[t+2,1] a[t+2,2]
 chunk @ t+3                               a[t+3,0] a[t+3,1]
                                  ▲
                     step t+2 has 3 overlapping predictions:
                     a[t,2], a[t+1,1], a[t+2,0]
```

Executing only the newest chunk makes motion **jerky and noisy**: a stochastic
policy returns a slightly different trajectory every call, so the commanded
action jumps at each re-query boundary. An **ensemble** averages the overlapping
predictions for the current step into one command. Averaging cancels
per-inference noise **without adding lag**, because we always include the newest
prediction — we just don't trust it alone.

This is exactly what the **Chunk overlap map** in the Live UI's *Action
Smoothing* card draws, live.

The only thing that separates the two smoothing methods is **how each
overlapping prediction is weighted** before the weighted average.

---

## 2. Shared buffer semantics (both methods)

Both `TemporalEnsemble` and `CogACTEnsemble` share the same buffer behaviour, so
they are drop-in swappable:

- Chunks are kept **whole** until fully consumed — an overlap is never evicted
  before the step that needs it.
- **Thread-safe** — the async inference worker adds chunks from a background
  thread while the control loop reads blended actions.
- `add_chunk(query_step, chunk)` after each inference · `get_action(current_step)`
  returns the blend (or `None` if nothing overlaps) · `reset()` per episode.
- `last_weights()` exposes the weights of the most recent blend for telemetry
  (this drives the **Blend weights** bar in the UI).

`make_ensemble(smoothing=False)` returns `None` → **no smoothing**: the loop uses
the newest chunk's action directly. (Async inference *requires* a non-None
ensemble.)

---

## 3. Method A — Temporal (exp-decay by age)

`smoothing_method = "temporal"` → `TemporalEnsemble`. Classic ACT-style
temporal ensembling. Overlapping predictions are ordered **oldest-query-first**
(`k = 0` = oldest chunk still overlapping) and weighted:

```
weight[k] ∝ exp(-decay · k)      then normalized to sum 1
```

One knob — `smoothing_decay`:

| `decay` | Effect |
|--------|--------|
| `0`     | Plain average of all overlaps (equal trust). |
| larger  | Trusts **older, already-committed** predictions more → smoother, less reactive. |

Temporal weighting looks only at **age**. It does not look at whether a
prediction agrees with the others — an outlier chunk still gets its age-based
share. That is the gap CogACT closes.

---

## 4. Method B — CogACT (consensus by agreement)

`smoothing_method = "cogact"` → `CogACTEnsemble`
(CogACT / Adaptive Action Ensemble, [arXiv 2411.19650](https://arxiv.org/abs/2411.19650)).

Instead of weighting by age, CogACT weights each overlapping prediction by **how
much it agrees with the others**, measured as **cosine similarity** between the
predicted action vectors. A prediction that points the same way as the consensus
gets a high weight; an **outlier** that disagrees gets suppressed. This is
adaptive: on a clean step every chunk agrees and it behaves like an average; on a
step where one inference glitched, the glitch is down-weighted automatically.

Mechanics (per overlapping step):
1. Unit-normalize each overlapping prediction: `â_i = a_i / ‖a_i‖`.
2. Similarity matrix `S = normed @ normedᵀ` (cosine sim of every pair).
3. Turn similarities into weights (see the three modes below).
4. Clip negatives to 0, normalize to sum 1, weighted-average.
5. Single overlap (`N = 1`) → weight `[1.0]`, return it unchanged.

### The three CogACT modes (`cogact_mode`)

Selected in the Live UI *only* when method = CogACT.

| Mode | Weight for prediction *i* | Meaning |
|------|---------------------------|---------|
| **`cogact`** (default) | `mean_j cos(a_i, a_j)` — row-mean of `S` | **Pure consensus.** Each prediction weighted by its average agreement with *all* overlapping predictions. Best outlier rejection; the "democratic" vote. |
| **`latest`** | `cos(a_i, a_newest)` | **Anchor to newest.** Weight = agreement with the most recent prediction. Most reactive — follows the freshest inference, keeps others only insofar as they agree with it. |
| **`hybrid`** | `λ · consensus + (1−λ) · latest` | **Blend of both.** `λ = lambda_mix` (default `0.5`). Trades consensus stability against latest-anchored reactivity. |

Notes:
- `latest` uses agreement with the **newest** overlapping prediction (last after
  the oldest-first sort), so it stays responsive while still discarding noise
  that disagrees with the fresh command.
- If all weights collapse to ~0 (degenerate step), CogACT falls back to a plain
  uniform average (`1/N`) rather than dividing by zero.

---

## 5. Choosing

| You want… | Use |
|-----------|-----|
| Simple, well-understood smoothing with one knob | **Temporal**, tune `decay`. |
| A single stochastic inference occasionally spikes / flips | **CogACT `cogact`** — consensus rejects the outlier. |
| Maximum responsiveness, still denoised | **CogACT `latest`**. |
| Middle ground, one dial to trade off | **CogACT `hybrid`**, tune `lambda_mix`. |

All of this is action-space agnostic: it blends the **decoded joint** chunks, so
it works identically for Joint and End-effector action spaces (EE chunks are
decoded to joints via IK *before* they reach the ensemble).

---

## 6. Where to see it live

Live UI → **Action Smoothing** card (hover the ⓘ for the in-app summary):

- **Smoothing Δ** — `‖blended − newest raw‖` (rad): how far the blend moved the
  command off the raw newest prediction = how much work smoothing is doing.
- **Accumulation** — overlap count at the current step (blend inputs).
- **Blend weights** — the normalized per-chunk weights (`last_weights()`).
- **Chunk overlap map** — the grid above, live: rows = recent chunks, columns =
  steps, cell fade = recency.

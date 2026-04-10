# ReZero — STU-augmented DreamerV3 fork

This is a fork of [`NM512/r2dreamer`](https://github.com/NM512/r2dreamer)
(an efficient PyTorch DreamerV3 / R2-Dreamer implementation) with an
optional **`STUEmbedMixer`** added: a forward-path STU sequence mixer
that takes the encoder output `[B, T, E]` and mixes it along the time
axis before the RSSM consumes it.

## Integration model

The integration mirrors the spectral-transformer-jax usage of STU
(`spectral-transformer-jax/src/stj/models/blocks.py::STUSandwichBlock`),
where STU is a first-class non-attention sequence mixer inside a residual
transformer-style block. That repo's STU usage was shown to be useful for
state tracking, which is the property we want to test here.

Vanilla R2-Dreamer flow with the one addition (`*`):

```
embed = encoder(data)              # [B, T, E]
embed = stu_mixer(embed)           # [B, T, E]      *  causal Hankel_L
post_stoch, post_deter, _ = rssm.observe(embed, ...)
```

The mixer is a stack of `STUSandwichBlock` modules. Each block is a
pre-norm residual sandwich:

```
sublayer 1:  x = x + LayerScale * FFN_in(LN(x))     # per-position
sublayer 2:  x = x + LayerScale * STU(LN(x))        # sequence mixing
sublayer 3:  x = x + LayerScale * FFN_out(LN(x))    # per-position
```

`LayerScale` is initialized to `1e-4` so the mixer is approximately the
identity at init — `model.stu_mixer.enabled=true` does not perturb the
initial training trajectory of vanilla R2-Dreamer; the mixer only
contributes if/when training pulls the LayerScale gammas up.

### Causality and the inference rolling buffer

The STU is built with `use_hankel_L=True`, the *causal* Hankel basis Z_L
from the STU paper. Causality means each output position t depends only
on input positions ≤ t. We need this so that:

- **Training** runs the mixer once over the full `[B, T, E]` batch
  sequence (T = `batch_length`, default 64) in parallel.
- **Inference** in `Dreamer.act` maintains a rolling embed history
  buffer of length `seq_len` in the agent state (`embed_history`,
  zero-initialized and reset on `is_first`). At each step the new embed
  is appended, the mixer runs over the full window, and the **last**
  position is fed to `rssm.obs_step`.

With causal filters, **at steady state** (after `seq_len-1` steps from
an episode reset, i.e., once the rolling buffer is full of real embeds)
the inference last-position output equals what training computes for
the final position of a length-`seq_len` input window with the same
contents. Before steady state — for the first `seq_len-1` steps of each
episode — the inference buffer contains zero-padding that does not match
how training position t (for t < seq_len-1) is computed; this is a
warm-up region where train and inference are not bit-equivalent.

A frozen copy of the mixer (`_frozen_stu_mixer`) is maintained alongside
`_frozen_encoder` / `_frozen_rssm` and refreshed in `clone_and_freeze`.

## What's been added vs vanilla r2dreamer

**New files**:
- `stu_dynamics/__init__.py`
- `stu_dynamics/stu_layer.py` — `MiniSTU` (FFT-based STU layer, supports
  causal Hankel_L)
- `stu_dynamics/filter_factory.py` — `make_filters(kind, ...)` for the
  6 filter ablation bases
- `stu_dynamics/stu_dynamics.py` — `STUSandwichBlock`, `STUEmbedMixer`,
  `LayerScale`

**Modified files**:
- `dreamer.py`:
  - imports `STUEmbedMixer`
  - constructs `self.stu_mixer` in `__init__` when enabled
  - registers it with the optimizer module dict
  - clones it into `_frozen_stu_mixer` in `clone_and_freeze`
  - applies it to `embed` in `_cal_grad` and the `dreamerpro` `embed_aug`
    branch before `rssm.observe`
  - adds `embed_history` to the agent state in `get_initial_state` and
    rolls it inside `act`
- `configs/model/_base_.yaml` — adds the `stu_mixer.*` config block
  (replacing the old `stu_dec.*` side-loss block)

The vanilla DreamerV3 / R2-Dreamer code paths are 100% backward
compatible: with `model.stu_mixer.enabled=false` (the default), the
behavior is identical to upstream r2dreamer.

## Setup

```bash
cd /path/to/ReZero
# Python 3.11 required (per upstream README).
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Smoke test (GPU recommended, but module-level test runs on CPU)

```bash
python -c "
import sys; sys.path.insert(0, '.')
import torch
from stu_dynamics import STUEmbedMixer, MiniSTU, make_filters

# Realistic Atari-12M shape
m = STUEmbedMixer(embed_dim=1024, seq_len=64, num_layers=2, num_filters=8, d_model=256)
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f'STUEmbedMixer params: {n:,}')

x = torch.randn(4, 64, 1024)
y = m(x)
assert y.shape == x.shape
print(f'init residual norm fraction: {((y - x).norm() / x.norm()).item():.3e}')

# Streaming-vs-parallel parity test (the key correctness claim)
m.eval()
buf = torch.zeros(4, 64, 1024)
for t in range(64):
    buf = torch.cat([buf[:, 1:], x[:, t:t+1]], dim=1)
    last = m(buf)[:, -1]
parallel = m(x)[:, -1]
print(f'streaming vs parallel last-position diff: {(last - parallel).abs().max().item():.3e}')

# Filter sanity check
for k in ['hankel', 'hankel_scaled', 'dct', 'dft', 'random_normalized', 'random']:
    f = make_filters(k, seq_len=64, num_filters=8, use_hankel_L=True, seed=42)
    norms = torch.linalg.norm(f, dim=0)
    print(f'  {k:18s} col_norms[:3]={[round(v.item(),4) for v in norms[:3]]}')
"
```

Expected: `~2.6M params`, residual norm fraction `~9e-3` (small but
non-zero so gradients flow to every interior parameter from step 0),
and streaming/parallel parity diff exactly `0.0` (the causal Hankel_L
filters guarantee bit-equivalence between training and the inference
rolling buffer at steady state).

## Atari 100K — running an experiment

The standard Atari 100K benchmark = 100K agent decisions × 4
action_repeat = 400K env frames. The existing
`configs/env/atari100k.yaml` already has `steps: 4.1e5`.

### Baseline (vanilla R2-Dreamer)

```bash
python train.py \
  env=atari100k \
  env.task=atari_pong \
  logdir=./logdir/pong_baseline_seed0 \
  seed=0
```

### Treatment (STUEmbedMixer with `hankel_scaled` filters)

```bash
python train.py \
  env=atari100k \
  env.task=atari_pong \
  model.stu_mixer.enabled=true \
  model.stu_mixer.filter_type=hankel_scaled \
  model.stu_mixer.num_layers=2 \
  model.stu_mixer.num_filters=8 \
  logdir=./logdir/pong_stu_mixer_hankelscaled_seed0 \
  seed=0
```

`stu_mixer.seq_len` defaults to `${batch_length}`; if you change
`batch_length` (default 64) the interpolation tracks it automatically.

### Filter ablation

`model.stu_mixer.filter_type` can be one of:

- `hankel` — top-K Hankel eigenvectors (the canonical STU basis)
- `hankel_scaled` — Hankel directions rescaled to `sqrt(seq_len)` column
  norms (the global best from the offline STUZero study)
- `dct` — top-K DCT-II cosines, scaled to match Hankel column norms
- `dft` — lowest-K Fourier modes (cos/sin), scaled to match Hankel
- `random_normalized` — Haar-orthonormal columns scaled to Hankel
- `random` — i.i.d. Gaussian, no normalization (unnormalized control)

Headline ablation: `hankel`, `hankel_scaled`, `dct`, `random_normalized`
× `seed ∈ {0, 1, 2}` × Pong = 12 runs.

## Monitoring

```bash
tensorboard --logdir ./logdir
```

Watch:
- `episode/eval_score` and `episode/train_score` — the actual RL signal
  the experiment is testing
- Existing world-model losses (`loss/dyn`, `loss/rep`, etc.) — these
  should at minimum not regress vs the baseline
- Gradient norms on `stu_mixer.*` params — the LayerScale gammas
  starting at `1e-4` should grow if the mixer is learning to contribute

There is **no** `loss/stu_mixer` curve because the mixer is part of the
forward path, not a side loss — its gradient signal flows through
`loss/dyn`, `loss/rep`, and the actor/critic objectives.

## Things to verify on the first real run

1. **No shape mismatch in `_cal_grad`.** The mixer expects
   `[B, batch_length, embed_size]`. If you change `batch_length` without
   updating `model.stu_mixer.seq_len` (or relying on the
   `${batch_length}` interpolation), construction will succeed but the
   forward will assert at training time.
2. **`get_initial_state` produces the rolling buffer.** Inspect the
   state TensorDict after `agent.get_initial_state(env_num)`; it should
   contain an `embed_history` field of shape
   `(env_num, seq_len, embed_size)`.
3. **`act` rolls the buffer correctly across episodes.** Check that on
   `is_first=True` steps the history is zeroed for that env index.
4. **Wall clock.** A 2-layer mixer over `[B=16, T=64, E=512]` is cheap
   (a handful of FFT-of-128 + GEMMs) and shouldn't move the per-step
   training time noticeably. Inference adds one mixer pass per step, but
   the actor/critic dominates.
5. **GPU memory.** The mixer is small (a few hundred K params for the
   default `num_layers=2`, `num_filters=8`, `d_model=embed_size`).
   Inference adds a `[B, seq_len, E]` tensor in agent state.

## What this experiment tests

**Hypothesis**: training Dreamer's encoder with a *causal* STU sequence
mixer between the encoder and the RSSM produces a better world model
than vanilla R2-Dreamer / DreamerV3, because STU's spectral filtering is
a useful inductive bias for state tracking (as observed in the
`spectral-transformer-jax` work).

**Mechanism**: at each timestep the RSSM now sees an embed that has
been temporally mixed via STU over the entire prior context (causally).
The encoder is trained end-to-end with this extra mixer in the loop,
which gives gradients that account for cross-time spectral structure.
At inference the same mixer runs over a rolling buffer so the
observation pipeline is identical.

**What this does NOT test**: STU as a replacement for the RSSM's
recurrent dynamics (the GRU). The RSSM is unchanged. A more invasive
follow-up would replace `Deter` with an STU-based recurrence — natural
if the mixer-only integration shows positive signal.

## Known limitations

- **Episode warm-up.** The mixer is built with a fixed `seq_len`. The
  first `seq_len-1` steps of each episode (both at inference and the
  early positions of every training batch row) run on a partially-filled
  (zero-padded) buffer. With causal filters the most recent position is
  only mildly affected by the zero pad, but train and inference are not
  bit-equivalent during this warm-up region.
- **Cross-episode contamination at training time.** If a sampled batch
  row spans an episode reset (which the replay buffer does not currently
  prevent), the mixer's spectral convolution will mix embeds from the
  pre-reset episode into the post-reset positions. We mask the immediate
  reset position back to the raw embed, but the convolutional "tail"
  still bleeds for up to `seq_len` steps. With Atari batch_length=64 and
  episode lengths in the thousands this is on the order of a few percent
  of training samples. The inference path is fine because we zero
  `embed_history` on `is_first`.
- `_video_pred` (used only for tensorboard video logging) does NOT
  apply the mixer because it slices `embed[:, :5]` and the mixer
  expects exactly `seq_len`. Fix is straightforward if you need it.
- Causality is enforced via `use_hankel_L=True`. The non-causal Hankel
  basis is not exposed by the mixer because it would break train/inference
  parity.

## Linking back to the offline study

Filter-basis intuition comes from
[STUZero/research_logs/logs/0409026.md][offline-log]. The headline
finding (Pong 5-episode multistep dynamics MSE) was that
`hankel_scaled` was the global best, beating plain `hankel`, `dct`, and
`random_normalized` at matched column scale. That offline study used
STU as an open-loop seq2seq dynamics decoder, which is *not* the
integration here — the present integration uses STU as a forward-path
sequence mixer over encoder outputs, which is closer to what was
shown to work in `spectral-transformer-jax`. Whether the offline filter
ranking transfers to the in-forward-path setting is an open question
the filter ablation will answer.

[offline-log]: ../STUZero/research_logs/logs/0409026.md

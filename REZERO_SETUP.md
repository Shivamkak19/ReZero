# ReZero — STU-augmented DreamerV3 fork

This is a fork of [`NM512/r2dreamer`](https://github.com/NM512/r2dreamer)
(an efficient PyTorch DreamerV3 / R2-Dreamer implementation) with an
optional **`STUDeterPredictor`** side network added: a JAX-faithful
sequence-to-sequence dynamics predictor that takes `(initial_deter,
action_sequence)` and predicts the entire deterministic-state
trajectory in a single forward pass.

The integration model is **STUDeterPredictor as auxiliary multi-step
training loss** (mirrors the EZv2 STUDecoder-aux integration):

- Standard RSSM is unchanged. It still drives the imagination loop and
  the actor-critic.
- A side network `STUDeterPredictor` (~1.8M params) is constructed
  alongside RSSM, gated by a config flag.
- During each training step, after the standard `rssm.observe(...)`
  posterior rollout, `STUDeterPredictor` is called once with
  `(post_deter[:, 0], data["action"][:, 1:])` and predicts
  `pred_deter` of shape `[B, T-1, deter_dim]`.
- An MSE loss is computed between `pred_deter` and the (detached)
  `post_deter[:, 1:]`.
- The MSE loss is added to the total optimizer loss with coefficient
  `loss_scales.stu_dec`. Encoder gradients flow back via the
  initial-state input.
- The MCTS / imagination loop is **never** affected — STU is only used
  at training time as a side prediction objective.

## What's been added vs vanilla r2dreamer

**New files**:
- `stu_dynamics/__init__.py`
- `stu_dynamics/stu_layer.py` — `MiniSTU` (FFT-based STU layer)
- `stu_dynamics/filter_factory.py` — `make_filters(kind, ...)` for the
  6 filter types (`hankel`, `random`, `random_normalized`, `dct`,
  `dft`, `hankel_scaled`)
- `stu_dynamics/stu_dynamics.py` — `STUResBlock` and
  `STUDeterPredictor`

**Modified files**:
- `dreamer.py` — adds STU import, conditional construction in
  `Dreamer.__init__`, optimizer module registration, and the auxiliary
  loss in `_cal_grad` after `rssm.observe(...)`
- `configs/model/_base_.yaml` — adds `stu_dec.*` config block and
  `loss_scales.stu_dec`

The vanilla DreamerV3 / R2-Dreamer code paths are 100% backward
compatible: with `model.stu_dec.enabled=false` (the default), the
behaviour is identical to upstream r2dreamer.

## Setup (on the GPU machine where you'll actually train)

```bash
cd /path/to/ReZero
# Python 3.11 is required (per upstream README).
# We use uv but you can use any venv tool.
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Verify the integration constructs without errors (no GPU needed):

```bash
python -c "
import sys; sys.path.insert(0, '.')
import torch
from stu_dynamics import STUDeterPredictor, make_filters, MiniSTU

m = STUDeterPredictor(
    deter_dim=2048, action_space_size=6, action_dim=6,
    max_action_seq_len=64, d_model=128, num_stu_layers=4, num_filters=8,
    is_continuous=True,
)
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f'STUDeterPredictor params: {n:,}')

# Filter sanity check
for k in ['hankel', 'random', 'random_normalized', 'dct', 'dft', 'hankel_scaled']:
    f = make_filters(k, seq_len=65, num_filters=8, seed=42)
    norms = torch.linalg.norm(f, dim=0)
    print(f'  {k:18s} col_norms[:3]={[round(v.item(),4) for v in norms[:3]]}')
"
```

Expected output:
```
STUDeterPredictor params: 1,841,920
  hankel             col_norms[:3]=[0.0274, 0.0434, 0.0669]
  random             col_norms[:3]=[6.8683, 6.5759, 7.0218]
  random_normalized  col_norms[:3]=[0.0274, 0.0434, 0.0669]
  dct                col_norms[:3]=[0.0274, 0.0434, 0.0669]
  dft                col_norms[:3]=[0.0274, 0.0434, 0.0669]
  hankel_scaled      col_norms[:3]=[8.0623, 8.0623, 8.0623]
```

## Atari 100K — running an experiment

The standard Atari 100K benchmark = 100K agent decisions × 4
action_repeat = 400K env frames. The existing `configs/env/atari100k.yaml`
already has `steps: 4.1e5`, which matches.

### Baseline (vanilla R2-Dreamer, no STU)

```bash
python train.py \
  env=atari100k \
  env.task=atari_pong \
  logdir=./logdir/pong_baseline_seed0 \
  seed=0
```

### Treatment (STUDeterPredictor with `hankel_scaled` filters)

```bash
python train.py \
  env=atari100k \
  env.task=atari_pong \
  model.stu_dec.enabled=true \
  model.stu_dec.filter_type=hankel_scaled \
  model.stu_dec.d_model=128 \
  model.stu_dec.num_layers=4 \
  model.stu_dec.num_filters=8 \
  model.stu_dec.max_action_seq_len=64 \
  model.loss_scales.stu_dec=1.0 \
  logdir=./logdir/pong_stu_decoder_hankelscaled_seed0 \
  seed=0
```

**`max_action_seq_len` must be ≥ `batch_length - 1`.** The default
`batch_length=64` means `max_action_seq_len=64` works. If you change
`batch_length`, change this accordingly.

### Filter ablation

To match the offline ablation matrix from the STUZero study, run with
`model.stu_dec.filter_type` set to each of:

- `hankel` — top-K Hankel eigenvectors (the canonical STU basis)
- `hankel_scaled` — Hankel directions scaled to `sqrt(seq_len)` column
  norms (the global best from the offline study; recommended default)
- `dct` — top-K DCT-II cosines, scaled to match Hankel column norms
- `dft` — lowest-K Fourier modes (cos/sin), scaled to match Hankel
- `random_normalized` — Haar-orthonormal columns scaled to Hankel
- `random` — i.i.d. Gaussian, no normalization (the unnormalized control)

The **headline filter ablation** to run if you have compute budget:
`hankel`, `hankel_scaled`, `dct`, `random_normalized` × `seed ∈ {0, 1, 2}`
× **same game** = 12 runs, 1 game (Pong is the cheapest). Each run is
~6-12 hours wall clock on a single A100.

### Multi-game

```bash
for game in atari_pong atari_breakout atari_asterix atari_mspacman; do
  python train.py \
    env=atari100k \
    env.task=$game \
    model.stu_dec.enabled=true \
    model.stu_dec.filter_type=hankel_scaled \
    logdir=./logdir/${game}_stu_seed0 \
    seed=0
done
```

Atari 100K spec: 26 games. Pong is the easiest to debug; if STU shows
positive signal on Pong, expand to a 5- or 10-game subset, then full 26.

## Monitoring

```bash
tensorboard --logdir ./logdir
```

Look for the new `loss/stu_dec` curve in the loss panel — that's the
STUDeterPredictor MSE against the posterior deter trajectory. It
should decrease alongside `loss/dyn` and `loss/rep`. If `loss/stu_dec`
plateaus much higher than `loss/dyn`, STUDeterPredictor isn't keeping
up with RSSM and the auxiliary signal is mostly noise. If it tracks
RSSM closely, the side network is learning a similar trajectory
representation.

The score curves to watch are `episode/eval_score` (from the eval
worker, every `trainer.eval_every` steps) and `episode/train_score`
(from the train worker). Atari 100K reports the **mean of last 10
eval episodes at 100K env steps** as the headline number.

## Things that need verification on the first real run

These are checks I would do on the first end-to-end run that I
couldn't do locally because the EZv2 experiment is hogging the GPU on
this machine:

1. **The first `update_weights` call doesn't crash.** The integration
   is verified at construction time (model builds, optimizer sees the
   new params, `STUDeterPredictor` forward+backward work in isolation),
   but the first time the auxiliary loss is computed inside
   `_cal_grad`, there could be a shape mismatch or device issue that
   I haven't caught. If you see a torch error in the first 100 steps,
   send me the traceback.
2. **`loss/stu_dec` is non-trivial.** It should start at some positive
   number and decrease over training. If it stays at 0 or NaN, the
   loss path isn't actually firing.
3. **Total wall clock for Pong 100K** with the side network. Adding
   1.8M params (~13% of total) should not slow training by more than
   ~15-20%. If it's 2× slower, something is wrong.
4. **GPU memory increase.** STUDeterPredictor adds ~1.8M params and a
   forward pass per training step. Expect ~1-2 GB of additional GPU
   memory vs vanilla. If it's much more, there might be an unintended
   activation retention issue.

## What this experiment tests

**Hypothesis**: training the world model with an auxiliary
sequence-to-sequence STU prediction loss (Hankel basis, ideally
`hankel_scaled`) improves Atari 100K eval scores compared to vanilla
R2-Dreamer / DreamerV3.

The mechanism: STUDeterPredictor is trained to predict the entire
deterministic latent trajectory in one shot from `(initial_state,
action_seq)`. Its gradients flow back through the encoder, providing
a parallel rollout supervision in addition to the standard RSSM
recurrent unroll. If STU's spectral basis is genuinely useful for
modeling latent dynamics (as the offline STUZero study suggests), the
encoder should learn a better representation, and the policy/value
should improve as a result.

**What this does NOT test**: STUDeterPredictor as the primary dynamics
function in the imagination loop. That would require a more invasive
refactor (open-loop planning where the policy outputs an entire action
sequence and STU predicts the resulting trajectory in one shot). If
the auxiliary loss approach shows positive signal, the open-loop
variant is the natural follow-up.

## Known limitations / context

- The integration assumes `is_continuous=True` for STUDeterPredictor's
  action encoder, since Atari one-hot actions are stored as
  `[B, T, action_space_size]` float vectors in the replay buffer (not
  as `[B, T]` int tensors). This works for any environment that uses
  Dreamer's `OneHotAction` wrapper (Atari, Crafter, MemoryMaze) and
  for naturally continuous environments (DMC, MetaWorld).
- `max_action_seq_len` is a fixed compile-time parameter for the FFT
  conv length. The model can be called with shorter sequences (it
  pads internally), but not longer. Set it >= `batch_length - 1`.
- The ported `MiniSTU` and `filter_factory` are byte-identical to the
  STUZero versions. Any STU-related research log entries from
  `STUZero/research_logs/logs/0409026.md` apply directly.

## Linking back to the offline study

The architectural choice (`STUDeterPredictor` with `hankel_scaled`
filters) is grounded in the offline ablation matrix from
[STUZero/research_logs/logs/0409026.md][offline-log]. Headline numbers
from that study (Pong 5-episode dynamics multistep MSE):

| filter             | Pong-5ep step-50 rollout MSE (STUDecoder, mean over 2 seeds) |
|---|---|
| `hankel`             | 0.001026                                |
| `dct`                | 0.001786 (+74% worse)                   |
| `random_normalized`  | 0.001801 (+76% worse)                   |
| **`hankel_scaled`**  | **0.000683 (-33% best)**                |

The offline study showed Hankel directions matter (vs DCT/random_norm
at matched scale), AND scale matters (`hankel_scaled` > Hankel). The
combination is the global best. Whether this offline ranking transfers
to actual Atari 100K eval scores is exactly what the runs in this
repo will determine.

[offline-log]: ../STUZero/research_logs/logs/0409026.md

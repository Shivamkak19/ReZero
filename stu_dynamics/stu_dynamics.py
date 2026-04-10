"""STU sequence mixer for the Dreamer encoder→RSSM path.

This module is modeled directly on the spectral-transformer-jax block design
(`spectral-transformer-jax/src/stj/models/blocks.py::STUSandwichBlock`), where
STU is used as a first-class non-attention sequence mixer inside a residual
transformer-style block, alongside FFN sublayers. That repo's STU usage was
shown to be useful for state tracking, which is the property we want here.

Integration model used in ReZero
--------------------------------
Vanilla R2-Dreamer flow, with one addition (marked ``*``)::

    embed = encoder(data)              # [B, T, E]
    embed = stu_mixer(embed)           # [B, T, E]    *
    post_stoch, post_deter, _ = rssm.observe(embed, ...)

The mixer is a stack of ``STUSandwichBlock`` modules. Each block is::

    sublayer 1:  x = x + LayerScale * FFN_in(LN(x))     # per-position
    sublayer 2:  x = x + LayerScale * STU(LN(x))        # sequence mixing
    sublayer 3:  x = x + LayerScale * FFN_out(LN(x))    # per-position

This is the FFN-STU-FFN "sandwich" residual block from the jax repo. The STU
sublayer is the only place where information crosses time positions; the FFNs
operate per-position. ``LayerScale`` is initialized to a small value (1e-4
by default, matching the jax repo's ``layer_scale_init``) so the mixer behaves
as the identity at initialization and gradually learns to contribute.

Causality
---------
The STU is configured with ``use_hankel_L=True``, i.e. the *causal* Hankel
matrix Z_L from the STU paper. This means each output position t depends only
on input positions ≤ t. Causality is required so that training (where the
mixer sees the entire batch sequence in parallel) is bit-equivalent to
inference (where the mixer sees a rolling window of past embeds in the agent
state). Without causality the model would leak future-step information at
training time and could not be deployed online.

Filter swap
-----------
After construction, the spectral filters of every ``MiniSTU`` instance can be
swapped to one of the ablation bases via ``filter_factory.make_filters``
(``hankel`` / ``hankel_scaled`` / ``dct`` / ``dft`` / ``random_normalized`` /
``random``). The headline filter from the offline STUZero study was
``hankel_scaled``; we keep that path available for ablation runs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .stu_layer import MiniSTU


class LayerScale(nn.Module):
    """Per-channel learnable scale, initialized small.

    Mirrors the ``LayerScale`` used in spectral-transformer-jax (and CaiT).
    With a small initial value the residual sublayer starts as a near-identity
    perturbation, which lets us insert STU into a pretrained-shaped model
    without disturbing initial dynamics.
    """
    def __init__(self, dim: int, init: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class STUSandwichBlock(nn.Module):
    """FFN → STU → FFN residual block, mirroring the jax repo.

    See ``spectral-transformer-jax/src/stj/models/blocks.py::STUSandwichBlock``.
    Each sublayer is pre-norm and gated by a ``LayerScale``. STU is the only
    sequence mixer; the two FFNs operate per-position.
    """
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        num_filters: int,
        mlp_ratio: float = 2.0,
        layer_scale_init: float = 1e-4,
        use_hankel_L: bool = True,  # causal by default
    ):
        super().__init__()
        hidden = int(d_model * mlp_ratio)

        self.norm1 = nn.LayerNorm(d_model)
        self.ff_in = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.ls1 = LayerScale(d_model, init=layer_scale_init)

        self.norm2 = nn.LayerNorm(d_model)
        self.stu = MiniSTU(
            seq_len=seq_len,
            num_filters=num_filters,
            input_dim=d_model,
            output_dim=d_model,
            use_hankel_L=use_hankel_L,
            dtype=torch.float32,
            device=None,
        )
        self.ls2 = LayerScale(d_model, init=layer_scale_init)

        self.norm3 = nn.LayerNorm(d_model)
        self.ff_out = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.ls3 = LayerScale(d_model, init=layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        x = x + self.ls1(self.ff_in(self.norm1(x)))
        x = x + self.ls2(self.stu(self.norm2(x)))
        x = x + self.ls3(self.ff_out(self.norm3(x)))
        return x


class STUEmbedMixer(nn.Module):
    """Stack of ``STUSandwichBlock`` operating on encoder embeddings.

    Inserts STU as a temporal mixer between the encoder and the RSSM. Input
    and output shape are both ``[B, L, embed_dim]``. If ``d_model`` differs
    from ``embed_dim`` the mixer projects in/out; otherwise it is purely
    additive on top of the embed (the residual structure of every block plus
    LayerScale-init keeps the identity path strong).

    Parameters
    ----------
    embed_dim
        Encoder output dimension. The mixer is a strict residual on top of an
        ``[B, L, embed_dim]`` tensor.
    seq_len
        Fixed temporal length used by the FFT path inside ``MiniSTU``. Should
        be set equal to the trainer's ``batch_length`` so that training and
        the inference rolling buffer use the exact same sequence length.
    num_layers
        Number of ``STUSandwichBlock`` stacked.
    num_filters
        Number of spectral filters K used in each ``MiniSTU``.
    d_model
        Internal block dimension. Defaults to ``embed_dim`` (no projections).
    mlp_ratio
        FFN hidden ratio inside each block.
    layer_scale_init
        Initial value for every ``LayerScale``. Small values keep the mixer
        near-identity at init.
    use_hankel_L
        If True (default), use causal Hankel filters Z_L. Required for the
        rolling-buffer inference path.
    """
    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        num_layers: int = 2,
        num_filters: int = 8,
        d_model: int | None = None,
        mlp_ratio: float = 2.0,
        layer_scale_init: float = 1e-4,
        use_hankel_L: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model) if d_model is not None else self.embed_dim
        self.num_filters = int(num_filters)
        self.num_layers = int(num_layers)

        # When d_model differs from embed_dim we project in/out. The output
        # projection is initialized to a *small* (but non-zero) Xavier scale
        # so the mixer is approximately the identity at init while still
        # propagating gradients to every interior parameter from step 0.
        # (A pure zero-init output_proj would block gradient flow to all
        # upstream layers until the projection itself moved off zero,
        # which is a slow bootstrap.)
        # When d_model == embed_dim the LayerScale-init small inside each
        # STUSandwichBlock already gives both properties, so we use Identity.
        if self.d_model != self.embed_dim:
            self.input_proj = nn.Linear(self.embed_dim, self.d_model)
            self.output_proj = nn.Linear(self.d_model, self.embed_dim)
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
            nn.init.xavier_uniform_(self.output_proj.weight, gain=1e-2)
            nn.init.zeros_(self.output_proj.bias)
            self._proj_residual = True
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()
            self._proj_residual = False

        self.layers = nn.ModuleList([
            STUSandwichBlock(
                d_model=self.d_model,
                seq_len=self.seq_len,
                num_filters=self.num_filters,
                mlp_ratio=mlp_ratio,
                layer_scale_init=layer_scale_init,
                use_hankel_L=use_hankel_L,
            )
            for _ in range(self.num_layers)
        ])

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """Mix the embed tensor along its time axis.

        embed: [B, L, embed_dim]  with L == self.seq_len
        returns: [B, L, embed_dim]

        Initialization is approximately the identity:
        - When ``d_model == embed_dim``: every ``STUSandwichBlock`` is itself
          a residual sublayer with ``LayerScale`` init small, so the stack
          output ≈ embed at init, and we return the stack output directly.
        - When ``d_model != embed_dim``: the output projection is
          zero-initialized, so the projected stack output = 0 at init and
          we add it as a residual to the raw embed.
        """
        assert embed.dim() == 3, f"expected [B, L, E], got {tuple(embed.shape)}"
        B, L, E = embed.shape
        assert L == self.seq_len, (
            f"STUEmbedMixer was constructed with seq_len={self.seq_len} but "
            f"received a sequence of length {L}. Use the rolling buffer at "
            f"inference time to maintain a fixed-length context."
        )
        assert E == self.embed_dim, f"expected embed_dim={self.embed_dim}, got {E}"

        h = self.input_proj(embed)
        for layer in self.layers:
            h = layer(h)
        if self._proj_residual:
            # delta starts at zero thanks to zero-init output_proj
            return embed + self.output_proj(h)
        # Identity projections + per-block residuals → stack ≈ identity at init.
        return h

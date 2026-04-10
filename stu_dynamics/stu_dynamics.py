"""STU-based dynamics predictor for Dreamer-style world models.

The headline finding from the STUZero offline ablation study (see
research_logs/logs/0409026.md in the STUZero repo) is that when STU is
used as a sequence-to-sequence dynamics decoder — taking an initial
state and an action sequence and predicting the entire future trajectory
in a single forward pass — the Hankel basis specifically matters by
~70% on multistep rollout MSE, and combining Hankel directions with
larger column scale (`hankel_scaled`) gives the global best result of
the entire study.

This module ports that architecture into ReZero (a Dreamer-V3 fork) so
we can test whether the offline finding transfers to actual Atari 100K
benchmark scores.

The integration model used here is "STU as auxiliary multi-step loss":
the standard Dreamer RSSM is trained as usual, and a side network
`STUDeterPredictor` is trained alongside to predict the deterministic
state trajectory in one shot. STUDeterPredictor's gradients flow back
through the encoder via its initial-state input, providing parallel
rollout supervision. The MCTS / imagination loop in Dreamer is
unaffected — STUDeterPredictor is never queried at action-selection
time, only at training-time loss computation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stu_layer import MiniSTU


class STUResBlock(nn.Module):
    """Pre-norm transformer-style block: STU sublayer + MLP sublayer.

    sublayer 1:  x = x + STU(LayerNorm(x))
    sublayer 2:  x = x + MLP(LayerNorm(x))

    The STU operates along the sequence (time) dimension. The MLP
    operates per-position. This is the JAX-repo-shaped block: STU
    replaces attention.
    """
    def __init__(self, d_model: int, seq_len: int, num_filters: int,
                 mlp_ratio: float = 2.0, use_hankel_L: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.stu = MiniSTU(
            seq_len=seq_len,
            num_filters=num_filters,
            input_dim=d_model,
            output_dim=d_model,
            use_hankel_L=use_hankel_L,
            dtype=torch.float32,
            device=None,
        )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.stu(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class STUDeterPredictor(nn.Module):
    """JAX-faithful STU dynamics predictor adapted for Dreamer's flat
    deterministic latent state.

    Given an initial deterministic state and an action sequence,
    predicts the entire trajectory of future deterministic states in a
    single forward pass. STU is the only sequence mixer.

    Architecture:
        encode(initial_deter) -> token 0 [B, 1, d_model]
        encode each action_k -> tokens 1..K [B, K, d_model]
        concat                                [B, K+1, d_model]
        N x STUResBlock(STU sublayer + MLP sublayer)
        LayerNorm out
        Linear decoder applied per output position 1..K
        return [B, K, deter_dim]

    forward(initial_deter, action_sequence) where:
        initial_deter:    [B, deter_dim]
        action_sequence:  [B, K] long (discrete) or [B, K, action_dim] (continuous)
        returns:          [B, K, deter_dim]
    """
    def __init__(
        self,
        deter_dim: int,
        action_space_size: int,
        action_dim: int = None,        # for continuous; None for discrete
        max_action_seq_len: int = 64,
        d_model: int = 128,
        num_stu_layers: int = 4,
        num_filters: int = 8,
        mlp_ratio: float = 2.0,
        use_hankel_L: bool = False,
        is_continuous: bool = False,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.action_space_size = action_space_size
        self.action_dim = action_dim
        self.is_continuous = is_continuous
        self.max_action_seq_len = max_action_seq_len
        self.d_model = d_model
        self.num_filters = num_filters

        # Per-state encoder: project flat deterministic state to d_model.
        self.state_encoder = nn.Linear(deter_dim, d_model)

        # Action token: embedding (discrete) or linear projection (continuous).
        if is_continuous:
            assert action_dim is not None
            self.action_proj = nn.Linear(action_dim, d_model)
        else:
            self.action_embed = nn.Embedding(action_space_size, d_model)

        # The fixed sequence length used by the STU FFT path is
        # max_action_seq_len + 1 (initial state token + action tokens).
        self.seq_len = 1 + max_action_seq_len

        self.layers = nn.ModuleList([
            STUResBlock(
                d_model=d_model,
                seq_len=self.seq_len,
                num_filters=num_filters,
                mlp_ratio=mlp_ratio,
                use_hankel_L=use_hankel_L,
            )
            for _ in range(num_stu_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.state_decoder = nn.Linear(d_model, deter_dim)

    def forward(self, initial_deter: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        """Single-shot multistep deterministic-state prediction.

        initial_deter:    [B, deter_dim]
        action_sequence:  [B, K] (discrete) or [B, K, action_dim] (continuous)
        returns:          [B, K, deter_dim]
        """
        B = initial_deter.shape[0]
        K = action_sequence.shape[1]
        assert K <= self.max_action_seq_len, (
            f"action_sequence length {K} exceeds max_action_seq_len "
            f"{self.max_action_seq_len}"
        )

        # Encode initial state to one token.
        s0_token = self.state_encoder(initial_deter).unsqueeze(1)            # [B, 1, d_model]

        # Encode each action to a token.
        if self.is_continuous:
            action_tokens = self.action_proj(action_sequence)                # [B, K, d_model]
        else:
            action_tokens = self.action_embed(action_sequence.long())        # [B, K, d_model]

        # Concat: [s0_token, action_0, ..., action_{K-1}]
        x = torch.cat([s0_token, action_tokens], dim=1)                      # [B, K+1, d_model]
        actual_seq_len = x.shape[1]

        # STUResBlock layers were initialized with seq_len = max_action_seq_len + 1.
        # If the actual sequence is shorter, pad with zeros at the end so the
        # FFT-conv works at a fixed length, then truncate back. This lets us use
        # a single trained model on variable K.
        if actual_seq_len < self.seq_len:
            pad_len = self.seq_len - actual_seq_len
            x = F.pad(x, (0, 0, 0, pad_len))

        for layer in self.layers:
            x = layer(x)

        x = self.norm_out(x)

        if actual_seq_len < self.seq_len:
            x = x[:, :actual_seq_len]

        # Decode each position; positions 1..K are predictions for s_1..s_K.
        out = self.state_decoder(x)                                          # [B, K+1, deter_dim]
        return out[:, 1:]                                                    # [B, K, deter_dim]

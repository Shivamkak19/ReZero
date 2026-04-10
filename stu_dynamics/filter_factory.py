"""Construct filter tensors for MiniSTU ablations.

Used to test whether the spectral structure of the Hankel eigenbasis is doing
real work in MiniSTU, or if any reasonable orthonormal low-rank conv basis
would suffice. Filter kinds:

- 'hankel'            : the standard top-K Hankel eigenvectors scaled by
                        sigma^(1/4). The default MiniSTU basis.
- 'random'            : i.i.d. Gaussian, no normalization. Drop-in random.
                        Confounded by init scale (column norms ~sqrt(seq_len)
                        instead of ~sigma^(1/4)).
- 'random_normalized' : Haar-orthonormal columns (QR-on-Gaussian with sign
                        de-bias) scaled by the SAME sigma^(1/4) factors as
                        Hankel. Same column-energy spectrum and orthogonality;
                        only the directions in R^seq_len differ. The clean
                        ablation of "do the Z-eigenvectors specifically matter?"
- 'dct'               : top K lowest-frequency DCT-II cosines, scaled by the
                        same Hankel sigma^(1/4) factors. Smooth low-frequency
                        orthonormal basis canonically optimized for natural
                        image-like data.
- 'dft'               : top K lowest-frequency Fourier modes as alternating
                        cos/sin columns, scaled by Hankel sigma^(1/4). Tests
                        whether plain Fourier directions match Hankel.
- 'hankel_scaled'     : Hankel directions, but column norms rescaled to
                        sqrt(seq_len) (~unnormalized random magnitude).
                        Tests "is unnormalized random's advantage explained
                        purely by larger scale, or also by non-orthogonality?"
                        If hankel_scaled matches/beats random, scale is the
                        whole story. If random still wins, non-orthogonality
                        also contributes.
"""
from __future__ import annotations

import torch

from .stu_layer import get_hankel, get_spectral_filters


def _hankel_sigma_scale(seq_len: int, num_filters: int, use_hankel_L: bool) -> torch.Tensor:
    """Return the sigma^(1/4) column-energy scaling used by Hankel filters,
    so that other bases can be matched to the exact same column norms."""
    Z = get_hankel(seq_len, use_hankel_L)
    sigma = torch.linalg.eigvalsh(Z)[-num_filters:]
    return torch.clamp(sigma, min=1e-8).pow(0.25)  # [num_filters]


def _dct_basis(seq_len: int, num_filters: int) -> torch.Tensor:
    """DCT-II orthonormal basis, top num_filters lowest-frequency columns.

    Column k has phi_k[n] = sqrt(c_k / N) * cos((pi/N) * (n + 0.5) * k),
    where c_0 = 1 and c_k = 2 for k > 0. Each column has unit L2 norm.
    """
    n = torch.arange(seq_len, dtype=torch.float64).unsqueeze(1)        # [N, 1]
    k = torch.arange(num_filters, dtype=torch.float64).unsqueeze(0)    # [1, K]
    basis = torch.cos((torch.pi / seq_len) * (n + 0.5) * k)            # [N, K]
    # Normalize each column
    norms_sq = (basis ** 2).sum(dim=0, keepdim=True)
    basis = basis / torch.sqrt(norms_sq)
    return basis  # [N, K], orthonormal columns


def _dft_real_basis(seq_len: int, num_filters: int) -> torch.Tensor:
    """K lowest-frequency real Fourier modes as orthonormal columns.

    Column 0 is the DC component (constant). Subsequent columns alternate
    cosine and sine of frequency 1, 2, ... up to K columns total. Each column
    has unit L2 norm.
    """
    n = torch.arange(seq_len, dtype=torch.float64)                     # [N]
    cols = []
    # DC
    cols.append(torch.ones(seq_len, dtype=torch.float64))
    freq = 1
    while len(cols) < num_filters:
        omega = 2.0 * torch.pi * freq / seq_len
        cols.append(torch.cos(omega * n))
        if len(cols) < num_filters:
            cols.append(torch.sin(omega * n))
        freq += 1
    basis = torch.stack(cols[:num_filters], dim=1)                     # [N, K]
    norms_sq = (basis ** 2).sum(dim=0, keepdim=True)
    norms_sq = torch.clamp(norms_sq, min=1e-12)
    basis = basis / torch.sqrt(norms_sq)
    return basis


def make_filters(
    kind: str,
    seq_len: int,
    num_filters: int,
    use_hankel_L: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Return a filter tensor of shape [seq_len, num_filters], dtype float32."""
    if kind == "hankel":
        return get_spectral_filters(
            seq_len, num_filters, use_hankel_L=use_hankel_L,
            device=torch.device("cpu"), dtype=torch.float32,
        )

    if kind == "random":
        gen = torch.Generator().manual_seed(seed)
        g = torch.randn(seq_len, num_filters, generator=gen, dtype=torch.float64)
        return g.to(torch.float32)

    if kind == "random_normalized":
        gen = torch.Generator().manual_seed(seed)
        g = torch.randn(seq_len, num_filters, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(g)
        q = q * torch.sign(torch.diag(r))  # de-bias to Haar
        scale = _hankel_sigma_scale(seq_len, num_filters, use_hankel_L)
        return (q * scale).to(torch.float32)

    if kind == "dct":
        basis = _dct_basis(seq_len, num_filters)
        scale = _hankel_sigma_scale(seq_len, num_filters, use_hankel_L)
        return (basis * scale).to(torch.float32)

    if kind == "dft":
        basis = _dft_real_basis(seq_len, num_filters)
        scale = _hankel_sigma_scale(seq_len, num_filters, use_hankel_L)
        return (basis * scale).to(torch.float32)

    if kind == "hankel_scaled":
        # Hankel directions, but rescale column norms from sigma^(1/4) up to
        # sqrt(seq_len) — i.e. roughly match the L2 norm of an unnormalized
        # i.i.d. Gaussian column. The directions (and orthogonality) are
        # exactly Hankel's; only the scale differs from kind='hankel'.
        phi = get_spectral_filters(
            seq_len, num_filters, use_hankel_L=use_hankel_L,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        target_norm = float(torch.tensor(seq_len, dtype=torch.float32).sqrt())
        cur_norms = torch.linalg.norm(phi, dim=0).clamp(min=1e-8)
        return phi * (target_norm / cur_norms)

    raise ValueError(
        f"Unknown filter kind: {kind!r}. "
        f"Expected 'hankel', 'random', 'random_normalized', 'dct', 'dft', "
        f"or 'hankel_scaled'."
    )

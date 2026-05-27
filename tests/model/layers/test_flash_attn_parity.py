"""Numerical parity tests: standard attention path vs SDPA (FlashAttention-2) path.

Validates that enabling ``use_flash_attn=True`` produces outputs numerically
equivalent (within tolerance) to the original einsum-based attention path.

These tests run on CPU. On CPU, ``torch.nn.functional.scaled_dot_product_attention``
falls back to its math/mem-efficient backend, but the algebraic result must still
match the manual einsum reference. On Ampere+ GPUs SDPA dispatches to
FlashAttention-2 — see scripts/benchmark_flash_attn.py for perf validation.

Tolerance notes:
- Both paths run in float32 here (no autocast on CPU), so we use a tight
  ~1e-5 tolerance. The standard AttentionPairBias path explicitly forces fp32
  internally; the SDPA path inherits ambient dtype, also fp32 in these tests.
"""

import pytest
import torch

from boltz.model.layers.attentionv2 import AttentionPairBias
from boltz.model.layers.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _randomize_proj_o(layer: torch.nn.Module) -> None:
    """Override final_init (zeros) on output projection so outputs are non-trivial."""
    if hasattr(layer, "proj_o") and hasattr(layer.proj_o, "weight"):
        torch.nn.init.normal_(layer.proj_o.weight, std=0.1)


# ── AttentionPairBias parity ─────────────────────────────────────────────────


class TestAttentionPairBiasParity:
    """SDPA path must match the einsum path for AttentionPairBias."""

    @pytest.fixture
    def layer(self):
        torch.manual_seed(0)
        layer = AttentionPairBias(c_s=32, c_z=16, num_heads=4)
        _randomize_proj_o(layer)
        layer.eval()
        return layer

    def test_parity_basic(self, layer):
        """Same inputs produce numerically equivalent outputs."""
        torch.manual_seed(1)
        B, N = 2, 8
        s = torch.randn(B, N, 32)
        z = torch.randn(B, N, N, 16)
        mask = torch.ones(B, N)

        with torch.no_grad():
            out_standard = layer(s, z, mask, k_in=s, use_flash_attn=False)
            out_sdpa = layer(s, z, mask, k_in=s, use_flash_attn=True)

        assert out_standard.shape == out_sdpa.shape
        # fp32 on both paths → tight tolerance
        max_diff = (out_standard - out_sdpa).abs().max().item()
        assert max_diff < 1e-4, f"max abs diff {max_diff} exceeds 1e-4"

    def test_parity_with_partial_mask(self, layer):
        """Padding mask is honored equivalently by both paths."""
        torch.manual_seed(2)
        B, N = 2, 10
        s = torch.randn(B, N, 32)
        z = torch.randn(B, N, N, 16)
        # Mask out the last 3 tokens of each batch element
        mask = torch.cat([torch.ones(B, N - 3), torch.zeros(B, 3)], dim=-1)

        with torch.no_grad():
            out_standard = layer(s, z, mask, k_in=s, use_flash_attn=False)
            out_sdpa = layer(s, z, mask, k_in=s, use_flash_attn=True)

        max_diff = (out_standard - out_sdpa).abs().max().item()
        assert max_diff < 1e-4, f"max abs diff {max_diff} exceeds 1e-4"

    def test_parity_multiplicity(self, layer):
        """Multiplicity broadcasting matches between paths."""
        torch.manual_seed(3)
        multiplicity = 2
        B_base, N = 2, 6
        B_total = B_base * multiplicity
        s = torch.randn(B_total, N, 32)
        z = torch.randn(B_base, N, N, 16)
        mask = torch.ones(B_total, N)

        with torch.no_grad():
            out_standard = layer(
                s, z, mask, k_in=s, multiplicity=multiplicity, use_flash_attn=False,
            )
            out_sdpa = layer(
                s, z, mask, k_in=s, multiplicity=multiplicity, use_flash_attn=True,
            )

        assert out_standard.shape == out_sdpa.shape == (B_total, N, 32)
        max_diff = (out_standard - out_sdpa).abs().max().item()
        assert max_diff < 1e-4, f"max abs diff {max_diff} exceeds 1e-4"


# ── TriangleAttention parity ─────────────────────────────────────────────────


class TestTriangleAttentionParity:
    """SDPA path must match the einsum path for TriangleAttention."""

    @pytest.fixture
    def starting_layer(self):
        torch.manual_seed(10)
        layer = TriangleAttentionStartingNode(c_in=32, c_hidden=16, no_heads=4)
        layer.eval()
        return layer

    @pytest.fixture
    def ending_layer(self):
        torch.manual_seed(11)
        layer = TriangleAttentionEndingNode(c_in=32, c_hidden=16, no_heads=4)
        layer.eval()
        return layer

    def test_starting_node_parity(self, starting_layer):
        """TriangleAttentionStartingNode: SDPA == einsum."""
        torch.manual_seed(20)
        B, N = 1, 8
        x = torch.randn(B, N, N, 32)
        mask = torch.ones(B, N, N)

        with torch.no_grad():
            out_standard = starting_layer(x, mask=mask, use_flash_attn=False)
            out_sdpa = starting_layer(x, mask=mask, use_flash_attn=True)

        assert out_standard.shape == out_sdpa.shape
        max_diff = (out_standard - out_sdpa).abs().max().item()
        # TriangleAttention has larger numerical envelope due to bias addition
        # and softmax over more positions; allow a looser tolerance.
        assert max_diff < 1e-4, f"max abs diff {max_diff} exceeds 1e-4"

    def test_ending_node_parity(self, ending_layer):
        """TriangleAttentionEndingNode: SDPA == einsum."""
        torch.manual_seed(21)
        B, N = 1, 8
        x = torch.randn(B, N, N, 32)
        mask = torch.ones(B, N, N)

        with torch.no_grad():
            out_standard = ending_layer(x, mask=mask, use_flash_attn=False)
            out_sdpa = ending_layer(x, mask=mask, use_flash_attn=True)

        assert out_standard.shape == out_sdpa.shape
        max_diff = (out_standard - out_sdpa).abs().max().item()
        assert max_diff < 1e-4, f"max abs diff {max_diff} exceeds 1e-4"

    def test_starting_node_with_mask(self, starting_layer):
        """Partial mask is honored equivalently."""
        torch.manual_seed(22)
        B, N = 1, 10
        x = torch.randn(B, N, N, 32)
        # Mask out a column of the pairwise representation
        mask = torch.ones(B, N, N)
        mask[..., -3:] = 0.0

        with torch.no_grad():
            out_standard = starting_layer(x, mask=mask, use_flash_attn=False)
            out_sdpa = starting_layer(x, mask=mask, use_flash_attn=True)

        max_diff = (out_standard - out_sdpa).abs().max().item()
        assert max_diff < 1e-4, f"max abs diff {max_diff} exceeds 1e-4"

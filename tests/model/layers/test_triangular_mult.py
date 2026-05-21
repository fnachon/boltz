"""Tests for boltz.model.layers.triangular_mult — TriangleMultiplicationOutgoing/Incoming."""

import pytest
import torch

from boltz.model.layers import triangular_mult
from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


@pytest.fixture
def dim():
    return 32


@pytest.fixture
def outgoing(dim):
    layer = TriangleMultiplicationOutgoing(dim=dim)
    layer.eval()
    return layer


@pytest.fixture
def incoming(dim):
    layer = TriangleMultiplicationIncoming(dim=dim)
    layer.eval()
    return layer


class TestTriangleMultiplication:
    """Tests for triangle multiplication layers."""

    def test_outgoing_shape(self, outgoing, dim):
        """Outgoing output shape is (B, N, N, D)."""
        B, N = 2, 6
        x = torch.randn(B, N, N, dim)
        mask = torch.ones(B, N, N)

        with torch.no_grad():
            out = outgoing(x, mask)
        assert out.shape == (B, N, N, dim)

    def test_incoming_shape(self, incoming, dim):
        """Incoming output shape is (B, N, N, D)."""
        B, N = 2, 6
        x = torch.randn(B, N, N, dim)
        mask = torch.ones(B, N, N)

        with torch.no_grad():
            out = incoming(x, mask)
        assert out.shape == (B, N, N, dim)

    def test_mask_effect(self, outgoing, dim):
        """Zero mask should reduce the output magnitude."""
        B, N = 1, 8
        torch.manual_seed(42)
        x = torch.randn(B, N, N, dim)
        # Need non-zero p_in weights for mask to matter
        with torch.no_grad():
            for p in outgoing.parameters():
                torch.nn.init.normal_(p, std=0.1)

        with torch.no_grad():
            out_full = outgoing(x, torch.ones(B, N, N))
            out_zero = outgoing(x, torch.zeros(B, N, N))

        # With zero mask, intermediate values are zeroed → output is different
        assert not torch.allclose(out_full, out_zero, atol=1e-3)

    def test_outgoing_ne_incoming(self, dim):
        """Outgoing and incoming produce different results."""
        torch.manual_seed(7)
        out_layer = TriangleMultiplicationOutgoing(dim=dim)
        in_layer = TriangleMultiplicationIncoming(dim=dim)

        # Share same weights
        with torch.no_grad():
            for po, pi in zip(out_layer.parameters(), in_layer.parameters()):
                torch.nn.init.normal_(po, std=0.5)
                pi.copy_(po)

        out_layer.eval()
        in_layer.eval()

        B, N = 1, 6
        x = torch.randn(B, N, N, dim)
        mask = torch.ones(B, N, N)

        with torch.no_grad():
            res_out = out_layer(x, mask)
            res_in = in_layer(x, mask)

        # Different einsum patterns → different results
        assert not torch.allclose(res_out, res_in, atol=1e-3)

    def test_determinism(self, outgoing, dim):
        """Eval mode produces deterministic results."""
        B, N = 1, 6
        x = torch.randn(B, N, N, dim)
        mask = torch.ones(B, N, N)

        with torch.no_grad():
            out1 = outgoing(x, mask)
            out2 = outgoing(x, mask)
        assert torch.allclose(out1, out2, atol=1e-8)

    def test_kernel_failure_falls_back_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outgoing: TriangleMultiplicationOutgoing,
        dim: int,
    ) -> None:
        """Recoverable kernel errors fall back and are not retried."""
        calls = []

        def fail_kernel(**kwargs: object) -> None:
            calls.append(kwargs["direction"])
            message = (
                "Failed to import Triton-based component: "
                "triangle_multiplicative_update:\nNot Supported"
            )
            raise Exception(message)  # noqa: TRY002

        monkeypatch.setitem(
            triangular_mult._kernel_failure,  # noqa: SLF001
            "reason",
            None,
        )
        monkeypatch.setattr(triangular_mult, "kernel_triangular_mult", fail_kernel)

        batch, num_tokens = 1, 6
        x = torch.randn(batch, num_tokens, num_tokens, dim)
        mask = torch.ones(batch, num_tokens, num_tokens)

        with pytest.warns(RuntimeWarning, match="falling back"):
            out1 = outgoing(x, mask, use_kernels=True)
        out2 = outgoing(x, mask, use_kernels=True)

        assert out1.shape == (batch, num_tokens, num_tokens, dim)
        assert out2.shape == (batch, num_tokens, num_tokens, dim)
        assert calls == ["outgoing"]

        incoming = TriangleMultiplicationIncoming(dim=dim)
        incoming.eval()
        out3 = incoming(x, mask, use_kernels=True)

        assert out3.shape == (batch, num_tokens, num_tokens, dim)
        assert calls == ["outgoing"]

    def test_unexpected_kernel_failure_is_not_suppressed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outgoing: TriangleMultiplicationOutgoing,
        dim: int,
    ) -> None:
        """Unexpected kernel errors still surface."""

        def fail_kernel(**kwargs: object) -> None:
            del kwargs
            message = "some unrelated CUDA operation is Not Supported"
            raise RuntimeError(message)

        monkeypatch.setitem(
            triangular_mult._kernel_failure,  # noqa: SLF001
            "reason",
            None,
        )
        monkeypatch.setattr(triangular_mult, "kernel_triangular_mult", fail_kernel)

        batch, num_tokens = 1, 6
        x = torch.randn(batch, num_tokens, num_tokens, dim)
        mask = torch.ones(batch, num_tokens, num_tokens)

        with pytest.raises(RuntimeError, match="unrelated CUDA operation"):
            outgoing(x, mask, use_kernels=True)

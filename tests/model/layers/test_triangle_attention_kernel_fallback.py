"""Tests for the cuequivariance kernel fallback in triangle attention."""

import pytest
import torch

from boltz.model.layers.triangular_attention import primitives
from boltz.model.layers.triangular_attention.attention import TriangleAttention


@pytest.fixture
def dim() -> int:
    return 32


@pytest.fixture
def attention(dim: int) -> TriangleAttention:
    layer = TriangleAttention(c_in=dim, c_hidden=8, no_heads=2)
    layer.eval()
    return layer


class TestTriangleAttentionKernelFallback:
    """Tests for kernel_triangular_attn fallback in Attention."""

    def test_kernel_failure_falls_back_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        attention: TriangleAttention,
        dim: int,
    ) -> None:
        """Recoverable kernel errors fall back and are not retried."""
        calls: list[dict] = []

        def fail_kernel(**kwargs: object) -> None:
            calls.append(kwargs)
            message = (
                "Failed to import Triton-based component: "
                "triangle_attention:\nNot Supported"
            )
            raise Exception(message)  # noqa: TRY002

        monkeypatch.setitem(
            primitives._kernel_failure,  # noqa: SLF001
            "reason",
            None,
        )
        monkeypatch.setattr(primitives, "kernel_triangular_attn", fail_kernel)

        batch, num_tokens = 1, 6
        x = torch.randn(batch, num_tokens, num_tokens, dim)
        mask = torch.ones(batch, num_tokens, num_tokens)

        with torch.no_grad():
            with pytest.warns(RuntimeWarning, match="falling back"):
                out1 = attention(x, mask, use_kernels=True)
            out2 = attention(x, mask, use_kernels=True)

        assert out1.shape == (batch, num_tokens, num_tokens, dim)
        assert out2.shape == (batch, num_tokens, num_tokens, dim)
        assert len(calls) == 1

    def test_import_error_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        attention: TriangleAttention,
        dim: int,
    ) -> None:
        """A bare ImportError (no cuequivariance_torch installed) also falls back."""

        def fail_kernel(**kwargs: object) -> None:
            del kwargs
            raise ModuleNotFoundError("No module named 'cuequivariance_torch'")

        monkeypatch.setitem(
            primitives._kernel_failure,  # noqa: SLF001
            "reason",
            None,
        )
        monkeypatch.setattr(primitives, "kernel_triangular_attn", fail_kernel)

        batch, num_tokens = 1, 6
        x = torch.randn(batch, num_tokens, num_tokens, dim)
        mask = torch.ones(batch, num_tokens, num_tokens)

        with torch.no_grad():
            with pytest.warns(RuntimeWarning, match="falling back"):
                out = attention(x, mask, use_kernels=True)

        assert out.shape == (batch, num_tokens, num_tokens, dim)

    def test_unexpected_kernel_failure_is_not_suppressed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        attention: TriangleAttention,
        dim: int,
    ) -> None:
        """Unexpected kernel errors still surface."""

        def fail_kernel(**kwargs: object) -> None:
            del kwargs
            raise RuntimeError("some unrelated CUDA operation is Not Supported")

        monkeypatch.setitem(
            primitives._kernel_failure,  # noqa: SLF001
            "reason",
            None,
        )
        monkeypatch.setattr(primitives, "kernel_triangular_attn", fail_kernel)

        batch, num_tokens = 1, 6
        x = torch.randn(batch, num_tokens, num_tokens, dim)
        mask = torch.ones(batch, num_tokens, num_tokens)

        with torch.no_grad():
            with pytest.raises(RuntimeError, match="unrelated CUDA operation"):
                attention(x, mask, use_kernels=True)

    def test_fallback_matches_pytorch_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        attention: TriangleAttention,
        dim: int,
    ) -> None:
        """When the kernel fails, the fallback output matches use_kernels=False."""

        def fail_kernel(**kwargs: object) -> None:
            del kwargs
            raise ModuleNotFoundError("No module named 'cuequivariance_torch'")

        batch, num_tokens = 1, 6
        torch.manual_seed(0)
        x = torch.randn(batch, num_tokens, num_tokens, dim)
        mask = torch.ones(batch, num_tokens, num_tokens)

        with torch.no_grad():
            reference = attention(x, mask, use_kernels=False)

        monkeypatch.setitem(
            primitives._kernel_failure,  # noqa: SLF001
            "reason",
            None,
        )
        monkeypatch.setattr(primitives, "kernel_triangular_attn", fail_kernel)

        with torch.no_grad():
            with pytest.warns(RuntimeWarning, match="falling back"):
                fallback = attention(x, mask, use_kernels=True)

        assert torch.allclose(reference, fallback, atol=1e-6)

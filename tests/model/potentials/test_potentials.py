"""Tests for boltz.model.potentials.potentials — DistancePotential, FlatBottomPotential, DihedralPotential, get_potentials."""

import math

import torch
import pytest

from boltz.model.potentials.potentials import (
    DistancePotential,
    FlatBottomPotential,
    DihedralPotential,
    ConnectionsPotential,
    ContactPotentital,
    get_potentials,
)


# ConnectionsPotential is a concrete class (FlatBottomPotential + DistancePotential)
# that we can use to test both compute_variable and compute_function.


class TestDistancePotential:
    """Tests for DistancePotential.compute_variable (via ConnectionsPotential)."""

    def test_known_distance(self):
        """Distance between two known points."""
        pot = ConnectionsPotential()
        # Two atoms at (0,0,0) and (3,4,0) → distance = 5
        coords = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]).unsqueeze(0)
        index = torch.tensor([[0], [1]])

        dist = pot.compute_variable(coords, index, compute_gradient=False)
        assert dist.item() == pytest.approx(5.0, abs=1e-4)

    def test_gradient(self):
        """Gradient matches finite difference approximation."""
        pot = ConnectionsPotential()
        torch.manual_seed(42)
        coords = torch.randn(1, 4, 3, dtype=torch.float64)
        index = torch.tensor([[0, 1], [2, 3]])

        dist, grad = pot.compute_variable(coords, index, compute_gradient=True)

        # Finite difference check
        eps = 1e-5
        for pair_idx in range(2):
            for atom_pos in range(2):
                atom_idx = index[atom_pos, pair_idx]
                for dim in range(3):
                    coords_plus = coords.clone()
                    coords_minus = coords.clone()
                    coords_plus[0, atom_idx, dim] += eps
                    coords_minus[0, atom_idx, dim] -= eps

                    dist_plus = pot.compute_variable(coords_plus, index, compute_gradient=False)
                    dist_minus = pot.compute_variable(coords_minus, index, compute_gradient=False)

                    fd_grad = (dist_plus[0, pair_idx] - dist_minus[0, pair_idx]) / (2 * eps)
                    analytic_grad = grad[0, atom_pos, pair_idx, dim]
                    assert fd_grad.item() == pytest.approx(analytic_grad.item(), abs=1e-3)


class TestFlatBottomPotential:
    """Tests for FlatBottomPotential.compute_function (via ConnectionsPotential)."""

    def test_in_bounds(self):
        """Value within bounds → zero energy."""
        pot = ConnectionsPotential()
        value = torch.tensor([5.0])
        k = torch.tensor([1.0])
        lower = torch.tensor([3.0])
        upper = torch.tensor([7.0])

        energy = pot.compute_function(value, k, lower, upper)
        assert energy.item() == pytest.approx(0.0, abs=1e-6)

    def test_out_of_bounds_lower(self):
        """Value below lower bound → positive energy."""
        pot = ConnectionsPotential()
        value = torch.tensor([1.0])
        k = torch.tensor([2.0])
        lower = torch.tensor([3.0])
        upper = torch.tensor([7.0])

        energy = pot.compute_function(value, k, lower, upper)
        # energy = k * (lower - value) = 2 * (3 - 1) = 4
        assert energy.item() == pytest.approx(4.0, abs=1e-6)

    def test_out_of_bounds_upper(self):
        """Value above upper bound → positive energy."""
        pot = ConnectionsPotential()
        value = torch.tensor([10.0])
        k = torch.tensor([2.0])
        lower = torch.tensor([3.0])
        upper = torch.tensor([7.0])

        energy = pot.compute_function(value, k, lower, upper)
        # energy = k * (value - upper) = 2 * (10 - 7) = 6
        assert energy.item() == pytest.approx(6.0, abs=1e-6)


class TestDihedralPotential:
    """Tests for DihedralPotential.compute_variable (via ChiralAtomPotential)."""

    def test_known_angle(self):
        """Known planar configuration → dihedral ≈ π or 0."""
        from boltz.model.potentials.potentials import ChiralAtomPotential

        pot = ChiralAtomPotential()
        # Four atoms in a plane: cis configuration → dihedral ≈ 0
        coords = torch.tensor([
            [0.0, 0.0, 0.0],   # i
            [1.0, 0.0, 0.0],   # j
            [1.5, 1.0, 0.0],   # k
            [2.5, 1.0, 0.0],   # l  (cis: same side)
        ]).unsqueeze(0)
        index = torch.tensor([[0], [1], [2], [3]])

        phi = pot.compute_variable(coords, index, compute_gradient=False)
        assert torch.isfinite(phi).all()
        assert abs(phi.item()) <= math.pi + 0.1

    def test_trans_configuration(self):
        """Trans configuration → dihedral ≈ ±π."""
        from boltz.model.potentials.potentials import ChiralAtomPotential

        pot = ChiralAtomPotential()
        # 3D trans configuration: i above plane, l below (opposite sides of j-k bond)
        # j-k along x-axis; i above in z; l below in z with slight y offset
        coords = torch.tensor([
            [0.0, 0.0, 1.0],    # i (above xz-plane)
            [1.0, 0.0, 0.0],    # j
            [2.0, 0.0, 0.0],    # k
            [3.0, 0.1, -1.0],   # l (below, slight y-offset to break degeneracy)
        ]).unsqueeze(0)
        index = torch.tensor([[0], [1], [2], [3]])

        phi = pot.compute_variable(coords, index, compute_gradient=False)
        # Trans configuration: dihedral should be close to ±π
        assert abs(abs(phi.item()) - math.pi) < 0.5


class TestGetPotentials:
    """Tests for get_potentials."""

    def test_fk_steering_returns_potentials(self):
        """fk_steering=True returns expected potential types."""
        potentials = get_potentials(
            {"fk_steering": True, "physical_guidance_update": False, "contact_guidance_update": False},
            boltz2=False,
        )
        assert len(potentials) > 0

        type_names = [type(p).__name__ for p in potentials]
        assert "SymmetricChainCOMPotential" in type_names
        assert "VDWOverlapPotential" in type_names
        assert "ConnectionsPotential" in type_names

    def test_no_steering_empty(self):
        """No steering → empty potentials list."""
        potentials = get_potentials(
            {"fk_steering": False, "physical_guidance_update": False, "contact_guidance_update": False},
            boltz2=False,
        )
        assert len(potentials) == 0

    def test_boltz2_contact_potentials(self):
        """boltz2=True with contact_guidance adds ContactPotentital."""
        potentials = get_potentials(
            {"fk_steering": True, "physical_guidance_update": False, "contact_guidance_update": True},
            boltz2=True,
        )
        type_names = [type(p).__name__ for p in potentials]
        assert "ContactPotentital" in type_names
        assert "TemplateReferencePotential" in type_names


class TestUnionContactPotential:
    """Union contact weighting should remain active for large violations."""

    def test_union_energy_does_not_underflow_to_zero(self) -> None:
        """Large positive contact violations keep finite union energy."""
        pot = ContactPotentital()
        coords = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [100.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [110.0, 0.0, 0.0],
                ]
            ]
        )
        feats = {
            "contact_pair_index": torch.tensor([[[0, 2], [1, 3]]]),
            "contact_union_index": torch.tensor([[0, 0]]),
            "contact_negation_mask": torch.tensor([[True, True]]),
            "contact_thresholds": torch.tensor([[1.0, 1.0]]),
        }

        energy = pot.compute(coords, feats, {"union_lambda": 8.0})

        assert torch.isfinite(energy).all()
        assert energy.item() > 90.0  # noqa: PLR2004

    def test_union_gradient_does_not_underflow_to_zero(self) -> None:
        """Large positive contact violations keep nonzero union gradients."""
        pot = ContactPotentital()
        coords = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [100.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [110.0, 0.0, 0.0],
                ]
            ]
        )
        feats = {
            "contact_pair_index": torch.tensor([[[0, 2], [1, 3]]]),
            "contact_union_index": torch.tensor([[0, 0]]),
            "contact_negation_mask": torch.tensor([[True, True]]),
            "contact_thresholds": torch.tensor([[1.0, 1.0]]),
        }

        grad = pot.compute_gradient(coords, feats, {"union_lambda": 8.0})

        assert torch.isfinite(grad).all()
        assert grad.abs().sum().item() > 0.0  # noqa: PLR2004

    def test_union_gradient_matches_finite_difference(self) -> None:
        """Union contact gradient matches a directional finite difference."""
        pot = ContactPotentital()
        coords = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [5.0, 1.0, 0.0],
                ]
            ],
            dtype=torch.float64,
        )
        direction = torch.tensor(
            [
                [
                    [0.1, -0.2, 0.0],
                    [0.3, 0.1, 0.0],
                    [-0.2, 0.0, 0.0],
                    [0.4, -0.1, 0.0],
                ]
            ],
            dtype=torch.float64,
        )
        feats = {
            "contact_pair_index": torch.tensor([[[0, 2], [1, 3]]]),
            "contact_union_index": torch.tensor([[0, 0]]),
            "contact_negation_mask": torch.tensor([[True, True]]),
            "contact_thresholds": torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        }
        params = {"union_lambda": 0.3}
        eps = 1e-6

        grad = pot.compute_gradient(coords, feats, params)
        analytic = (grad * direction).sum()
        finite_diff = (
            pot.compute(coords + eps * direction, feats, params).sum()
            - pot.compute(coords - eps * direction, feats, params).sum()
        ) / (2 * eps)

        assert analytic.item() == pytest.approx(finite_diff.item(), abs=1e-5)

    def test_union_rejects_negative_indices(self) -> None:
        """Union indices must not use negative sentinel values."""
        pot = ContactPotentital()
        coords = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [5.0, 1.0, 0.0],
                ]
            ]
        )
        feats = {
            "contact_pair_index": torch.tensor([[[0, 2], [1, 3]]]),
            "contact_union_index": torch.tensor([[0, -1]]),
            "contact_negation_mask": torch.tensor([[True, True]]),
            "contact_thresholds": torch.tensor([[1.0, 1.0]]),
        }

        with pytest.raises(ValueError, match="non-negative"):
            pot.compute(coords, feats, {"union_lambda": 0.3})


class TestBfloat16Potential:
    """FlatBottomPotential must work with bfloat16 inputs on CPU.

    When AMP autocast produces bfloat16 tensors, arithmetic may upcast
    to float32. The indexed assignment back into a bfloat16 tensor
    requires explicit dtype casts.
    """

    def test_energy(self):
        """compute_function returns bfloat16 energy for bfloat16 inputs."""
        pot = ConnectionsPotential()
        value = torch.tensor([1.0], dtype=torch.bfloat16)
        k = torch.tensor([2.0], dtype=torch.bfloat16)
        lower = torch.tensor([3.0], dtype=torch.bfloat16)
        upper = torch.tensor([7.0], dtype=torch.bfloat16)

        energy = pot.compute_function(value, k, lower, upper)
        assert energy.dtype == torch.bfloat16
        assert torch.isfinite(energy).all()
        # energy = k * (lower - value) = 2 * (3 - 1) = 4
        assert energy.item() == pytest.approx(4.0, abs=0.1)

    def test_derivative(self):
        """compute_function derivative preserves bfloat16 dtype."""
        pot = ConnectionsPotential()
        value = torch.tensor([1.0], dtype=torch.bfloat16)
        k = torch.tensor([2.0], dtype=torch.bfloat16)
        lower = torch.tensor([3.0], dtype=torch.bfloat16)
        upper = torch.tensor([7.0], dtype=torch.bfloat16)

        energy, dEnergy = pot.compute_function(
            value, k, lower, upper, compute_derivative=True
        )
        assert dEnergy.dtype == torch.bfloat16
        assert torch.isfinite(dEnergy).all()

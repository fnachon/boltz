from torch import nn

from boltz.model.models import boltz2


class _DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def _minimal_boltz2(validate_structure):
    return boltz2.Boltz2(
        atom_s=2,
        atom_z=2,
        token_s=2,
        token_z=2,
        num_bins=4,
        training_args={},
        validation_args={},
        embedder_args={},
        msa_args={},
        pairformer_args={},
        score_model_args={
            "atom_encoder_depth": 1,
            "atom_encoder_heads": 1,
            "token_transformer_depth": 1,
            "token_transformer_heads": 1,
            "atom_decoder_depth": 1,
            "atom_decoder_heads": 1,
            "conditioning_transition_layers": 1,
        },
        diffusion_process_args={},
        diffusion_loss_args={},
        confidence_prediction=False,
        validate_structure=validate_structure,
        validators=[],
    )


def test_boltz2_stores_validate_structure(monkeypatch):
    for module_name in [
        "InputEmbedder",
        "RelativePositionEncoder",
        "ContactConditioning",
        "MSAModule",
        "PairformerModule",
        "DiffusionConditioning",
        "AtomDiffusion",
        "DistogramModule",
    ]:
        monkeypatch.setattr(boltz2, module_name, _DummyModule)

    for validate_structure in [False, True]:
        model = _minimal_boltz2(validate_structure)

        assert model.validate_structure is validate_structure

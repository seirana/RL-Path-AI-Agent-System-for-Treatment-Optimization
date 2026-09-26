import numpy as np

from src.preprocess import EffectMatrix, load_effects, save_effects


def test_effect_matrix_round_trip_without_pickle(tmp_path):
    matrix = EffectMatrix(
        drug_names=["drug_a", "drug_b"],
        pathway_ids=["R-HSA-1", "R-HSA-2"],
        pathway_names=["path_a", "path_b"],
        effects=np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        ),
    )
    path = tmp_path / "effects.npz"

    save_effects(matrix, path)
    loaded = load_effects(path)

    assert loaded.drug_names == matrix.drug_names
    assert loaded.pathway_ids == matrix.pathway_ids
    assert loaded.pathway_names == matrix.pathway_names
    np.testing.assert_allclose(loaded.effects, matrix.effects)

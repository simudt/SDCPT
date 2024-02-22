import torch

from sdcpt.distance.validators.dm_validator import DistanceMatrixValidator


def test_distance_matrix_validation():
    mat = torch.tensor(
        [
            [0.0, 1.1, 1.2, 1.3],
            [1.1, 0.0, 1.0, 1.4],
            [1.2, 1.0, 0.0, 1.5],
            [1.3, 1.4, 1.5, 0.0],
        ]
    )

    validator = DistanceMatrixValidator(mat)

    is_valid = validator.is_valid()
    print(f"Is the provided matrix a valid distance matrix? {is_valid}")

    assert is_valid, "Matrix is not a valid distance matrix"


test_distance_matrix_validation()

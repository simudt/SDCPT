import torch

from sdcpt.distance.hausdorff_dist import DirectedHausdorffDist


def test_DirectsHausdorffDistance():
    A = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    B = torch.tensor([[2.0, 2.5], [2.5, 2.5]])

    hausdorff_distance = DirectedHausdorffDist(A, B)
    result = hausdorff_distance.directed_hausdorff()
    print("Directed Hausdorff Distance: ", result)

    assert (
        0 <= result
    ), f"Expected Hausdorff distance to be non-negative, but got {result}"


test_DirectsHausdorffDistance()

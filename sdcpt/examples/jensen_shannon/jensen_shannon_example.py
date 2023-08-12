import torch

from sdcpt.distance.jensen_shannon import JensenShannonDiv


def test_JSD():
    input = torch.tensor([0.1, 0.3, 0.4, 0.2])
    target = torch.tensor([0.2, 0.2, 0.3, 0.3])

    jsd = JensenShannonDiv.JSDiv(input, target)
    print(jsd)

    assert (
        0 <= jsd.item() <= 1
    ), f"Expected Jensen-Shannon Divergence between 0 and 1, but got {jsd}"


test_JSD()

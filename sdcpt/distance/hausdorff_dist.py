import torch


class DirectedHausdorffDist:
    def __init__(self, A, B):
        self.validate_inputs(A, B)
        self.A = A
        self.B = B

    def validate_inputs(self, A, B):
        if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
            raise ValueError("Both A and B should be instances of torch.Tensor")

        if A.dim() != 2 or A.size(1) != 2 or B.dim() != 2 or B.size(1) != 2:
            raise ValueError("Both A and B should be Nx2 tensors (N number of points)")

        if A.device != B.device:
            raise ValueError("Both A and B should be on the same device")

        if A.dtype != B.dtype:
            raise ValueError("Both A and B should have the same data type")

    def directed_hausdorff(self):
        n = self.A.shape[0]
        m = self.B.shape[0]

        A_exp = self.A.unsqueeze(1).expand(n, m, 2)
        B_exp = self.B.unsqueeze(0).expand(n, m, 2)

        dists = torch.norm(A_exp - B_exp, dim=2)

        min_dists, _ = dists.min(dim=1)

        return min_dists.max().item()

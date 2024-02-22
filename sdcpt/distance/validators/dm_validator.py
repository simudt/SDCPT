import torch


class DistanceMatrixValidator:
    def __init__(self, mat: torch.Tensor):
        self.mat = mat

    def is_square(self) -> bool:
        return self.mat.shape[0] == self.mat.shape[1]

    def has_zero_diagonal(self) -> bool:
        return torch.all(self.mat.diag() == 0)

    def is_symmetric(self) -> bool:
        return torch.all(self.mat == self.mat.T)

    def has_non_negative_entries(self) -> bool:
        return torch.all(self.mat >= 0)

    def satisfies_triangle_inequality(self) -> bool:
        n = self.mat.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if self.mat[i, j] + self.mat[j, k] < self.mat[i, k]:
                        return False
        return True

    def is_valid(self) -> bool:
        return (
            self.is_square()
            and self.has_zero_diagonal()
            and self.is_symmetric()
            and self.has_non_negative_entries()
            and self.satisfies_triangle_inequality()
        )

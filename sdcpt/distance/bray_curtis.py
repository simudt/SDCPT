import torch


class BrayCurtisDistance:
    def __init__(self, arr1, arr2):
        if len(arr1) != len(arr2):
            raise ValueError("Input arrays must have the same len()")

        self.tensor1 = torch.tensor(arr1, dtype=torch.float32)
        self.tensor2 = torch.tensor(arr2, dtype=torch.float32)

    def compute_distance(self):
        if len(self.tensor1) == 0:
            raise ValueError("Input arrays must not be empty")

        abs_diff = torch.abs(self.tensor1 - self.tensor2)
        sum1 = torch.sum(self.tensor1)
        sum2 = torch.sum(self.tensor2)

        if sum1 + sum2 == 0:
            raise ValueError("Sum of both arrays must not be zero")

        bray_curtis = torch.sum(abs_diff) / (sum1 + sum2)
        return bray_curtis.item()

import torch
import torch.nn.functional as F
import warnings


class JensenShannonDiv:
    @staticmethod
    def _validate_distribution(tensor: torch.Tensor, tolerance: float = 1e-6) -> None:
        if not torch.isclose(tensor.sum(), torch.tensor(1.0), atol=tolerance):
            raise ValueError("Input tensor should sum to 1")

        if (tensor < 0).any() or (tensor > 1).any():
            raise ValueError("All values in the tensor should be in the range [0, 1]")

    @staticmethod
    def KLDiv(
        input,
        target,
        size_average=None,
        reduce=None,
        reduction="batchmean",
        log_target=False,
    ):
        JensenShannonDiv._validate_distribution(input)

        if not log_target:
            JensenShannonDiv._validate_distribution(target)

        if size_average is not None or reduce is not None:
            warnings.warn(
                "The arguments 'size_average' and 'reduce' are deprecated now. Use 'reduction' instead of this.",
                DeprecationWarning,
            )
            reduction = "sum" if not size_average and reduce else "none"

        log_q = target.log() if not log_target else target
        return F.kl_div(log_q, input, reduction=reduction)

    @staticmethod
    def JSDiv(
        input,
        target,
        size_average=None,
        reduce=None,
        reduction="batchmean",
        log_target=False,
    ):
        JensenShannonDiv._validate_distribution(input)
        JensenShannonDiv._validate_distribution(target)

        m = 0.5 * (input + target)

        return 0.5 * JensenShannonDiv.KLDiv(
            input, m, size_average, reduce, reduction, log_target
        ) + 0.5 * JensenShannonDiv.KLDiv(
            target, m, size_average, reduce, reduction, log_target
        )

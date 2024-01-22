from recbole.evaluator.base_metric import AbstractMetric
from recbole.utils import EvaluatorType
import numpy as np


class MSE(LossMetric):
    metric_type = EvaluatorType.RANKING
    smaller = True

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric("mse", dataobject)

    def metric_info(self, preds, trues):
        return mean_squared_error(trues, preds)

 
class ECE(LossMetric):
    metric_type = EvaluatorType.RANKING
    smaller = True

    def __init__(self, config, num_bins=100):
        super().__init__(config)
        self.num_bins = num_bins

    def calculate_metric(self, dataobject):
        return self.output_metric("ece", dataobject)

    def metric_info(self, preds, trues):
        bin_lowers = np.linspace(0., 1., self.num_bins + 1)[:-1]
        bin_uppers = np.linspace(0., 1., self.num_bins + 1)[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(preds >= bin_lower, preds < bin_upper)
            bin_true_labels = trues[in_bin]
            bin_scores = preds[in_bin]

            if len(bin_true_labels) == 0:
                continue

            bin_accuracy = np.mean(bin_true_labels)
            bin_confidence = np.mean(bin_scores)

            ece += np.abs(bin_accuracy - bin_confidence) * len(bin_true_labels)

        ece /= len(preds)
        return ece

        return {"ece": round(ece, self.decimal_place)}

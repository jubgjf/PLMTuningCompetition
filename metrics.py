import torch
import torch.nn as nn
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import _get_func_signature
from sklearn.metrics import accuracy_score
from utils import hinge_loss


class TaskMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        self.label_map = {}

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()

        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}


class SST2Metric(TaskMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super(SST2Metric, self).__init__(pred, target, seq_len, tokenizer)
        self.label_map = {
            tokenizer.encode('bad', add_special_tokens=False)[0]: 0,  # negative
            tokenizer.encode('great', add_special_tokens=False)[0]: 1,  # positive
        }


class SNLIMetric(TaskMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super(SNLIMetric, self).__init__(pred, target, seq_len, tokenizer)
        self.label_map = {
            tokenizer.encode('Yes', add_special_tokens=False)[0]: 0,
            tokenizer.encode('Maybe', add_special_tokens=False)[0]: 1,
            tokenizer.encode('No', add_special_tokens=False)[0]: 2,
        }


class DBPediaMetric(TaskMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super(DBPediaMetric, self).__init__(pred, target, seq_len, tokenizer)
        self.label_map = {
            tokenizer.encode("Company", add_special_tokens=False)[0]: 0,
            tokenizer.encode("EducationalInstitution", add_special_tokens=False)[0]: 1,
            tokenizer.encode("Artist", add_special_tokens=False)[0]: 2,
            tokenizer.encode("Athlete", add_special_tokens=False)[0]: 3,
            tokenizer.encode("OfficeHolder", add_special_tokens=False)[0]: 4,
            tokenizer.encode("MeanOfTransportation", add_special_tokens=False)[0]: 5,
            tokenizer.encode("Building", add_special_tokens=False)[0]: 6,
            tokenizer.encode("NaturalPlace", add_special_tokens=False)[0]: 7,
            tokenizer.encode("Village", add_special_tokens=False)[0]: 8,
            tokenizer.encode("Animal", add_special_tokens=False)[0]: 9,
            tokenizer.encode("Plant", add_special_tokens=False)[0]: 10,
            tokenizer.encode("Album", add_special_tokens=False)[0]: 11,
            tokenizer.encode("Film", add_special_tokens=False)[0]: 12,
            tokenizer.encode("WrittenWork", add_special_tokens=False)[0]: 13,
        }


class QNLIMetric(TaskMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super(QNLIMetric, self).__init__(pred, target, seq_len, tokenizer)
        self.label_map = {
            tokenizer.encode('entailment', add_special_tokens=False)[0]: 0,
            tokenizer.encode('not_entailment', add_special_tokens=False)[0]: 1,
        }


class QQPMetric(TaskMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super(QQPMetric, self).__init__(pred, target, seq_len, tokenizer)
        self.label_map = {
            tokenizer.encode('Yes', add_special_tokens=False)[0]: 0,
            tokenizer.encode('No', add_special_tokens=False)[0]: 1,
        }

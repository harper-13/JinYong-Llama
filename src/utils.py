import os
from typing import List

import mindspore as ms
import numpy as np
from mindspore.nn.learning_rate_schedule import (
    CosineDecayLR,
    LearningRateSchedule,
    PolynomialDecayLR,
    WarmUpLR,
)
from mindspore.ops import functional as F

SEQ_LEN = 128
VOCAB_SIZE = 2048


def read_jinyong(path: str) -> str:
    """read Jin Yong fictions"""
    files: List[str] = []
    if os.path.isdir(path):
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                files.append(os.path.join(root, filename))
    elif os.path.isfile(path):
        files.append(path)
    else:
        raise ValueError("Invalid path")

    data = ""
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            data += f.read()

    return data


class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for GPT network.
    """

    def __init__(
        self,
        learning_rate,
        end_learning_rate,
        warmup_steps,
        decay_steps,
        power=1.0,
        use_cosine=True,
    ):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(
            learning_rate, end_learning_rate, decay_steps, power
        )
        self.cosine_decay_lr = CosineDecayLR(
            end_learning_rate, learning_rate, decay_steps
        )
        self.warmup_steps = ms.Tensor(np.array([warmup_steps]).astype(np.float32))

        self.one = ms.Tensor(np.array([1.0]).astype(np.float32))
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = F.greater(self.warmup_steps, global_step).float()
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr

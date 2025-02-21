import argparse

import mindspore.communication.management as D
import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.train.callback import (
    CheckpointConfig,
    LossMonitor,
    ModelCheckpoint,
    TimeMonitor,
)
from mindspore.train.model import Model

from src.dataset import create_dataset
from src.llama import Llama, LlamaWithLoss, ModelArgs
from src.utils import VOCAB_SIZE, LearningRate

# TODO: 根据需要调整训练超参数以及各种路径等，如优化器、学习率等


def run_train():
    """train function"""
    parser = argparse.ArgumentParser(description="Llama training")
    parser.add_argument(
        "--device_id", type=int, default=0, help="Device id, default is 0."
    )
    parser.add_argument(
        "--device_num", type=int, default=1, help="Use device nums, default is 1."
    )
    parser.add_argument(
        "--distribute",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Run distribute, default is false.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "lamb"],
        help="select which optimizer to be used, default adam",
    )
    parser.add_argument(
        "--epoch_size", type=int, default=2, help="Epoch size, default is 2."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset",
        help="Data path of your MindRecord files.",
    )
    parser.add_argument(
        "--start_lr",
        type=float,
        default="5e-4",
        help="Start learning rate, default is 5e-4.",
    )
    parser.add_argument(
        "--end_lr",
        type=float,
        default="1e-6",
        help="End learning rate, default is 1e-6.",
    )
    parser.add_argument(
        "--sink_size",
        type=int,
        default=16, #origin:100
        help="Sink size for every iteration, default is 100",
    )

    args_opt = parser.parse_args()

    device_id = args_opt.device_id
    context.set_context(
        mode=context.PYNATIVE_MODE, device_target="CPU", device_id=device_id
    )
    if args_opt.distribute == "true":
        D.init()
        device_num = args_opt.device_num
        rank = device_id % device_num
        print("device_id is {}, rank_id is {}".format(device_id, rank))

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        rank = 0
        device_num = 1

    model_args: ModelArgs = ModelArgs(vocab_size=VOCAB_SIZE)
    llama = Llama(model_args)
    llama_with_loss = LlamaWithLoss(llama)

    ds = create_dataset(
        batch_size=model_args.max_batch_size,
        data_path=args_opt.data_path,
        device_num=device_num,
        rank=rank,
    )

    print("crated dataset")

    epoch_num = args_opt.epoch_size
    step_per_epoch = ds.get_dataset_size()

    print(f"step_per_epoch: {step_per_epoch}")

    lr = LearningRate(
        learning_rate=args_opt.start_lr,
        end_learning_rate=args_opt.end_lr,
        warmup_steps=int(step_per_epoch * epoch_num * 0.1),
        decay_steps=epoch_num * step_per_epoch,
    )

    def decay_filter(x):
        return "norm" not in x.name.lower() and "bias" not in x.name.lower()

    params = llama.trainable_params()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [
        {"params": decay_params, "weight_decay": 1e-2},
        {"params": other_params, "weight_decay": 0.0},
        {"order_params": params},
    ]

    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    else:
        optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr)

    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [TimeMonitor(callback_size), LossMonitor(callback_size)]

    config_ck = CheckpointConfig(
        save_checkpoint_steps=1000, keep_checkpoint_max=1
    )
    ckpoint_cb = ModelCheckpoint(prefix="Llama", config=config_ck)
    callback.append(ckpoint_cb)

    llama_with_loss.set_train(True)
    model = Model(llama_with_loss, optimizer=optimizer)
    print("start training")
    model.train(
        actual_epoch_num,
        ds,
        callbacks=callback,
        dataset_sink_mode=True,
        sink_size=callback_size,
    )


if __name__ == "__main__":
    set_seed(2024)
    run_train()

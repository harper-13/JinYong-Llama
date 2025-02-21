import argparse

import mindspore as ms
import numpy as np
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.llama import Attention, KVCache, Llama, ModelArgs
from src.tokenizer import Tokenizer, get_tokenizer
from src.utils import VOCAB_SIZE

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", device_id=0)

# TODO: 根据需要调整生成过程所需的超参数（包括检查点路径），以及采样方式等。


def generate(prompt: str, tokenizer: Tokenizer, model: Llama, max_gen_len: int = 200):
    """
    Text generation
    """
    TOPK = 10

    params = model.params
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)
    total_len = prompt_len + max_gen_len

    # TODO: 在所有 Attention 层中插入 KV 缓存，以避免 KV 的重复计算，加速推理
    for block in model.layers:
        atten: Attention = block.attention
        atten.cache = KVCache(1, total_len, params.n_kv_heads, model.head_dim, np.float32)

    prev_pos = 0
    stop_tokens = list(tokenizer.stop_tokens)
    tokens = ms.Tensor([prompt_tokens + [0] * (total_len - prompt_len)])

    for cur_pos in range(prompt_len, total_len):
        # get the logits for the next token in all the batch rows
        logits = model(tokens[:, prev_pos:cur_pos], prev_pos).asnumpy()
        # sample the next token
        probs = logits[0, -1, :]
        p_args = probs.argsort()[::-1][:TOPK]
        p = probs[p_args]
        p = np.exp(p) / np.sum(np.exp(p))
        target_index = np.random.choice(len(p), p=p)

        prod = int(p_args[target_index])

        tokens[0, cur_pos] = prod

        print(tokenizer.decode(tokens[0, :cur_pos].tolist()), flush=True, end="")

        prev_pos = cur_pos
        print(cur_pos)
        if prod in stop_tokens:
            break

    print("")


def continuation(tokenizer: Tokenizer, model: Llama):
    """Using Llama for fiction continuation.

    Args:
        model (nn.Cell): Llama model
    """
    print(
        'Continuing the text in the style of Jin Yong\'s novels. Press "Ctrl+D" to'
        " exit."
    )
    while True:
        try:
            print("输入一个开头：", end="")
            prompt = input()

            generate(prompt, tokenizer, model)

        except EOFError:
            print("\nBye!")
            break


def main():
    parser = argparse.ArgumentParser(description="GPT inferencing")
    parser.add_argument(
        "--task_type", type=str, default="continuation", help="Evaluation task."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./Llama-1_4.ckpt",
        help="path of checkpoint file.",
    )

    args = parser.parse_args()
    task = args.task_type
    ckpt_path = args.ckpt_path

    model_args: ModelArgs = ModelArgs(vocab_size=VOCAB_SIZE)
    ckpt_dict = load_checkpoint(ckpt_path)

    model = Llama(model_args)

    model.set_train(False)
    load_param_into_net(model, ckpt_dict)

    tokenizer = get_tokenizer("./data/jinyong/射雕英雄传.txt")

    if task == "continuation":
        continuation(tokenizer=tokenizer, model=model)


if __name__ == "__main__":
    main()

"""
Preprocess datasets and transform dataset to mindrecord.
"""

import argparse
import os

import numpy as np
from mindspore.mindrecord import FileWriter
from tqdm.auto import tqdm

from tokenizer import get_tokenizer
from utils import SEQ_LEN, read_jinyong


def chunks(lst, n):
    """yield n sized chunks from list"""
    for i in tqdm(range(len(lst) - n + 1)):
        yield lst[i : i + n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="./dataset/mindrecord")
    parser.add_argument("--file_partition", type=int, default=1)
    parser.add_argument("--file_batch_size", type=int, default=512)
    parser.add_argument("--num_process", type=int, default=16)

    args = parser.parse_args()

    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    mindrecord_schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
    }

    tokenizer = get_tokenizer("./data/jinyong/射雕英雄传.txt")

    transforms_count = 0
    wiki_writer = FileWriter(file_name=args.output_file, shard_num=args.file_partition)
    wiki_writer.add_schema(mindrecord_schema, "JinYong fictions")

    data = read_jinyong("./data/jinyong/射雕英雄传.txt")
    tokens = tokenizer.encode(data)

    for x in chunks(tokens, SEQ_LEN):
        transforms_count += 1
        sample = {
            "input_ids": np.array(x, dtype=np.int32),
        }
        wiki_writer.write_raw_data([sample])
    wiki_writer.commit()
    print("Transformed {} records.".format(transforms_count))

    out_file = args.output_file
    if args.file_partition > 1:
        out_file += "0"
    print("Transform finished, output files refer: {}".format(out_file))

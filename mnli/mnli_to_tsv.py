import sys
import os
import pandas as pd


def convert_mnli_to_sent_pair(mnli_dir, output_dir):
    """
    Convert the entailment
    :return:
    """
    mnli_train_path = os.path.join(mnli_dir, "train.tsv")
    out_path = os.path.join(output_dir, "mnli_train.tsv")

    df = pd.read_csv(mnli_train_path, sep='\t', error_bad_lines=False)
    df.dropna(subset=["sentence1", "sentence2", "gold_label"], inplace=True)
    entail_df = df[df.gold_label == "entailment"]

    col_list = ["sentence1", "sentence2"]
    entail_df = entail_df[col_list]

    entail_df.info()

    entail_df.to_csv(out_path, sep="\t", index=False, header=False)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python ... [mnli_dir] [output_dir]", file=sys.stderr)
        exit(1)

    mnli_dir = sys.argv[1]
    output_dir = sys.argv[2]

    convert_mnli_to_sent_pair(mnli_dir, output_dir)
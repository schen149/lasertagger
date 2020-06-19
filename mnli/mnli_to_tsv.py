import sys
import os
import pandas as pd


def convert_mnli_to_sent_pair(mnli_path, out_path):
    """
    Convert the entailment
    :return:
    """
    df = pd.read_csv(mnli_path, sep='\t', error_bad_lines=False)
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

    mnli_train_path = os.path.join(mnli_dir, "train.tsv")
    train_out_path = os.path.join(output_dir, "mnli_train.tsv")
    convert_mnli_to_sent_pair(mnli_train_path, train_out_path)

    mnli_dev_matched_path = os.path.join(mnli_dir, "dev_matched.tsv")
    dev_matched_out_path = os.path.join(output_dir, "mnli_dev_matched.tsv")
    convert_mnli_to_sent_pair(mnli_dev_matched_path, dev_matched_out_path)

    mnli_dev_mismatched_path = os.path.join(mnli_dir, "dev_mismatched.tsv")
    dev_mismatched_path = os.path.join(output_dir, "mnli_dev_mismatched.tsv")
    convert_mnli_to_sent_pair(mnli_dev_mismatched_path, dev_mismatched_path)
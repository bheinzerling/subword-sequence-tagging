from pathlib import Path
import argparse


def get_arg_parser():
    desc = "Multilingual sequence tagging with subwords"
    a = argparse.ArgumentParser(description=desc)
    a.add_argument("command", type=str)
    a.add_argument("--dataset", type=str, required=True)
    a.add_argument("--lang", type=str, nargs="+", required=True)
    a.add_argument("--bpemb-lang", type=str)
    a.add_argument("--tag", type=str, required=True)
    a.add_argument("--tag-scheme", type=str, choices=["BIO", "IOBES"])
    a.add_argument("--data-dir", type=Path, default="data")
    a.add_argument("--cache-dir", type=Path, default="out/cache")
    a.add_argument("--no-dataset-cache", action="store_true")
    a.add_argument("--no-dataset-tensorize", action="store_true")
    a.add_argument("--outdir", type=Path, default="out")
    a.add_argument("--rundir", type=Path)
    a.add_argument("--vocab-size", type=int, default=10000)
    a.add_argument("--eval-inst", type=int)
    a.add_argument("--test-every-eval", action="store_true")
    a.add_argument("--emb", type=str)
    a.add_argument("--types", type=str, choices=["3class"], default="3class")
    a.add_argument("--bpemb-dim", type=int, default=100)
    a.add_argument("--shape-emb-dim", type=int, default=100)
    a.add_argument("--char-emb-dim", type=int, default=100)
    a.add_argument("--len-emb-dim", type=int, default=32)
    a.add_argument("--emb-fixed", action="store_true")
    a.add_argument("--bert-fixed", action="store_true")
    a.add_argument("--bert-max-seq-len", type=int, default=512)
    a.add_argument("--char-emb-fixed", action="store_true")
    a.add_argument("--shape-emb-fixed", action="store_true")
    a.add_argument("--relearn-repr", action="store_true")
    a.add_argument("--relearn-dim", type=int)
    a.add_argument("--use-char", action="store_true")
    a.add_argument("--use-shape", action="store_true")
    a.add_argument("--use-bpe", action="store_true")
    a.add_argument("--use-fasttext", action="store_true")
    a.add_argument("--use-bert", action="store_true")
    a.add_argument("--use-meta-rnn", action="store_true")
    a.add_argument(
        "--fasttext-emb-file", type=str,
        default="data/{dataset}/fasttext/{lang}.w2v.bin")
    a.add_argument(
        "--bert-model-name",
        type=str,
        default="bert-base-multilingual-uncased")
    a.add_argument("--emb-random-init", action="store_true")
    a.add_argument("--dropout", type=float, default=0.5)
    a.add_argument("--emb-dropout", type=float, default=0.0)
    a.add_argument("--rnn-type", type=str, default="LSTM")
    a.add_argument("--nonlin", type=str, default="ReLU")
    a.add_argument("--batch-size", type=int, default=64)
    a.add_argument("--eval-batch-size", type=int, default=1)
    a.add_argument("--max-epochs", type=int, default=1000)
    a.add_argument("--min-epochs", type=int, default=50)
    a.add_argument("--optim", type=str, default="adam")
    a.add_argument("--lr-scheduler", type=str)
    a.add_argument("--lr-patience", type=int, default=2)
    a.add_argument("--momentum", type=float, default=0)
    a.add_argument("--learning-rate", type=float, default=0.0001)
    a.add_argument("--learning-decay-rate", type=float, default=0.05)
    a.add_argument("--weight-decay", type=float, default=0)
    a.add_argument("--model-file", type=str)
    a.add_argument("--char-enc-file", type=str)
    a.add_argument("--tag-enc-file", type=str)
    a.add_argument("--random-seed", type=int, default=1)
    a.add_argument("--eval-every", type=int, default=5)
    a.add_argument("--n-examples", type=int, default=3)
    a.add_argument("--nlayers", type=int, default=2)
    a.add_argument("--char-nhidden", type=int, default=256)
    a.add_argument("--bpe-nhidden", type=int, default=256)
    a.add_argument("--meta-nhidden", type=int, default=256)
    a.add_argument("--rnn-dropout", type=float, default=0.1)
    a.add_argument("--repr-dropout", type=float, default=0.3)
    a.add_argument("--max-grad-norm", type=float, default=-1)
    a.add_argument("--print-examples", action="store_true")
    a.add_argument("--runid", type=str)
    a.add_argument("--early-stop", type=int, default=10)
    a.add_argument("--early-stop-min-delta", type=float, default=0.0)
    a.add_argument("--gpu-id", type=int, default=0)
    a.add_argument("--first-eval-epoch", type=int, default=1)
    a.add_argument("--crossval-idx", type=int)
    a.add_argument("--max-ninst", type=int)
    a.add_argument("--max-eval-inst", type=int)
    a.add_argument("--max-eval-ninst", type=int)
    a.add_argument("--best-vocab-size", action="store_true")
    a.add_argument(
        "--best-vocab-size-file", type=Path,
        default="data/ud_1_2_best_vocab_size.json")
    return a


def get_args():
    a = get_arg_parser()
    args = a.parse_args()
    if len(args.lang) == 1:
        args.lang = args.lang[0]
    return args

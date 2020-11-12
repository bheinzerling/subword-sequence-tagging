from copy import deepcopy

# from scripts.conll17_ud_eval import (
#     load_conllu_file, evaluate as conll2017_eval)


def save_conllu(sents, rundir, eval_name):
    outfile = rundir / f"{eval_name}.conllu"
    with outfile.open("w") as out:
        out.write("".join(sent.serialize() for sent in sents))
    return outfile


def evaluate(dataset, split_raw, pred, *, rundir=None, eval_name=None):
    split_pred = deepcopy(split_raw)
    for sent, sent_pred in zip(split_pred, pred):
        sent_pred = dataset.tag_codec.inverse_transform(sent_pred)
        for token, pred_tag in zip(sent, sent_pred):
            token[dataset.tag] = pred_tag
    gold_file = save_conllu(split_raw, rundir=rundir, eval_name="gold")
    pred_file = save_conllu(split_pred, rundir=rundir, eval_name=eval_name)
    gold_ud = load_conllu_file(gold_file)
    pred_ud = load_conllu_file(pred_file)
    eval_result = conll2017_eval(gold_ud, pred_ud)
    eval_key = {"xpostag": "XPOS", "upostag": "UPOS"}[dataset.tag]
    return eval_result[eval_key].aligned_accuracy

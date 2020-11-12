from itertools import islice
from functools import partial

import torch

from argparser import get_args
from trainer import Trainer


def train(conf):
    t = Trainer(conf)
    optim = t.optimizer
    model = t.model

    def train_epoch(*args, **kwargs):
        for i, batch in enumerate(t.data.batch_iter_train):
            optim.zero_grad()
            tag_true = batch["token"][3]
            _, loss = model(batch, tag_true=tag_true)
            loss.backward()
            optim.step()
            t.losses.append(loss)
        if hasattr(t, "lr_scheduler"):
            t.lr_scheduler.step(t.losses[0].current)

    def train_epoch_multilang(*args, **kwargs):
        for i, (lang_idx, batch) in enumerate(t.batch_iter_train_multilang):
            optim.zero_grad()
            tag_true = batch["token"][3]
            _, loss = model(batch, tag_true=tag_true, lang_idx=lang_idx)
            loss.backward()
            optim.step()
            t.losses.append(loss)

    def do_eval(ds_iter, *args, **kwargs):
        for batch in islice(ds_iter(), conf.max_eval_inst):
            sorted_len, sort_idx, tag_true = batch["token"][1:4]
            tag_pred, loss = model(batch, tag_true=tag_true)
            unsort_idx = torch.sort(sort_idx)[1]
            for l, true, pred in zip(
                    sorted_len[unsort_idx],
                    tag_pred[unsort_idx],
                    tag_true[unsort_idx]):
                t.score.add(pred[:l], true[:l])
        return t.score.current

    def do_eval_multi(ds_iter, *args, **kwargs):
        for lang, ds in ds_iter():
            for batch in ds:
                sorted_len, sort_idx, tag_true = batch["token"][1:4]
                tag_pred, loss = model(batch, tag_true=tag_true)
                unsort_idx = torch.sort(sort_idx)[1]
                for l, true, pred in zip(
                        sorted_len[unsort_idx],
                        tag_pred[unsort_idx],
                        tag_true[unsort_idx]):
                    t.score[lang].add(pred[:l], true[:l])
        return t.avg_score.current

    _train_epoch = train_epoch
    if t.data.is_multilingual:
        _do_eval = partial(do_eval_multi, lambda: t.data.iter_dev)
        do_test = partial(do_eval_multi, lambda: t.data.iter_test)
    else:
        _do_eval = partial(do_eval, lambda: t.main_lang_data.iter_dev)
        do_test = partial(do_eval, lambda: t.main_lang_data.iter_test)
    _do_test = do_test if conf.test_every_eval else None
    t.train(_train_epoch, _do_eval, do_test=_do_test, eval_ds_name="dev")
    if t.data.is_multilingual:
        score = t.avg_score
    else:
        score = t.score
    conf.model_file = score.best_model
    test_score = test(conf)
    if t.data.is_multilingual:
        test_score, lang_scores = test_score
        for lang, lang_score in lang_scores.items():
            t.log.info(f"{lang} score: {lang_score.current:.4}")
    t.log.info(f"final score: {test_score:.4}")


def test(conf, model=None):
    t = Trainer(conf)
    if model is None:
        model = t.model

    if t.data.is_multilingual:
        def do_test(*args, **kwargs):
            for lang, ds in t.data.iter_test:
                for batch in ds:
                    sorted_len, sort_idx, tag_true = batch["token"][1:4]
                    tag_pred, loss = model(batch, tag_true=tag_true)
                    unsort_idx = torch.sort(sort_idx)[1]
                    for l, true, pred in zip(
                            sorted_len[unsort_idx],
                            tag_pred[unsort_idx],
                            tag_true[unsort_idx]):
                        t.score[lang].add(pred[:l], true[:l])
            return t.avg_score.current
    else:
        def do_test(*args, **kwargs):
            for batch in islice(
                    t.main_lang_data.iter_test, conf.max_eval_inst):
                sorted_len, sort_idx, tag_true = batch["token"][1:4]
                tag_pred, loss = model(batch, tag_true=tag_true)
                unsort_idx = torch.sort(sort_idx)[1]
                for l, true, pred in zip(
                        sorted_len[unsort_idx],
                        tag_pred[unsort_idx],
                        tag_true[unsort_idx]):
                    t.score.add(pred[:l], true[:l])
            return t.score.current

    score = t.do_eval(do_test, eval_ds_name="test")
    if t.data.is_multilingual:
        avg_score = score
        lang_scores = t.score
        for lang, lang_score in lang_scores.items():
            t.log.info(f"{lang} score: {lang_score.current:.4}")
        t.log.info(f"avg score: {avg_score:.4}")
        return avg_score, lang_scores
    return score


if __name__ == "__main__":
    conf = get_args()
    conf.bpemb_lang = conf.lang
    globals()[conf.command.replace("-", "_")](conf)

import random

import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import datasets
from model import SequenceTagger
from bert_wrapper import Transformer as Bert
from util import (
    get_logger,
    load_word2vec_file,
    mkdir,
    next_rundir,
    json_load,
    dump_args,
    get_optim,
    emb_layer,
    LossTrackers,
    Score,
    EarlyStopping,
    set_random_seed,
    save_model,
    ConllScore,
    )


def load_dataset(conf, lang, bert=None):
    if conf.best_vocab_size:
        conf.vocab_size = json_load(conf.best_vocab_size_file)[conf.lang]
    data = datasets[conf.dataset].load(conf, lang, bert=bert)
    data.describe()
    return data


class Trainer():
    def __init__(self, conf):
        self.conf = conf
        self.device = torch.device(f"cuda:{conf.gpu_id}")
        self.log = get_logger()
        torch.set_printoptions(precision=8)
        if conf.runid:
            conf.rundir = mkdir(conf.outdir / conf.runid)
        if not conf.rundir:
            conf.rundir = next_rundir(conf.outdir, log=self.log)
        self.rundir = conf.rundir
        dump_args(conf, conf.rundir / "conf.json")
        set_random_seed(conf.random_seed)
        if self.conf.use_bert:
            assert self.conf.lang in Bert.supported_langs, self.conf.lang
            self.bert = Bert(self.conf.bert_model_name, device=self.device)
        else:
            self.bert = None
        self.data = load_dataset(conf, conf.lang, bert=self.bert)
        _data = [self.data]
        for d in _data:
            self.log.info(
                f"{len(d.train_loader)} batches | bs {conf.batch_size}")
        self.model = self.get_model()
        self.optimizer = get_optim(conf, self.model)
        optimum = "min"
        if conf.lr_scheduler == "plateau":
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer, factor=0.1, patience=2, mode=optimum,
                verbose=True)
        elif conf.lr_scheduler:
            raise ValueError("Unknown lr_scheduler: " + conf.lr_scheduler)
        self.losses = LossTrackers.from_names("loss", log=self.log)
        if (
                self.main_lang_data.tag == "ner" or
                self.conf.dataset.startswith("sr3de")):
            if self.data.is_multilingual:
                self.sentence_texts = {
                    split_name: self.main_lang_data.token_texts(split_name)
                    for split_name in ["dev", "test"]}
                self.conll_score = {
                    lang: ConllScore(tag_enc=self.main_lang_data.tag_enc)
                    for lang in self.data.dev}
                self.score = {
                    lang: Score(
                        "f1", save_model=False, log=self.log,
                        score_func=self.conll_score[lang],
                        add_mode="append")
                    for lang in self.data.dev}
                self.avg_score = Score(
                    "avg_f1",
                    log=self.log,
                    score_func="dummy",
                    add_mode="append")
            else:
                self.sentence_texts = {
                    split_name: self.main_lang_data.token_texts(
                        split_name)[:conf.max_eval_inst]
                    for split_name in ["dev", "test"]}
                self.conll_score = ConllScore(
                    tag_enc=self.main_lang_data.tag_enc)
                self.score = Score(
                    "f1",
                    log=self.log,
                    score_func=self.conll_score,
                    add_mode="append")
        else:
            if self.data.is_multilingual:
                self.score = {
                    lang: Score("acc", log=self.log)
                    for lang in self.data.dev}
                self.avg_score = Score(
                    "avg_acc", log=self.log,
                    score_func="dummy",
                    add_mode="append")
            else:
                self.score = Score("acc", log=self.log)
        if conf.early_stop > 0:
            score_optimum = (
                "max" if (
                    self.conf.dataset.startswith("wikiannmulti") or
                    self.data.is_multilingual)
                else self.score.optimum)
            self.early_stop = EarlyStopping(
                score_optimum,
                min_delta=conf.early_stop_min_delta,
                patience=conf.early_stop)
        else:
            self.early_stop = None
        self.epoch = 0

    def get_model(self):
        ntags = self.data.tag_enc.nlabels
        nshapes = self.data.shape_enc.nlabels
        nchars = self.data.char_enc.nlabels
        bpe_emb = emb_layer(
            self.data.bpemb.vectors,
            trainable=not self.conf.emb_fixed,
            use_weights=not self.conf.emb_random_init)
        if self.conf.use_fasttext:
            fasttext_file = self.conf.fasttext_emb_file.format(
                dataset=self.conf.dataset, lang=self.data.lang)
            fasttext_emb = emb_layer(
                load_word2vec_file(fasttext_file, add_unk=True),
                trainable=not self.conf.emb_fixed,
                use_weights=not self.conf.emb_random_init)
        else:
            fasttext_emb = None
        model = SequenceTagger(
            bpe_emb,
            ntags,
            self.conf,
            nchars=nchars,
            nshapes=nshapes,
            fasttext_emb=fasttext_emb,
            bert=self.bert,
            tag_enc=self.main_lang_data.tag_enc,
            ).to(self.device)
        self.log.info(f'model repr dim: {model.repr_dim}')
        if self.conf.model_file:
            self.log.info(f"loading model {self.conf.model_file}")
            model.load_state_dict(torch.load(self.conf.model_file))
            self.log.info(f"loaded model {self.conf.model_file}")
        return model

    def train(self, train_epoch, do_eval, do_test=None, eval_ds_name=None):
        try:
            for epoch in range(1, self.conf.max_epochs + 1):
                self.epoch = epoch
                self.model.train()
                train_epoch(epoch=epoch)
                self.losses.interval_end_log(epoch, ds_name="train")
                burnin_done = epoch >= self.conf.first_eval_epoch
                if burnin_done and not epoch % self.conf.eval_every:
                    score = self.do_eval(
                        do_eval, epoch=epoch, eval_ds_name=eval_ds_name)
                    if do_test:
                        self.do_eval(
                            do_test, epoch=epoch, eval_ds_name="test")
                    if score is not None and self.early_stop:
                        if self.early_stop.step(score):
                            if epoch >= self.conf.min_epochs:
                                patience = self.early_stop.patience
                                self.log.info(
                                    f"Early stop after {patience} steps")
                                break
        except KeyboardInterrupt:
            self.log.info("Stopping training due to keyboard interrupt")

    def do_eval(self, eval_func, epoch=None, eval_ds_name=None):
        self.model.eval()
        eval_func(epoch=epoch)
        self.log_eval(ds_name=eval_ds_name, epoch=epoch)
        if self.data.is_multilingual:
            return self.avg_score.current
        return self.score.current

    def log_eval(self, ds_name=None, epoch=None):
        self.losses.interval_end(ds_name=ds_name)
        if self.data.is_multilingual:
            for lang in getattr(self.data, ds_name):
                if hasattr(self, "conll_score"):
                    self.conll_score[lang].sentences = \
                        self.sentence_texts[ds_name][lang]
                    fname = f"{epoch}.{ds_name}.{lang}.conll"
                    self.conll_score[lang].outfile = self.rundir / fname
                self.score[lang].update()
            avg_score = np.average([
                score.current for score in self.score.values()])
            self.avg_score.update_log(
                model=self.model,
                rundir=self.rundir,
                epoch=epoch,
                score=avg_score)
        else:
            if hasattr(self, "conll_score"):
                self.conll_score.sentences = self.sentence_texts[ds_name]
                fname = f"{epoch}.{ds_name}.conll"
                self.conll_score.outfile = self.rundir / fname
            self.score.update_log(self.model, self.rundir, epoch)

    def save_model(self):
        model_file = self.rundir / f"model.e{self.epoch}.pt"
        save_model(self.model, model_file, self.log)

    @property
    def main_lang_data(self):
        return self.data[0] if isinstance(self.data, list) else self.data

    @property
    def batch_iter_train_multilang(self):
        main_lang_len = len(self.data[0].train_loader)
        max_sim_lang_len = int(self.conf.sim_lang_ratio * main_lang_len)

        def get_sim_lang_len(i):
            sim_lang_len = len(self.data[i].train_loader)
            return min(sim_lang_len, max_sim_lang_len)

        lang_idxs = [
            i
            for i, data in enumerate(self.data)
            for _ in range(main_lang_len if i == 0 else get_sim_lang_len(i))]
        random.shuffle(lang_idxs)
        iters = [data.batch_iter_train for data in self.data]
        return ((i, next(iters[i])) for i in lang_idxs)

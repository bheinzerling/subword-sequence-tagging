from pathlib import Path
from abc import ABC, abstractmethod
from collections import Counter
from itertools import islice


import torch
from torch import tensor, sort, cat
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import joblib

import conllu

from boltons.iterutils import split_iter

from util import (
    lines,
    flatten,
    mkdir,
    token_shapes,
    load_word2vec_file,
    to_word_indexes,
    map_assert,
    map_skip_assert_error,
    iob_to,
    split_by_ratios,
    LabelEncoder,
    cached_property,
    )
from bpemb import BPEmb


class ListDataset(list, Dataset):
    def __init__(self, items):
        super().__init__(items)


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def collate_fn(sents):
    # collate char-level data for the whole sentence
    char_len = tensor([len(sent["char"]) for sent in sents]).cuda()
    char_sorted_len, char_sort_idx = sort(char_len, descending=True)

    # collate char-level data for each token
    char_token = [ct for sent in sents for ct in sent["char_token"]]
    char_token_len = cat([sent["char_token_len"] for sent in sents])
    char_token_sorted_len, char_token_sort_idx = sort(
        char_token_len, descending=True)
    char_token_padded = pad_sequence(
        [char_token[i] for i in char_token_sort_idx], batch_first=True)

    # collate bpe-segment-level data for the whole sentence
    bpe_len = tensor([len(sent["bpe"]) for sent in sents]).cuda()
    bpe_sorted_len, bpe_sort_idx = sort(bpe_len, descending=True)

    # collate bpe-level data for each token
    bpe_token = [ct for sent in sents for ct in sent["bpe_token"]]
    bpe_token_len = cat([sent["bpe_token_len"] for sent in sents])
    bpe_token_sorted_len, bpe_token_sort_idx = sort(
        bpe_token_len, descending=True)
    bpe_token_padded = pad_sequence(
        [bpe_token[i] for i in bpe_token_sort_idx], batch_first=True)

    # collate token-level data
    token_len = tensor([len(sent["tag"]) for sent in sents]).cuda()
    token_sorted_len, token_sort_idx = sort(token_len, descending=True)
    tag = [sents[i]["tag"] for i in token_sort_idx]
    tag_padded = pad_sequence(tag, batch_first=True, padding_value=-1)
    token_shape = [sents[i]["token_shape"] for i in token_sort_idx]
    token_shape_padded = pad_sequence(token_shape, batch_first=True)
    if "word" in sents[0]:
        word = [sents[i]["word"] for i in token_sort_idx]
        word_padded = pad_sequence(word, batch_first=True)
    else:
        word_padded = None
    if "fasttext" in sents[0]:
        fasttext = [sents[i]["fasttext"] for i in token_sort_idx]
        fasttext_padded = pad_sequence(fasttext, batch_first=True)
    else:
        fasttext_padded = None
    if "bert_ids" in sents[0]:
        bert_batch = [
            torch.cat([sents[i][key] for i in token_sort_idx], dim=0)
            for key in ("bert_ids", "bert_mask", "bert_token_starts")]
    else:
        bert_batch = None
    tokens_raw = [sents[i]["token"] for i in token_sort_idx]

    return {
        "char": (
            char_token_padded, char_token_sorted_len, char_token_sort_idx,
            token_len),
        "bpe": [
            bpe_token_padded, bpe_token_sorted_len, bpe_token_sort_idx,
            token_len],
        "token": (
            token_shape_padded, token_sorted_len, token_sort_idx, tag_padded,
            word_padded),
        "fasttext": (
            token_shape_padded, token_sorted_len, token_sort_idx, tag_padded,
            fasttext_padded),
        "bert": bert_batch,
        "token_raw": tokens_raw}


class DatasetBase(ABC):
    is_multilingual = False

    def __init__(self, conf, lang, bert=None):
        self.conf = conf
        self.lang = lang
        self.bert = bert
        self.device = torch.device(f"cuda:{conf.gpu_id}")
        self.name = conf.dataset
        self.tag = conf.tag
        self.batch_size = conf.batch_size
        self.eval_batch_size = conf.eval_batch_size
        self.examples_to_print = conf.n_examples

        if self.conf.tag_scheme:
            self.convert_tags = iob_to[self.tag_scheme]
        self.load_data_raw()
        self.NO_TAG = "NO_TAG"
        tags = self.get_tags()
        print(Counter(tags).most_common())
        shapes = self.get_shapes()
        char_enc = None
        if conf.char_enc_file:
            assert Path(conf.char_enc_file).exists()
            char_enc = joblib.load(conf.char_enc_file)
        if self.name.endswith("multi_finetune"):
            assert char_enc
        if char_enc:
            self.char_enc = char_enc
        else:
            chars = self.get_chars()
            self.char_enc = LabelEncoder(
                to_torch=True, device=self.device).fit(chars)
        tag_enc = None
        if conf.tag_enc_file:
            assert Path(conf.tag_enc_file).exists()
            tag_enc = joblib.load(conf.tag_enc_file)
        if tag_enc:
            self.tag_enc = tag_enc
        else:
            self.tag_enc = LabelEncoder(
                to_torch=True, device=self.device).fit(tags)
        self.shape_enc = LabelEncoder(
            to_torch=True, device=self.device).fit(shapes)

        self.bpemb = BPEmb(
            lang=conf.bpemb_lang,
            vs=conf.vocab_size,
            dim=conf.bpemb_dim,
            add_pad_emb=True)
        if conf.use_fasttext:
            f = conf.fasttext_emb_file.format(dataset=self.name, lang=lang)
            self.fasttext_emb = load_word2vec_file(f, add_unk=True)
        self.pad_idx = self.bpemb.emb.vocab["<pad>"].index
        if not conf.no_dataset_tensorize:
            self.tensorize()

    @abstractmethod
    def load_data_raw(self):
        pass

    @abstractmethod
    def get_chars(self):
        pass

    @abstractmethod
    def get_tags(self):
        pass

    @abstractmethod
    def get_shapes(self):
        pass

    @abstractmethod
    def tensorize(self):
        pass

    def tensorize_sent(self, sent):
        tags_str = [token[self.tag] or self.NO_TAG for token in sent]
        tags = self.tag_enc.transform(tags_str)
        tokens = [token["form"] for token in sent]
        token_shape = self.shape_enc.transform(token_shapes(tokens))

        bpe_ids = [
            self.bpemb.encode_ids([token["form"]])[0] for token in sent]
        bpe_token_start_mask = self.start_mask(bpe_ids)
        bpe_token_end_mask = self.end_mask(bpe_ids)
        bpe_ids = tensor(list(flatten(bpe_ids))).to(device=self.device)
        assert bpe_token_start_mask.shape == bpe_ids.shape
        assert bpe_token_start_mask.sum().item() == len(tags)
        assert bpe_token_end_mask.shape == bpe_ids.shape
        assert bpe_token_end_mask.sum().item() == len(tags)

        try:
            chars = self.char_enc.transform([
                [char for char in token["form"]] for token in sent])
        except ValueError as e:
            print(e)
            return None
        char_token_start_mask = self.start_mask(chars)
        char_token_end_mask = self.end_mask(chars)
        chars = tensor(list(flatten(chars))).to(device=self.device)

        char_token, char_token_len = self.sub_token_and_len(
            chars, char_token_start_mask)
        bpe_token, bpe_token_len = self.sub_token_and_len(
            bpe_ids, bpe_token_start_mask)

        tensorized = {
            "token": tokens,
            "tag": tags,
            "token_shape": token_shape,
            "bpe": bpe_ids,
            "bpe_token": bpe_token,
            "bpe_token_len": bpe_token_len,
            "bpe_token_start_mask": bpe_token_start_mask,
            "bpe_token_end_mask": bpe_token_end_mask,
            "char": chars,
            "char_token_start_mask": char_token_start_mask,
            "char_token_end_mask": char_token_end_mask,
            "char_token": char_token,
            "char_token_len": char_token_len,
            }
        if hasattr(self, "fasttext_emb"):
            tensorized["fasttext"] = tensor(
                to_word_indexes(
                    [token["form"].lower() for token in sent],
                    self.fasttext_emb,
                    unk="<unk>")).to(device=self.device)
        if self.bert is not None:
            try:
                tensorized["bert_ids"], \
                    tensorized["bert_mask"], \
                    tensorized["bert_token_starts"] = \
                    self.bert.subword_tokenize_to_ids(tokens)
                assert len(tensorized["bert_ids"]) <= self.conf.bert_max_seq_len
                if self.examples_to_print > 0:
                    print(tokens)
                    print(
                        self.bert.model_name,
                        self.bert.subword_tokenize(tokens))
                    self.examples_to_print -= 1
            except AssertionError as e:
                print(e)
                return None
            # TODO: ta (Tamil) WikiAnn has weird whitespace characters that
            # are treated differently by the BERT tokenizer, leading to
            # mismatches in tag and token counts
            if len(tags) != tensorized["bert_token_starts"].sum():
                print("Skipping instance with inconsistent tokenization:")
                print(" ## ".join(tags_str))
                print(" ## ".join(tokens))
                return None

        return tensorized

    @staticmethod
    def start_mask(subsegments):
        mask = list(flatten(
            [[1] + [0] * (len(ids) - 1) for ids in subsegments]))
        return tensor(mask).cuda().byte()

    @staticmethod
    def end_mask(subsegments):
        mask = list(flatten(
            [([0] * (len(ids) - 1)) + [1] for ids in subsegments]))
        return tensor(mask).cuda().byte()

    @staticmethod
    def sub_token_and_len(sub, sub_token_mask):
        char_token_start = sub_token_mask.nonzero().squeeze(1)
        char_token_end = cat([
            char_token_start[1:],
            tensor([sub_token_mask.size(0)]).to(char_token_start)])
        char_token = [
            sub[s:e] for s, e in zip(char_token_start, char_token_end)]
        char_token_len = char_token_end - char_token_start
        return char_token, char_token_len

    def token_texts(self, split_name):
        split = getattr(self, split_name)
        return [instance["token"] for instance in split]

    def assert_batch_size(self):
        if not hasattr(self, "batch_size"):
            raise ValueError(
                "Need to set batch_size before calling train_loader")

    def assert_eval_batch_size(self):
        if not hasattr(self, "eval_batch_size"):
            raise ValueError(
                "Need to set eval_batch_size before calling"
                "dev_loader or test_loader")

    def loader(self, dataset, **kwargs):
        return DataLoader(dataset, collate_fn=collate_fn, **kwargs)


class CachedDataset(DatasetBase):
    """Mixin for automatically storing and loadinng a tensorized
    dataset to/from a file cache.
    """
    @classmethod
    def load(cls, conf, lang, bert=None):
        mkdir(conf.cache_dir)
        fasttext_emb = conf.fasttext_emb_file if conf.use_fasttext else None
        fname = (
            f"{conf.dataset}.{lang}." +
            (f"max{conf.max_ninst}." if conf.max_ninst else "") +
            (f"maxeval{conf.max_eval_ninst}." if conf.max_eval_ninst else "") +
            (f"cv{conf.crossval_idx}." if conf.crossval_idx is not None else "") +
            (f"bert{conf.bert_max_seq_len}." if bert is not None else "") +
            (f"fasttext." if fasttext_emb is not None else "") +
            f"vs{conf.vocab_size}.{conf.tag}." +
            (f"{conf.tag_scheme}." if conf.tag_scheme else "") +
            "pt"
            )
        cache_file = conf.cache_dir / fname
        ds = None
        try:
            print("loading", cache_file)
            ds = torch.load(cache_file)
            print("loaded", cache_file)
            ds.bpemb = BPEmb(
                lang=conf.bpemb_lang,
                vs=conf.vocab_size,
                dim=conf.bpemb_dim,
                add_pad_emb=True)
        except FileNotFoundError:
            pass
        if ds is None:
            print(f"Loading dataset {conf.dataset} {lang}")
            ds = cls(conf, lang, bert=bert)
            bpemb = ds.bpemb
            ds.bpemb = None  # cannot pickle SwigPyObject
            torch.save(ds, cache_file)
            ds.bpemb = bpemb
        return ds


class SplitDataset(CachedDataset):
    """Mixin providing train/dev/test Dataloaders.
    """
    split_names = ["train", "dev", "test"]

    @cached_property
    def train_loader(self):
        self.assert_batch_size()
        return self.loader(self.train, batch_size=self.batch_size)

    @property
    def batch_iter_train(self):
        return iter(self.train_loader)

    @cached_property
    def dev_loader(self):
        self.assert_eval_batch_size()
        return self.loader(
            self.dev, batch_size=self.eval_batch_size, shuffle=False)

    @property
    def iter_dev(self):
        return iter(self.dev_loader)

    @cached_property
    def test_loader(self):
        self.assert_eval_batch_size()
        return self.loader(
            self.test, batch_size=self.eval_batch_size, shuffle=False)

    @property
    def iter_test(self):
        return iter(self.test_loader)

    def describe(self, log=None):
        info = log.info if log else print
        if hasattr(self, "split_files"):
            files = self.split_files
        else:
            files = [self.file] * len(self.splits)
        for name, split, split_raw, split_file in zip(
                self.split_names, self.splits, self.splits_raw, files):
            info(f"{name}: {len(split)}/{len(split_raw)} {split_file}")
            i = 3
            tokens = self.bpemb.decode_ids(split[i]["bpe"]).split()
            tags = self.tag_enc.inverse_transform(split[i]["tag"].cpu())
            info(" ".join(f"{tok}/{tag}" for tok, tag in zip(tokens, tags)))


class FixedSplitDataset(SplitDataset):
    """A dataset with pre-defined train/dev/test splits.
    """
    def load_data_raw(self):
        self.load_splits_raw()
        for split_name, split_raw in zip(self.split_names, self.splits_raw):
            setattr(self, split_name + "_raw", split_raw)

    def get_tags(self):
        return [
            token[self.tag] or self.NO_TAG
            for split in self.splits_raw
            for sent in split
            for token in sent]

    def get_shapes(self):
        return [
            shape
            for split in self.splits_raw
            for sent in split
            for shape in token_shapes([token["form"] for token in sent])]

    def get_chars(self):
        return [
            char
            for split in self.splits_raw
            for sent in split
            for token in sent
            for char in token["form"]]

    def tensorize(self):
        self.splits = []
        for split_raw, split_name in zip(self.splits_raw, self.split_names):
            split = ListDataset(
                filter(
                    lambda t: t is not None,
                    map(self.tensorize_sent, split_raw)))
            self.splits.append(split)
            setattr(self, split_name, split)


class SingleDataset(CachedDataset):
    """A dataset without any splits.
    """
    def get_tags(self):
        return [
            token[self.tag] or self.NO_TAG
            for sent in self.data_raw
            for token in sent]

    def get_shapes(self):
        return [
            shape
            for sent in self.data_raw
            for shape in token_shapes([token["form"] for token in sent])]

    def get_chars(self):
        return [
            char
            for sent in self.data_raw
            for token in sent
            for char in token["form"]]

    def tensorize(self):
        self.data = ListDataset(
            filter(
                lambda t: t is not None,
                map(self.tensorize_sent, self.data_raw)))

    def describe(self, log=None):
        info = log.info if log else print
        info(f"{len(self.data)}/{len(self.data_raw)} {self.file}")
        i = 3
        tokens = self.bpemb.decode_ids(self.data[i]["bpe"]).split()
        tags = self.tag_enc.inverse_transform(self.data[i]["tag"].cpu())
        info(" ".join(f"{tok}/{tag}" for tok, tag in zip(tokens, tags)))


class CrossValidationDataset(SingleDataset, SplitDataset):
    """A datasets with random train/dev/test splits for
    cross-validation.
    """
    def __init__(self, *args, split_ratios=[0.6, 0.2], **kwargs):
        super().__init__(*args, **kwargs)
        self.split_ratios = split_ratios
        self.new_crossval_split()
        self.check_token_text_tag_lengths("dev")
        self.check_token_text_tag_lengths("test")

    def new_crossval_split(self):
        idxs = torch.randperm(len(self.data))
        split_idxss = split_by_ratios(idxs, *self.split_ratios)
        self.splits = []
        for split_name, split_idxs in zip(self.split_names, split_idxss):
            split = Subset(self.data, split_idxs)
            self.splits.append(split)
            setattr(self, split_name, split)
            split_raw = [self.data_raw[i] for i in split_idxs]
            setattr(self, split_name + "_raw", split_raw)
        # invalidate @cached_property for each loader
        for split_name in self.split_names:
            try:
                del self.__dict__[split_name + "_loader"]
            except KeyError:
                pass

    def check_token_text_tag_lengths(self, split_name):
        split = getattr(self, split_name)
        token_texts = self.token_texts(split_name)
        mismatches = 0
        for i in range(len(split)):
            if len(split[i]["tag"]) != len(token_texts[i]):
                mismatches += 1
        print(split_name, mismatches, "/", len(split), "mismatches")


class MultilingualDataset(CachedDataset):
    """A multilingual dataset comprised of multiple monolingual subsets.
    """
    split_names = ["train", "dev", "test"]
    is_multilingual = True

    def iter_forms(self):
        return (
            token["form"]
            for sent in flatten(self.split_sents())
            for token in sent)

    def split_sents(self):
        return [
            self.splits_raw[0],
            flatten(self.splits_raw[1].values()),
            flatten(self.splits_raw[2].values())]

    def get_chars(self):
        return list(flatten(self.iter_forms()))

    def get_shapes(self):
        return ["Aa", "a", ".", "0", "A", "0a0", "%"]

    def tensorize(self):
        self.splits = []

        def make_ds(raw):
            return ListDataset(
                filter(
                    lambda t: t is not None,
                    map(self.tensorize_sent, raw)))

        def make_per_lang_ds(lang2raw):
            return {lang: make_ds(raw) for lang, raw in lang2raw.items()}

        self.train = make_ds(self.splits_raw[0])
        self.dev = make_per_lang_ds(self.splits_raw[1])
        self.test = make_per_lang_ds(self.splits_raw[2])

    def describe(self, log=None):
        info = log.info if log else print
        info(f"train: {len(self.train)}/{len(self.train_raw)}")

        def ninst(dict_ds):
            return sum(1 for _ in flatten(dict_ds.values()))

        info(f"dev: {ninst(self.dev)}/{ninst(self.dev_raw)}")
        info(f"test: {ninst(self.test)}/{ninst(self.test_raw)}")

        for ds in (self.train, ):
            i = 3
            tokens = self.bpemb.decode_ids(ds[i]["bpe"]).split()
            tags = self.tag_enc.inverse_transform(ds[i]["tag"].cpu())
            info(" ".join(f"{tok}/{tag}" for tok, tag in zip(tokens, tags)))

    def token_texts(self, split_name):
        split = getattr(self, split_name)
        if isinstance(split, list):
            return [instance["token"] for instance in split]
        if isinstance(split, dict):
            return {
                lang: [instance["token"] for instance in val]
                for lang, val in split.items()}

    @cached_property
    def train_loader(self):
        self.assert_batch_size()
        return self.loader(self.train, batch_size=self.batch_size)

    @property
    def batch_iter_train(self):
        return iter(self.train_loader)

    def ds_dict_loader(self, ds_dict):
        self.assert_eval_batch_size()
        return {
            key: self.loader(
                ds, batch_size=self.eval_batch_size, shuffle=False)
            for key, ds in ds_dict.items()}

    @cached_property
    def dev_loader(self):
        return self.ds_dict_loader(self.dev)

    @cached_property
    def test_loader(self):
        return self.ds_dict_loader(self.test)

    @property
    def iter_dev(self):
        return iter(self.dev_loader.items())

    @property
    def iter_test(self):
        return iter(self.test_loader.items())


class UD_1_2(FixedSplitDataset):
    """A monolingual dataset in Universersal Dependencies 1.2 format.
    """
    def load_splits_raw(self):
        self.split_files = [
            (
                self.conf.data_dir / self.name / split_name / self.lang
                ).with_suffix(".conllu")
            for split_name in self.split_names]
        self.splits_raw = [
            conllu.parse("\n".join(lines(f)))
            for f in self.split_files]


class UD_1_2_Multi(MultilingualDataset):
    """A multilingual dataset in Universersal Dependencies 1.2 format.
    Comprises treebanks for the 21 high-res languages tested in
    Yasunaga et al. 2017
    """
    langs = "bg cs da de en es eu fa fi fr he hi hr id it nl no pl pt sl sv".split()  # NOQA
    fnames = [lang + ".conllu" for lang in langs]

    def load_data_raw(self):
        data_dir = self.conf.data_dir / "ud_1_2"
        self.splits_raw = [
            list(flatten(map(
                self.parse_file, [
                    data_dir / "train" / fn for fn in self.fnames]))),
            *[
                {
                    f.stem: self.parse_file(f)
                    for f in [data_dir / split / fn for fn in self.fnames]}
                for split in ("dev", "test")]
            ]
        for split_name, split_raw in zip(self.split_names, self.splits_raw):
            setattr(self, split_name + "_raw", split_raw)

    def parse_file(self, file):
        sents = split_iter(lines(file), lambda line: line == "")
        sents = islice(filter(bool, sents), self.conf.max_ninst)
        sents = map("\n".join, sents)
        return [conllu.parse(sent)[0] for sent in sents]

    def get_tags(self):
        return [
            token[self.tag] or self.NO_TAG
            for sent in self.splits_raw[0]
            for token in sent] + [
            token[self.tag] or self.NO_TAG
            for split in self.splits_raw[1:]
            for lang_sents in split.values()
            for sent in lang_sents
            for token in sent]

    def split_sents(self):
        return [
            self.splits_raw[0],
            flatten(self.splits_raw[1].values()),
            flatten(self.splits_raw[2].values())]

    def iter_forms(self):
        return (
            token["form"]
            for sent in flatten(self.split_sents())
            for token in sent)

    def get_chars(self):
        return [c for form in self.iter_forms() for c in form]


class UD_1_2_Lowres_Multi(UD_1_2_Multi):
    """A multilingual dataset in Universersal Dependencies 1.2 format.
    Comprises treebanks for the 6 low-res langs tested in Yasunaga et al. 2017
    """
    langs = "el et ga hu ro ta".split()
    fnames = [lang + ".conllu" for lang in langs]


class UD_1_2_Multi_finetune(UD_1_2):
    pass


class WikiAnn(CrossValidationDataset):
    """A monolingual subset of The WikiANN dataset by Pan et al. 2017.
    """
    def load_data_raw(self):
        self.file = self.conf.data_dir / self.name / self.lang
        self.data_raw = self.parse_file(self.file)

    def parse_file(self, file):
        sents = split_iter(lines(file), lambda l: l == "")
        sents = islice(filter(bool, sents), self.conf.max_ninst)

        def parse_sent(sent):
            parts = map_assert(
                str.split, lambda parts: len(parts) in {3, 7}, sent)
            forms, tags = zip(*map(lambda ps: (ps[0], ps[-1]), parts))
            assert len(forms) == len(tags) == len(sent)
            return [
                {"form": form, "ner": tag} for form, tag in zip(forms, tags)]

        return list(map_skip_assert_error(parse_sent, sents, verbose=True))

    def get_tags(self):
        if self.tag_scheme == "BIO":
            return ["B-LOC", "B-ORG", "B-PER", "I-LOC", "I-ORG", "I-PER", "O"]
        elif self.tag_scheme == "IOBES":
            return [
                "B-LOC", "B-ORG", "B-PER",
                "I-LOC", "I-ORG", "I-PER",
                "E-LOC", "E-ORG", "E-PER",
                "S-LOC", "S-ORG", "S-PER",
                "O"]
        else:
            raise ValueError(f"Unknown tag scheme {self.tag_scheme}")

    def get_shapes(self):
        return ["Aa", "a", ".", "0", "A", "0a0", "%"]


class WikiAnnMulti_finetune(FixedSplitDataset):
    """A monolingual subset of The WikiANN dataset by Pan et al. 2017.
    This class differs slightly from the WikiAnn class to make it
    suitable for for monolingual finetuning of a pretrained multiingual
    model.
    """

    split_names = ["train", "dev", "test"]

    def load_splits_raw(self):
        assert self.crossval_idx is not None
        data_dir = self.conf.data_dir / self.name / str(self.crossval_idx)
        self.split_files = [
            data_dir / f"{self.lang}.{split_name}"
            for split_name in self.split_names]
        ninsts = [
            self.conf.max_ninst,
            self.conf.max_eval_ninst,
            self.conf.max_eval_ninst]
        self.splits_raw = [
            self.parse_file(f, ninst)
            for f, ninst in zip(self.split_files, ninsts)]

    def parse_file(self, file, ninst):
        sents = split_iter(lines(file), lambda l: l == "")
        sents = islice(filter(bool, sents), ninst)

        def parse_sent(sent):
            parts = map_assert(
                str.split, lambda parts: len(parts) in {3, 7}, sent)
            forms, tags = zip(*map(lambda ps: (ps[0], ps[-1]), parts))
            assert len(forms) == len(tags) == len(sent)
            return [
                {"form": form, "ner": tag} for form, tag in zip(forms, tags)]

        return list(map_skip_assert_error(parse_sent, sents, verbose=True))

    def get_tags(self):
        if self.tag_scheme == "BIO":
            return ["B-LOC", "B-ORG", "B-PER", "I-LOC", "I-ORG", "I-PER", "O"]
        elif self.tag_scheme == "IOBES":
            return [
                "B-LOC", "B-ORG", "B-PER",
                "I-LOC", "I-ORG", "I-PER",
                "E-LOC", "E-ORG", "E-PER",
                "S-LOC", "S-ORG", "S-PER",
                "O"]
        else:
            raise ValueError(f"Unknown tag scheme {self.tag_scheme}")

    def get_shapes(self):
        return ["Aa", "a", ".", "0", "A", "0a0", "%"]

    def get_chars(self):
        return self.char_enc.labels.tolist()


class WikiAnnMulti(MultilingualDataset):
    """The multilingual WikiANN dataset by Pan et al. 2017.
    """

    def load_data_raw(self):
        assert self.crossval_idx is not None
        data_dir = self.conf.data_dir / self.name / str(self.crossval_idx)
        self.splits_raw = [
            list(flatten(map(self.parse_file, data_dir.glob("*.train")))),
            *[
                {
                    f.stem: self.parse_file(f)
                    for f in data_dir.glob(f"*.{split}")}
                for split in ("dev", "test")]
            ]
        for split_name, split_raw in zip(self.split_names, self.splits_raw):
            setattr(self, split_name + "_raw", split_raw)

    def parse_file(self, file):
        sents = split_iter(lines(file), lambda line: line == "")
        sents = islice(filter(bool, sents), self.conf.max_ninst)

        def parse_sent(sent):
            parts = map_assert(
                str.split, lambda parts: len(parts) in {3, 7}, sent)
            forms, tags = zip(*map(lambda ps: (ps[0], ps[-1]), parts))
            assert len(forms) == len(tags) == len(sent)
            return [
                {"form": form, "ner": tag} for form, tag in zip(forms, tags)]

        return list(map_skip_assert_error(parse_sent, sents, verbose=True))

    def get_tags(self):
        if self.tag_scheme == "BIO":
            return ["B-LOC", "B-ORG", "B-PER", "I-LOC", "I-ORG", "I-PER", "O"]
        elif self.tag_scheme == "IOBES":
            return [
                "B-LOC", "B-ORG", "B-PER",
                "I-LOC", "I-ORG", "I-PER",
                "E-LOC", "E-ORG", "E-PER",
                "S-LOC", "S-ORG", "S-PER",
                "O"]
        else:
            raise ValueError(f"Unknown tag scheme {self.tag_scheme}")


datasets = {ds.__name__.lower(): ds for ds in [
    UD_1_2,
    UD_1_2_Multi,
    UD_1_2_Multi_finetune,
    UD_1_2_Lowres_Multi,
    WikiAnn,
    WikiAnnMulti,
    WikiAnnMulti_finetune,
   ]}

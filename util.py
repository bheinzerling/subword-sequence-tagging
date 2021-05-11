import json
from pathlib import Path
import logging
import random
from collections import defaultdict
from subprocess import run, PIPE

import numpy as np

import torch
from torch import nn, optim, tensor

from boltons.iterutils import pairwise_iter as pairwise


def lines(file, max=None, skip=0, apply_func=str.strip):
    """Iterate over lines in (text) file. Optionally skip first `skip`
    lines, only read the first `max` lines, and apply `apply_func` to
    each line. By default lines are stripped, set `apply_func` to None
    to disable this."""
    from itertools import islice
    if apply_func:
        with open(str(file), encoding="utf8") as f:
            for line in islice(f, skip, max):
                yield apply_func(line)
    else:
        with open(str(file), encoding="utf8") as f:
            for line in islice(f, skip, max):
                yield line


def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item


def map_assert(map_fn, assert_fn, iterable):
    """Assert that assert_fn is True for all results of applying
    map_fn to iterable"""
    for item in map(map_fn, iterable):
        assert assert_fn(item), item
        yield item


def map_skip_assert_error(map_fn, iterable, verbose=False):
    """Same as built-in map, but skip all items in iterabe that raise
    an assertion error when map_fn is appllied"""
    errors = 0
    for i, item in enumerate(iterable):
        try:
            yield map_fn(item)
        except AssertionError:
            if verbose:
                errors += 1
    if verbose:
        total = i + 1
        print(f"Skipped {errors} / {total} AssertionErrors")


def split_lengths_for_ratios(nitems, *ratios):
    """Return the lengths of the splits obtained when splitting nitems
    by the given ratios"""
    lengths = [int(ratio * nitems) for ratio in ratios]
    i = 1
    while sum(lengths) != nitems and i < len(ratios):
        lengths[-i] += 1
        i += 1
    assert sum(lengths) == nitems, f'{sum(lengths)} != {nitems}\n{ratios}'
    return lengths


def split_idxs_for_ratios(nitems, *ratios, end_inclusive=False):
    assert len(ratios) >= 1
    assert all(0 < ratio < 1 for ratio in ratios)
    assert sum(ratios) <= 1.0
    idxs = list(np.cumsum(split_lengths_for_ratios(nitems, *ratios)))
    if end_inclusive:
        idxs = [0] + idxs
        if idxs[-1] != nitems:
            idxs.append(nitems)
    return idxs


def split_by_ratios(items, *ratios):
    nitems = len(items)
    split_idxs = split_idxs_for_ratios(nitems, *ratios, end_inclusive=True)
    return [
        items[split_idxs[i]:split_idxs[i+1]]
        for i in range(len(split_idxs) - 1)]


def get_formatter(fmt=None, datefmt=None):
    if not fmt:
        fmt = '%(asctime)s| %(message)s'
    if not datefmt:
        datefmt = "%Y-%m-%d %H:%M:%S"
    return logging.Formatter(fmt, datefmt=datefmt)


def get_logger(file=None, fmt=None, datefmt=None):
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    formatter = get_formatter(fmt, datefmt)
    if not logging.root.handlers:
        logging.root.addHandler(logging.StreamHandler())
    logging.root.handlers[0].formatter = formatter
    return log


def to_path(maybe_str):
    if isinstance(maybe_str, str):
        return Path(maybe_str)
    return maybe_str


def json_load(json_file):
    """Load object from json file."""
    with to_path(json_file).open(encoding="utf8") as f:
        return json.load(f)


def mkdir(dir, parents=True, exist_ok=True):
    """Convenience function for Path.mkdir"""
    dir = to_path(dir)
    dir.mkdir(parents=parents, exist_ok=exist_ok)
    return dir


def get_and_increment_runid(file=Path("runid")):
    """Get the next run id by incrementing the id stored in a file.
    (faster than taking the maximum over all subdirs)"""
    attempts = 0
    runid = None
    try:
        try:
            from filelock import FileLock
            lockfile = file.parent / (file.name + '.lock')
            lock = FileLock(lockfile, timeout=3)
        except ImportError:
            import contextlib
            lock = contextlib.nullcontext()
        while runid is None:
            try:
                with lock:
                    with file.open() as f:
                        runid = int(f.read()) + 1
            except ValueError as e:
                if attempts < 3:
                    print('failed to read runid from file', file)
                    attempts += 1
                else:
                    raise e
    except FileNotFoundError:
        runid = 0
    with file.open("w") as out:
        out.write(str(runid))
    return runid


def next_rundir(basedir=Path("out"), runid_fname="runid", log=None):
    """Create a directory for running an experiment."""
    mkdir(basedir)
    runid = get_and_increment_runid(basedir / runid_fname)
    rundir = mkdir(basedir / str(runid))
    if log:
        log.info(f"rundir: {rundir.resolve()}")
    return rundir


def dump_args(args, file):
    """Write argparse args to file."""
    def _maybe_to_str(v):
        try:
            json.dumps(v)
        except TypeError:
            return str(v)
        return v

    with to_path(file).open("w", encoding="utf8") as out:
        json.dump({
            k: _maybe_to_str(v)
            for k, v in args.__dict__.items()}, out, indent=4)


def add_embeddings(keyed_vectors, *words, init=None):
    from gensim.models.keyedvectors import Vocab
    if init is None:
        init = np.zeros
    syn0 = keyed_vectors.syn0
    for word in words:
        keyed_vectors.key_to_index[word] = Vocab(count=0, index=syn0.shape[0])
        keyed_vectors.syn0 = np.concatenate([syn0, init((1, syn0.shape[1]))])
        keyed_vectors.index2word.append(word)
    return syn0.shape[0]


def add_unk_embedding(keyed_vectors, unk_str="<unk>", init=None):
    """Add a vocab entry and embedding for unknown words to keyed_vectors."""
    add_embeddings(keyed_vectors, unk_str, init)


def load_word2vec_file(
        word2vec_file,
        weights_file=None, normalize=False, add_unk=False, unk="<unk>",
        add_pad=False, pad="<pad>"):
    """Load a word2vec file in either text or bin format, optionally
    supplying custom embedding weights and normalizing embeddings."""
    from gensim.models import KeyedVectors
    word2vec_file = str(word2vec_file)
    binary = word2vec_file.endswith(".bin")
    vecs = KeyedVectors.load_word2vec_format(word2vec_file, binary=binary)
    if add_unk:
        if unk not in vecs:
            add_unk_embedding(vecs)
        else:
            pass
            # raise ValueError("Attempted to add <unk>, but already present")
    if add_pad:
        if pad not in vecs:
            add_embeddings(vecs, pad)
        else:
            raise ValueError("Attempted to add <pad>, but already present")
    if weights_file:
        import torch
        weights = torch.load(weights_file)
        vecs.syn0 = weights.cpu().float().numpy()
    if normalize:
        log.info("normalizing %s", word2vec_file)
        vecs.init_sims(replace=True)
    return vecs


def get_optim(
        conf, model, optimum='max', n_train_instances=None,
        additional_params_dict=None):
    """Create an optimizer according to command line args."""
    params = [p for p in model.parameters() if p.requires_grad]
    optim_name = conf.optim.lower()
    lr = getattr(conf, 'learning_rate', None) or conf.lr
    betas = getattr(conf, 'adam_betas', [0.9, 0.999])
    eps = getattr(conf, 'adam_eps', 1e-8)
    weight_decay = getattr(conf, 'weight_decay', 0.0)
    if optim_name == "adam":
        return optim.Adam(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optim_name == "adamw":
        from transformers import AdamW
        no_decay = ['bias', 'LayerNorm.weight']
        additional_params = (
            set(additional_params_dict['params'])
            if additional_params_dict
            else {})
        grouped_params = [
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and p not in additional_params],
                'weight_decay': conf.weight_decay},
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and p not in additional_params],
                'weight_decay': 0.0}]
        if additional_params_dict:
            grouped_params.append(additional_params_dict)
        return AdamW(
            grouped_params,
            betas=betas,
            weight_decay=weight_decay,
            lr=lr,
            eps=eps)
    elif optim_name == "sgd":
        return optim.SGD(
            params,
            lr=conf.lr,
            momentum=conf.momentum,
            weight_decay=conf.weight_decay)
    elif optim_name == 'radam':
        from .radam import RAdam
        return RAdam(params, lr=lr)
    raise ValueError("Unknown optimizer: " + conf.optim)


def emb_layer(
        vecs, trainable=False, use_weights=True, dtype=torch.float32,
        device='cuda',
        **kwargs):
    """Create an Embedding layer from a numpy array."""
    emb_weights = tensor(vecs, dtype=dtype).to(device=device)
    emb = nn.Embedding(*emb_weights.shape, **kwargs)
    if use_weights:
        emb.weight = nn.Parameter(emb_weights)
    emb.weight.requires_grad = trainable
    return emb


class Score():
    """Keep track of a score computed by score_func, save model
    if score improves.
    """
    def __init__(
            self, name, score_func=None, shuffle_baseline=False,
            comp=float.__gt__, save_model=True, log=None,
            add_mode="extend"):
        self.name = name
        if comp == float.__lt__:
            self.current = float("inf")
            self.best = float("inf")
            self.optimum = "min"
        else:
            self.current = 0.0
            self.best = 0.0
            self.optimum = "max"
        self.best_model = None
        self.pred = []
        self.true = []
        self.shuffle = []
        self.score_func = score_func or self.accuracy
        self.shuffle_baseline = shuffle_baseline
        self.comp = comp
        self.save_model = save_model
        self.info = log.info if log else print
        if add_mode == "extend":
            self.add = self.extend
        elif add_mode == "append":
            self.add = self.append
        else:
            raise ValueError("Unknown add_mode: " + add_mode)

    def extend(self, pred, true=None):
        """extend predicted and true labels"""
        if hasattr(pred, "tolist"):
            pred = pred.tolist()
        self.pred.extend(pred)
        if true is not None:
            if hasattr(pred, "shape"):
                assert pred.shape == true.shape, (pred.shape, true.shape)
            else:
                assert len(pred) == len(true)
            if hasattr(true, "tolist"):
                true = true.tolist()
            self.true.extend(true)

    def append(self, pred, true=None):
        """append predicted and true labels"""
        if hasattr(pred, "tolist"):
            pred = pred.tolist()
        self.pred.append(pred)
        if true is not None:
            if hasattr(pred, "shape"):
                assert pred.shape == true.shape, (pred.shape, true.shape)
            else:
                assert len(pred) == len(true)
            if hasattr(true, "tolist"):
                true = true.tolist()
            self.true.append(true)

    def update(self, model=None, rundir=None, epoch=None, score=None):
        if score is None:
            if not self.true:
                score = self.score_func(self.pred)
            else:
                score = self.score_func(self.pred, self.true)
        self.current = score
        if self.comp(score, self.best):
            self.best = score
            if self.save_model and model:
                assert rundir
                epoch_str = f"e{epoch}_" if epoch is not None else ""
                fname = f"{epoch_str}{self.name}_{score:.4f}_model.pt"
                model_file = rundir / fname
                save_model(model, model_file)
                self.best_model = model_file
        if self.shuffle_baseline:
            random.shuffle(self.pred)
            shuffle_score = self.score_func(self.pred, self.true)
        else:
            shuffle_score = None
        self.true = []
        self.pred = []
        return score, shuffle_score

    def update_log(
            self, model=None, rundir=None, epoch=None, score=None):
        score, shuffle_score = self.update(
            model=model, rundir=rundir, epoch=epoch, score=score)
        self.info(f"score {self.name}_{score:.4f}/{self.best:.4f}")
        if self.best_model:
            self.info(str(self.best_model))
        if shuffle_score is not None:
            self.info(f"\nshuffle {self.name}_{shuffle_score:.4f}")
        return score

    @staticmethod
    def accuracy(pred, true):
        n = len(pred)
        assert n != 0
        assert n == len(true)
        correct = sum(p == t for p, t in zip(pred, true))
        return correct / n

    @staticmethod
    def f1_score(pred, true):
        import sklearn
        return sklearn.metrics.f1_score(true, pred)

    @staticmethod
    def f1_score_multiclass(pred, true, average='macro'):
        import sklearn
        f1_score = sklearn.metrics.f1_score
        if average == 'macro':
            return np.average([f1_score(t, p) for t, p in zip(true, pred)])
        elif average == 'micro':
            return f1_score(list(flatten(true)), list(flatten(pred)))

    @staticmethod
    def f1_score_multiclass_micro(pred, true):
        return Score.f1_score_multiclass(pred, true, average='micro')

    @staticmethod
    def f1_score_multiclass_macro(pred, true):
        return Score.f1_score_multiclass(pred, true, average='macro')

    @property
    def best_str(self):
        return f"{self.name}_{self.best:.4f}"

    @property
    def current_str(self):
        return f"{self.name}_{self.current_score:.4f}"


class LossTracker(list):
    """Keep track of losses, save model if loss improves."""
    def __init__(self, name, save_model=True, log=None):
        self.name = name
        self.best_loss = defaultdict(lambda: float("inf"))
        self.best_model = None
        self.save_model = save_model
        self.info = log.info if log else print

    def interval_end(
            self, epoch=None, model=None, model_file=None, ds_name=None):
        loss = self.current
        if loss < self.best_loss[ds_name]:
            self.best_loss[ds_name] = loss
            if self.save_model and model:
                model_file = Path(str(model_file).format(
                    epoch=epoch,
                    ds_name=ds_name,
                    loss=loss))
                save_model(model, model_file)
                self.best_model = model_file
        self.clear()
        return loss

    @property
    def current(self):
        return np.average(self)


class LossTrackers():
    """Keep track of multiple losses."""
    def __init__(self, *loss_trackers, log=None):
        self.loss_trackers = loss_trackers
        self.info = log.info if log else print

    def append(self, *losses):
        for lt, loss in zip(self.loss_trackers, losses):
            try:
                loss = loss.item()
            except AttributeError:
                pass
            lt.append(loss)

    def interval_end(
            self, *, epoch=None, model=None, model_file=None, ds_name=None):
        for lt in self.loss_trackers:
            yield (
                lt.name,
                lt.interval_end(
                    epoch=epoch,
                    model=model, model_file=model_file, ds_name=ds_name),
                lt.best_loss[ds_name])

    def interval_end_log(
            self, epoch, *, model=None, model_file=None, ds_name=None):
        self.info(f"e{epoch} {ds_name} " + " ".join(
            f"{name}_{loss:.6f}/{best:.6f}"
            for name, loss, best in self.interval_end(
                epoch=epoch,
                model=model, model_file=model_file, ds_name=ds_name)))

    def best_log(self):
        self.info("best: " + " ".join(
            f"{lt.name}_{lt.best_loss:.6f}" for lt in self.loss_trackers))

    @staticmethod
    def from_names(*names, **kwargs):
        loss_trackers = map(lambda name: LossTracker(name, **kwargs), names)
        return LossTrackers(*loss_trackers, log=kwargs.get("log"))

    def __iter__(self):
        return iter(self.loss_trackers)

    def __getitem__(self, i):
        return self.loss_trackers[i]


# source: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping():
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, model_file, log=None):
    """Save a pytorch model to model_file."""
    if isinstance(model_file, str):
        model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("wb") as out:
        torch.save(model.state_dict(), out)
    if log:
        log.info("saved %s", model_file)


CONLLEVAL = str(Path("scripts/conlleval").absolute())


def conll_ner(sents, pred, true, tag_enc=None, outfile=None):
    if tag_enc is not None:
        pred = tag_enc.inverse_transform(pred)
        true = tag_enc.inverse_transform(true)
    token_lines = list(map(" ".join, zip(flatten(sents), true, pred)))
    sent_offsets = np.cumsum([0] + list(map(len, sents)))
    sent_lines = "\n\n".join(map(
        lambda p: "\n".join(token_lines[slice(*p)]), pairwise(sent_offsets)))
    if outfile:
        with outfile.open("w", encoding="utf8") as out:
            out.write(sent_lines)
    eval_out, eval_parsed = run_conll_eval(sent_lines)
    print(eval_out)
    return eval_parsed


def run_conll_eval(eval_in):
    out = run(CONLLEVAL, input=eval_in, encoding="utf8", stdout=PIPE).stdout
    return out, parse_conlleval(out)


def parse_conlleval(output):
    lines = output.replace("%", "").replace(" ", "").split("\n")[1:-1]
    return dict(map(
        lambda kv: (kv[0], float(kv[1])),
        map(lambda s: s.split(":"), lines[0].split(";"))))


class ConllScore():
    def __init__(self, tag_enc=None):
        self.sentences = []
        self.outfile = None
        self.tag_enc = tag_enc

    def __call__(self, pred, true):
        assert len(self.sentences) == len(pred) == len(true)
        for s, p, t in zip(self.sentences, pred, true):
            assert len(s) == len(p) == len(t)
        result = conll_ner(
            self.sentences, list(flatten(pred)), list(flatten(true)),
            tag_enc=self.tag_enc, outfile=self.outfile)
        return result["FB1"]


def token_shapes(tokens, collapse=True):
    """Returns strings which encode the shape of tokens. If collapse
    is set, repeats are collapsed and infrequent shapes encoded as "other":
        Aa  | capitalized
        a   | all lowercase
        .   | all punctuation
        0   | all digits
        A   | all UPPERCASE
        0a0 | digits - lower - digits
        %   | other
    """
    collapsed_shapes = {"Aa", "a", ".", "0", "A", "0a0"}

    def char_shape(char):
        if not char.isalnum():
            return "."
        if char.isdigit():
            return "0"
        if char.isupper():
            return "A"
        return "a"

    shapes = [[char_shape(c) for c in token] for token in tokens]
    if collapse:
        def _collapse(chars):
            last = None
            for c in chars:
                if c != last:
                    yield c
                    last = c
    else:
        def _collapse(chars):
            return chars
    shapes = ["".join(_collapse(shape)) for shape in shapes]
    if collapse:
        return [s if s in collapsed_shapes else "%" for s in shapes]
    return shapes


def to_word_indexes(tokens, keyed_vectors, unk=None, fallback_transform=None):
    """Look up embedding indexes for tokens."""
    if fallback_transform:
        assert unk
        unk = keyed_vectors.key_to_index[unk]
        return [
            keyed_vectors.key_to_index.get(
                token,
                keyed_vectors.key_to_index.get(fallback_transform(token), unk)).index
            for token in tokens]
    if unk is None:
        return [keyed_vectors.key_to_index[token].index for token in tokens]
    unk = keyed_vectors.key_to_index[unk]
    return [keyed_vectors.key_to_index.get(token, unk).index for token in tokens]


# https://github.com/glample/tagger/blob/master/utils.py
def ensure_iob2(tags):
    """Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    tags = list(tags)
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return tags


# https://github.com/zalandoresearch/flair/blob/master/flair/data.py
def iob_iobes(tags):
    """IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                            tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                            tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


iob_to = {
    "BIO": ensure_iob2,
    "IOBES": iob_iobes}


class DictLabelEncoder():
    def fit(self, labels):
        self.label2idx = {label: idx for idx, label in enumerate(labels)}
        self.idx2label = self.classes_ = labels
        return self

    def transform(self, labels):
        return np.array(list(map(self.label2idx.__getitem__, labels)))

    def inverse_transform(self, idxs):
        return list(map(self.idx2label.__getitem__, idxs))


class LabelEncoder(object):
    """Encodes and decodes labels. Decoding from idx representation.
    Optionally return pytorch tensors instead of numpy arrays.
    Use backend 'sklearn' (default) for speed or backend 'dict' if the
    order of labels should be preserved (i.e. first label will be idx 0...)."""
    def __init__(self, to_torch=False, device="cuda", backend='sklearn'):
        self.to_torch = to_torch
        self.device = device
        if backend == 'sklearn':
            from sklearn.preprocessing import LabelEncoder as _LabelEncoder
            self._LabelEncoder = _LabelEncoder
        elif backend == 'dict':
            self._LabelEncoder = DictLabelEncoder
        else:
            raise ValueError('unknown backend:', backend)

    def fit(self, labels, min_count=0, unk_label=None):
        self.unk_label = unk_label
        if min_count > 0:
            from collections import Counter
            counts = Counter(labels)
            labels = [
                label for label in set(labels)
                if counts[label] > min_count]
            assert unk_label is not None
            labels.append(unk_label)
            self.label_set = set(labels)
        self.label_enc = self._LabelEncoder().fit(labels)
        self.labels = self.label_enc.classes_
        self.nlabels = len(self.labels)
        idxs = list(range(self.nlabels))
        self.idx2label = dict(zip(idxs, self.inverse_transform(idxs)))
        return self

    def __len__(self):
        return self.nlabels

    def transform(self, labels):
        if isinstance(labels, str):
            labels = [labels]
        if labels and isinstance(labels[0], list):
            return [self.transform(l) for l in labels]
        if self.unk_label is not None:
            labels = [
                label if label in self.label_set else self.unk_label
                for label in labels]
        if self.to_torch:
            if not labels:
                return torch.tensor([]).to(
                    dtype=torch.int64, device=self.device)
            tensors = []
            bs = 1000000
            for i in range(0, len(labels), bs):
                labels_enc = self.label_enc.transform(labels[i:i+bs])
                tensors.append(torch.LongTensor(labels_enc))
            return torch.cat(tensors).to(device=self.device)
        else:
            return self.label_enc.transform(labels)

    def inverse_transform(self, idx, ignore_idx=None):
        if ignore_idx is not None:
            def filter_idxs(idxs):
                return [i for i in idxs if i != ignore_idx]
        else:
            def filter_idxs(idxs):
                return idxs
        try:
            idx = idx.tolist()
        except AttributeError:
            pass
        try:
            if isinstance(idx[0], list):
                return [
                    self.label_enc.inverse_transform(
                        filter_idxs(_idx)).tolist()
                    for _idx in idx]
        except TypeError:
            return self.label_enc.inverse_transform([idx])[0]
        return self.label_enc.inverse_transform(filter_idxs(idx))

    @staticmethod
    def from_file(
            file,
            additional_labels=None,
            to_torch=False,
            save_to=None,
            device="cuda"):
        """Create LabelEncoder instance from file, which contains
        one label per line. Optionally dump instance to save_to."""
        from .io import lines
        codec = LabelEncoder(to_torch, device=device)
        if additional_labels is None:
            additional_labels = []
        codec.fit(list(lines(file)) + additional_labels)
        if save_to:
            import joblib
            joblib.dump(codec, save_to)
        return codec


class _Missing(object):

    def __repr__(self):
        return 'no value'

    def __reduce__(self):
        return '_missing'


_missing = _Missing()


class _cached_property(property):
    """A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value::
        class Foo(object):
            @cached_property
            def foo(self):
                # calculate something important here
                return 42
    The class has to have a `__dict__` in order for this property to
    work.
    """
    # source: https://github.com/pallets/werkzeug/blob/master/werkzeug/utils.py

    # implementation detail: A subclass of python's builtin property
    # decorator, we override __get__ to check for a cached value. If one
    # choses to invoke __get__ by hand the property will still work as
    # expected because the lookup logic is replicated in __get__ for
    # manual invocation.

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __set__(self, obj, value):
        obj.__dict__[self.__name__] = value

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


def cached_property(func=None, **kwargs):
    # https://stackoverflow.com/questions/7492068/python-class-decorator-arguments
    if func:
        return _cached_property(func)
    else:
        def wrapper(func):
            return _cached_property(func, **kwargs)

        return wrapper

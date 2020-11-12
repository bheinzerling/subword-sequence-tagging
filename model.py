import torch
from torch import (
    nn,
    cat,
    zeros,
    cumsum,
    )
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
    )
from torch.nn.functional import cross_entropy


class SequenceTagger(nn.Module):

    def __init__(
            self,
            emb,
            ntags,
            args,
            *,
            nchars=None,
            nshapes=None,
            fasttext_emb=None,
            bert=None,
            tag_enc=None,
            ):
        super().__init__()
        self.many_emb = False
        if isinstance(emb, list):
            # all embeddings need to have the same dim
            assert len(set(e.weight.size(1) for e in emb)) == 1
            self.embs = nn.ModuleList(emb)
            emb_dim = self.embs[0].weight.size(1)
            self.many_emb = True
        else:
            self.emb = emb
            self.emb.weight.requires_grad = not args.emb_fixed
            emb_dim = emb.weight.size(1)
        self.ntags = ntags
        self.nshapes = nshapes
        self.repr_dim = 0
        self.dropout = LockedDropout(args.dropout)
        self.emb_dropout = self.repr_dropout = self.dropout
        self.use_char = args.use_char
        self.use_shape = args.use_shape
        self.use_bpe = args.use_bpe
        self.use_fasttext = args.use_fasttext
        self.use_meta_rnn = args.use_meta_rnn
        self.relearn_repr = args.relearn_repr
        if self.use_char:
            if isinstance(nchars, list):
                self.many_emb = True
                self.char_embs = nn.ModuleList([
                    nn.Embedding(n, args.char_emb_dim) for n in nchars])
                if args.char_emb_fixed:
                    raise NotImplementedError
            else:
                self.char_emb = nn.Embedding(nchars, args.char_emb_dim)
                self.char_emb.weight.requires_grad = not args.char_emb_fixed
        if self.use_char:
            self.char_rnn = getattr(nn, args.rnn_type)(
                args.char_emb_dim, hidden_size=args.char_nhidden,
                num_layers=args.nlayers,
                dropout=args.rnn_dropout if args.nlayers > 1 else 0.0,
                bidirectional=True, batch_first=True)
            self.repr_dim += 2 * args.char_nhidden
        if self.use_shape:
            self.shape_emb = nn.Embedding(nshapes, args.shape_emb_dim)
            self.shape_emb.weight.requires_grad = not args.shape_emb_fixed
            self.repr_dim += args.shape_emb_dim
        if self.use_bpe:
            self.bpe_rnn = getattr(nn, args.rnn_type)(
                emb_dim, hidden_size=args.bpe_nhidden, num_layers=args.nlayers,
                dropout=args.rnn_dropout if args.nlayers > 1 else 0.0,
                bidirectional=True, batch_first=True)
            self.repr_dim += 2 * args.bpe_nhidden
        if self.use_fasttext:
            assert fasttext_emb is not None
            self.fasttext_emb = fasttext_emb
            self.repr_dim += self.fasttext_emb.weight.size(1)
        if bert is not None:
            self.bert_model = bert.model
            self.use_bert = True
            self.repr_dim += bert.dim
        else:
            self.use_bert = False
        token_in_dim = args.relearn_dim if args.relearn_dim else self.repr_dim
        if self.relearn_repr:
            self.relearn = nn.Linear(self.repr_dim, token_in_dim)
        if self.use_meta_rnn:
            self.meta_rnn = getattr(nn, args.rnn_type)(
                token_in_dim, hidden_size=args.meta_nhidden,
                num_layers=args.nlayers,
                dropout=args.rnn_dropout if args.nlayers > 1 else 0.0,
                bidirectional=True, batch_first=True)
            self.out = nn.Linear(2 * args.meta_nhidden, ntags)
        else:
            self.out = nn.Linear(token_in_dim, ntags)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.crit = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, batch, tag_true=None, lang_idx=0):
        token_shape, token_len, token_sort_idx, _, word = batch["token"]
        token_shape, token_len, token_sort_idx, _, fasttext = batch["fasttext"]
        reprs = []
        if self.use_bpe:
            if self.many_emb:
                emb = self.embs[lang_idx]
            else:
                emb = self.emb
        if self.use_bpe:
            bpe_repr = self.encode_subsegments(
                batch["bpe"], emb, self.bpe_rnn, token_sort_idx)
            reprs.append(bpe_repr)
        if self.use_char:
            if self.many_emb:
                char_emb = self.char_embs[lang_idx]
            else:
                char_emb = self.char_emb
        if self.use_char:
            char_repr = self.encode_subsegments(
                batch["char"], char_emb, self.char_rnn, token_sort_idx)
            reprs.append(char_repr)
        if self.use_shape:
            token_shape_emb = self.emb_dropout(self.shape_emb(token_shape))
            reprs.append(token_shape_emb)
        if self.use_fasttext:
            fasttext_emb = self.fasttext_emb(fasttext)  # no dropout here
            reprs.append(fasttext_emb)
        if self.use_bert:
            bert_ids, bert_mask, bert_token_starts = batch["bert"]

            max_length = (bert_mask != 0).max(0)[0].nonzero()[-1].item()
            if max_length < bert_ids.shape[1]:
                bert_ids = bert_ids[:, :max_length]
                bert_mask = bert_mask[:, :max_length]

            segment_ids = torch.zeros_like(bert_mask)
            bert_repr = self.bert_model(bert_ids, segment_ids)[0]
            bert_token_reprs = [
                layer[starts.nonzero().squeeze(1)]
                for layer, starts in zip(bert_repr, bert_token_starts)]
            padded_bert_token_reprs = pad_sequence(
                bert_token_reprs, batch_first=True, padding_value=-1)
            reprs.append(padded_bert_token_reprs.float())

        token_repr = self.repr_dropout(torch.cat(reprs, dim=2))
        if self.relearn_repr:
            token_repr = self.relearn(token_repr)
        if self.use_meta_rnn:
            # .cpu() because of https://github.com/pytorch/pytorch/issues/43227
            token_repr_pack = pack_padded_sequence(
                token_repr, token_len.cpu(), batch_first=True)
            rnn_out_pack, rnn_hid = self.meta_rnn(token_repr_pack)
            rnn_out, seq_len = pad_packed_sequence(
                rnn_out_pack, batch_first=True)
            token_repr = rnn_out

        token_logit = self.out(self.repr_dropout(token_repr))
        mask = tag_true != -1
        # CrossEntropyLoss input shape: (batch_size, n_classes, seq_len)
        # for some reason, this is not equivalent to the loop below
        # loss = self.crit(token_logit.transpose(1, 2), tag_true)
        loss = 0
        for _token_logit, _tag_true, l in zip(
                token_logit, tag_true, token_len):
            loss += cross_entropy(_token_logit, _tag_true, ignore_index=-1)
        tag_pred = self.softmax(token_logit).max(dim=2)[1]
        loss /= mask.float().sum()
        return tag_pred, loss

    def encode_subsegments(
            self, subsegment_batch, sub_emb_layer, sub_rnn, token_sort_idx):
        sub, sub_len, sub_sort_idx, token_len = subsegment_batch
        sub_unsort_idx = torch.sort(sub_sort_idx)[1]
        sub_emb = self.emb_dropout(sub_emb_layer(sub))
        # .cpu() because of https://github.com/pytorch/pytorch/issues/43227
        sub_emb_pack = pack_padded_sequence(sub_emb, sub_len.cpu(), batch_first=True)
        rnn_out_pack, rnn_hid = sub_rnn(sub_emb_pack)
        rnn_out, seq_len = pad_packed_sequence(rnn_out_pack, batch_first=True)

        batch_idx = torch.arange(len(sub_len)).to(sub_len)
        rnn_state_idx = sub_len - 1
        sub_repr = rnn_out[batch_idx, rnn_state_idx]
        # re-sort to [s0t0, s0t1, s0t2, ..., s1t0, s1t2, ...]
        _sub_repr_resorted = sub_repr[sub_unsort_idx]
        # collect sentences, sorted by token length
        sent_offsets = cat([zeros(1).to(token_len), cumsum(token_len, dim=0)])
        sub_repr_resorted = [
            _sub_repr_resorted[sent_offsets[i]:sent_offsets[i + 1]]
            for i in token_sort_idx]
        sub_repr_pad = pad_sequence(sub_repr_resorted, batch_first=True)
        return sub_repr_pad


class LockedDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(
            1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(
            m, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)
        return mask * x

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np


class Beam(object):
    ''' Beam search '''

    def __init__(self, size, pad_idx, sos_idx, eos_idx, device=False):

        self.size = size
        self._done = False
        self.sos_idx = sos_idx
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), self.pad_idx, dtype=torch.long, device=device)]
        self.next_ys[0][0] = self.sos_idx
        self.finished = [False for _ in range(size)]

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)
        
        for i in range(self.size):
            if self.finished[i]:
                word_prob[i, :].fill_(-1000)
                word_prob[i, self.pad_idx].fill_(0)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        #best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        #if self.next_ys[-1][0].item() == Constants.EOS:
        #    self._done = True
        #    self.all_scores.append(self.scores)
        self.finished = []
        for i in range(self.size):
            self.finished.append(self.next_ys[-1][i].item() in [self.eos_idx, self.pad_idx])
            
        if all(self.finished):
            self._done = True
        #self._done = self.finished[0]
           
        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.sos_idx] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))

class KnowledgeEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(KnowledgeEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.entity_embeddings = nn.Embedding(config.max_entity_embeddings, config.hidden_size)
        self.triple_embeddings = nn.Embedding(config.max_triple_embeddings, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, entity_ids, triple_ids, position_ids):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        inputs_embeds = self.word_embeddings(input_ids)
        entity_embeddings = self.entity_embeddings(entity_ids)
        triple_embeddings = self.triple_embeddings(triple_ids)
        position_embeddings = self.position_embeddings(triple_ids)

        embeddings = inputs_embeds + entity_embeddings + triple_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=1024):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

def get_non_pad_mask(seq, pad_idx):
    assert seq.dim() == 2
    return seq.ne(pad_idx).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


class TransformerDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, config, n_head, n_layers):

        super(TransformerDecoder, self).__init__()
        d_k = config.hidden_size // n_head
        d_v = config.hidden_size // n_head
        d_inner = config.hidden_size * 4
        self.pad_idx = config.pad_token_id
        self.sos_idx = config.sos_token_id
        self.eos_idx = config.eos_token_id

        self.embedding = KnowledgeEmbeddings(config)        

        self.post_word_emb = PositionalEmbedding(d_model=config.hidden_size, max_len=1024)
        
        self.enc_layer_stack = nn.ModuleList([
            EncoderLayer(config.hidden_size, d_inner, n_head, d_k, d_v, dropout=config.hidden_dropout_prob)
            for _ in range(n_layers)])

        self.dec_layer_stack = nn.ModuleList([
            DecoderLayer(config.hidden_size, d_inner, n_head, d_k, d_v, dropout=config.hidden_dropout_prob)
            for _ in range(n_layers)])

        if config.untie_embedding:
            self.proj = nn.Linear(config.hidden_size, config.vocab_size)
        else:
            self.proj = None

    def last_layer(self, dec_output):
        if self.proj is not None:
            logits = self.proj(dec_output)
        else:
            logits = torch.matmul(dec_output, self.embedding.word_embeddings.weight.transpose(1, 0))
        return logits

    def forward(self, input_ids, entity_ids, triple_ids, position_ids, tgt_seq):
        enc_inp = self.embedding(input_ids, entity_ids, triple_ids, position_ids)
        
        # -- Encode source
        non_pad_mask = get_non_pad_mask(input_ids, self.pad_idx)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=input_ids, pad_idx=self.pad_idx)        
        for layer in self.enc_layer_stack:
            enc_inp, _ = layer(enc_inp, non_pad_mask, slf_attn_mask)
        enc_output = enc_inp

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq, self.pad_idx)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq, pad_idx=self.pad_idx)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=tgt_seq, pad_idx=self.pad_idx)

        dec_output = self.embedding.word_embeddings(tgt_seq) + self.post_word_emb(tgt_seq)

        for layer in self.dec_layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

        return self.last_layer(dec_output)

    def greedy_decode(self, input_ids, entity_ids, triple_ids, position_ids, banwords=[], max_token_seq_len=30):
        enc_inp = self.embedding(input_ids, entity_ids, triple_ids, position_ids)
        
        # -- Encode source
        non_pad_mask = get_non_pad_mask(input_ids, self.pad_idx)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=input_ids, pad_idx=self.pad_idx)        
        for layer in self.enc_layer_stack:
            enc_inp, _ = layer(enc_inp, non_pad_mask, slf_attn_mask)
        enc_output = enc_inp

        batch_size = input_ids.shape[0]
        tgt_seq = torch.LongTensor(batch_size, 1).fill_(self.sos_idx).to(input_ids.device)

        for step in range(max_token_seq_len):
            non_pad_mask = get_non_pad_mask(tgt_seq, self.pad_idx)

            slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq, pad_idx=self.pad_idx)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
            dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=tgt_seq, pad_idx=self.pad_idx)

            dec_output = self.embedding.word_embeddings(tgt_seq) + self.post_word_emb(tgt_seq)

            for layer in self.dec_layer_stack:
                dec_output, dec_slf_attn, dec_enc_attn = layer(
                    dec_output, enc_output,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask,
                    dec_enc_attn_mask=dec_enc_attn_mask)

            logits = self.last_layer(dec_output)[:, -1, :]
            if step <= 5:
                logits[:, banwords] = -np.inf

            decoded = torch.argmax(logits, -1).unsqueeze(-1)
            tgt_seq = torch.cat([tgt_seq, decoded], -1)

        return tgt_seq[:, 1:]

    def beam_search(self, inputs, n_bm, max_token_seq_len=30, banwords=[]):
        ''' Translation work in one batch '''
        with torch.no_grad():
            #-- Repeat data for beam search
            n_inst, len_s = inputs[0].size()
            
            inputs = [_.repeat(1, n_bm).view(n_inst * n_bm, len_s) for _ in inputs]

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, device=inputs[0].device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, max_token_seq_len + 1):
                active_inst_idx_list = beam_decode_step(self, inst_dec_beams, len_dec_seq, active_inst_idx_list,
                                                        inputs, inst_idx_to_position_map, n_bm, banwords)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                inputs, inst_idx_to_position_map = collate_active_info(inputs, inst_idx_to_position_map, active_inst_idx_list, n_bm)
        
            batch_hyp, batch_scores = collect_hypothesis_and_scores(tokenizer.pad_token_id, inst_dec_beams, n_bm)
            
            result = []
            for _ in batch_hyp:
                finished = False
                for r in _:
                    if len(r) >= 8:
                        result.append(r)
                        finished = True
                        break
                if not finished:
                    result.append(_[0])
            
            return result

class GatedTransformerDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, config, n_head, n_layers):

        super(GatedTransformerDecoder, self).__init__()
        d_k = config.hidden_size // n_head
        d_v = config.hidden_size // n_head
        d_inner = config.hidden_size * 4
        self.pad_idx = config.pad_token_id
        self.sos_idx = config.sos_token_id
        self.eos_idx = config.eos_token_id

        self.embedding = KnowledgeEmbeddings(config)        

        self.post_word_emb = PositionalEmbedding(d_model=config.hidden_size, max_len=1024)
        
        self.enc_layer_stack = nn.ModuleList([
            EncoderLayer(config.hidden_size, d_inner, n_head, d_k, d_v, dropout=config.hidden_dropout_prob)
            for _ in range(n_layers)])

        self.dec_layer_stack = nn.ModuleList([
            DecoderLayer(config.hidden_size, d_inner, n_head, d_k, d_v, dropout=config.hidden_dropout_prob)
            for _ in range(n_layers)])

        self.gate = nn.Linear(config.hidden_size, 1)

        self.proj = nn.Linear(config.hidden_size, config.vocab_size)


    def forward(self, input_ids, entity_ids, triple_ids, position_ids, tgt_seq,  positionwise_copy_prob=False):
        enc_inp = self.embedding(input_ids, entity_ids, triple_ids, position_ids)
        
        # -- Encode source
        non_pad_mask = get_non_pad_mask(input_ids, self.pad_idx)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=input_ids, pad_idx=self.pad_idx)        
        for layer in self.enc_layer_stack:
            enc_inp, _ = layer(enc_inp, non_pad_mask, slf_attn_mask)
        enc_output = enc_inp

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq, self.pad_idx)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq, pad_idx=self.pad_idx)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=tgt_seq, pad_idx=self.pad_idx)

        dec_output = self.embedding.word_embeddings(tgt_seq) + self.post_word_emb(tgt_seq)

        for layer in self.dec_layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

        copy_gate = torch.sigmoid(self.gate(dec_output))

        in_vocab_prob = torch.softmax(self.proj(dec_output), -1)
        full_vocab_prob = (1 - copy_gate) * in_vocab_prob

        scores = torch.bmm(dec_output, enc_inp.transpose(2, 1))
        scores = scores.masked_fill(dec_enc_attn_mask, -np.inf)

        oov_vocab_prob = torch.softmax(scores, -1)

        full_vocab_prob = full_vocab_prob.scatter_add(2, input_ids.unsqueeze(1).repeat(1, full_vocab_prob.shape[1], 1), oov_vocab_prob * copy_gate)

        if positionwise_copy_prob:
            return torch.log(full_vocab_prob + 1e-8), oov_vocab_prob * copy_gate
        else:
            return torch.log(full_vocab_prob + 1e-8)

    def greedy_decode(self, input_ids, entity_ids, triple_ids, position_ids, banwords=[], max_token_seq_len=30):
        enc_inp = self.embedding(input_ids, entity_ids, triple_ids, position_ids)
        
        # -- Encode source
        non_pad_mask = get_non_pad_mask(input_ids, self.pad_idx)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=input_ids, pad_idx=self.pad_idx)        
        for layer in self.enc_layer_stack:
            enc_inp, _ = layer(enc_inp, non_pad_mask, slf_attn_mask)
        enc_output = enc_inp

        batch_size = input_ids.shape[0]
        tgt_seq = torch.LongTensor(batch_size, 1).fill_(self.sos_idx).to(input_ids.device)

        for step in range(max_token_seq_len):
            non_pad_mask = get_non_pad_mask(tgt_seq, self.pad_idx)

            slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq, pad_idx=self.pad_idx)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
            dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=tgt_seq, pad_idx=self.pad_idx)

            dec_output = self.embedding.word_embeddings(tgt_seq) + self.post_word_emb(tgt_seq)

            for layer in self.dec_layer_stack:
                dec_output, dec_slf_attn, dec_enc_attn = layer(
                    dec_output, enc_output,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask,
                    dec_enc_attn_mask=dec_enc_attn_mask)

            dec_output = dec_output[:, -1, :].unsqueeze(1)
            
            copy_gate = torch.sigmoid(self.gate(dec_output))

            in_vocab_prob = torch.softmax(self.proj(dec_output), -1)
            full_vocab_prob = (1 - copy_gate) * in_vocab_prob

            scores = torch.bmm(dec_output, enc_inp.transpose(2, 1))
            scores = scores.masked_fill(dec_enc_attn_mask, -np.inf)

            oov_vocab_prob = torch.softmax(scores, -1)

            full_vocab_prob = full_vocab_prob.scatter_add(2, input_ids.unsqueeze(1).repeat(1, full_vocab_prob.shape[1], 1), oov_vocab_prob * copy_gate)

            decoded = torch.argmax(full_vocab_prob, -1)

            tgt_seq = torch.cat([tgt_seq, decoded], -1)

        return tgt_seq[:, 1:]


    def beam_search(self, inputs, tokenizer, n_bm, max_token_seq_len=30, banwords=[]):
        with torch.no_grad():
            #-- Repeat data for beam search
            n_inst, len_s = inputs[0].size()
            
            inputs = [_.repeat(1, n_bm).view(n_inst * n_bm, len_s) for _ in inputs]

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, device=inputs[0].device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, max_token_seq_len + 1):
                active_inst_idx_list = beam_decode_step(self, inst_dec_beams, len_dec_seq, active_inst_idx_list,
                                                        inputs, inst_idx_to_position_map, n_bm, banwords)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                inputs, inst_idx_to_position_map = collate_active_info(inputs, inst_idx_to_position_map, active_inst_idx_list, n_bm)
        
            batch_hyp, batch_scores = collect_hypothesis_and_scores(tokenizer.pad_token_id, inst_dec_beams, n_bm)
            
            result = []
            for _ in batch_hyp:
                finished = False
                for r in _:
                    if len(r) >= 8:
                        result.append(r)
                        finished = True
                        break
                if not finished:
                    result.append(_[0])
            
            return result


class GraphGatedTransformerDecoder(nn.Module):
    def __init__(self, config, n_head, n_layers):
        super(GraphGatedTransformerDecoder, self).__init__()
        d_k = config.hidden_size // n_head
        d_v = config.hidden_size // n_head
        d_inner = config.hidden_size * 4
        self.pad_idx = config.pad_token_id
        self.sos_idx = config.sos_token_id
        self.eos_idx = config.eos_token_id

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.post_word_emb = PositionalEmbedding(d_model=config.hidden_size, max_len=1024)
        
        self.enc_layer_stack = nn.ModuleList([
            EncoderLayer(config.hidden_size, d_inner, n_head, d_k, d_v, dropout=config.hidden_dropout_prob)
            for _ in range(n_layers)])

        self.dec_layer_stack = nn.ModuleList([
            DecoderLayer(config.hidden_size, d_inner, n_head, d_k, d_v, dropout=config.hidden_dropout_prob)
            for _ in range(n_layers)])

        self.gate = nn.Linear(config.hidden_size, 1)

        self.proj = nn.Linear(config.hidden_size, config.vocab_size)

        self.term_layer = [0, 1]
        self.triple_layer = [2]
        self.entity_layer = [3]
        self.fact_layer = [4]
        self.gather_layer = [5]

    def encode(self, input_ids, term_level_mask, triple_level_mask, entity_level_mask, fact_level_mask, gather_level_mask):
        # -- Encode source
        enc_inp = self.embedding(input_ids) + self.post_word_emb(input_ids)
        non_pad_mask = get_non_pad_mask(input_ids, self.pad_idx)        
        for layer_idx, layer in enumerate(self.enc_layer_stack):
            if layer_idx in self.term_layer:
                enc_inp, _ = layer(enc_inp, non_pad_mask, term_level_mask)
            elif layer_idx in self.triple_layer:
                enc_inp, _ = layer(enc_inp, non_pad_mask, triple_level_mask)                
            elif layer_idx in self.entity_layer:
                enc_inp, _ = layer(enc_inp, non_pad_mask, entity_level_mask)
            elif layer_idx in self.fact_layer:
                enc_inp, _ = layer(enc_inp, non_pad_mask, fact_level_mask)
            elif layer_idx in self.gather_layer:
                enc_inp, _ = layer(enc_inp, non_pad_mask, gather_level_mask)                
            else:
                raise NotImplementedError
        enc_output = enc_inp

        return enc_output

    def forward(self, input_ids, term_level_mask, triple_level_mask, entity_level_mask, fact_level_mask, gather_level_mask, tgt_seq, positionwise_copy_prob=False):
        enc_output = self.encode(input_ids, term_level_mask, triple_level_mask, entity_level_mask, fact_level_mask, gather_level_mask)
        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq, self.pad_idx)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq, pad_idx=self.pad_idx)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=tgt_seq, pad_idx=self.pad_idx)

        dec_output = self.embedding(tgt_seq) + self.post_word_emb(tgt_seq)

        for layer in self.dec_layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

        copy_gate = torch.sigmoid(self.gate(dec_output))

        in_vocab_prob = torch.softmax(self.proj(dec_output), -1)
        full_vocab_prob = (1 - copy_gate) * in_vocab_prob

        scores = torch.bmm(dec_output, enc_output.transpose(2, 1))
        scores = scores.masked_fill(dec_enc_attn_mask, -np.inf)

        oov_vocab_prob = torch.softmax(scores, -1)

        full_vocab_prob = full_vocab_prob.scatter_add(2, input_ids.unsqueeze(1).repeat(1, full_vocab_prob.shape[1], 1), oov_vocab_prob * copy_gate)

        if positionwise_copy_prob:
            return torch.log(full_vocab_prob + 1e-8), oov_vocab_prob * copy_gate
        else:
            return torch.log(full_vocab_prob + 1e-8)

    def greedy_decode(self, input_ids, term_level_mask, triple_level_mask, entity_level_mask, fact_level_mask, gather_level_mask, banwords=[], max_token_seq_len=30):
        enc_output = self.encode(input_ids, term_level_mask, triple_level_mask, entity_level_mask, fact_level_mask, gather_level_mask)        

        batch_size = input_ids.shape[0]
        tgt_seq = torch.LongTensor(batch_size, 1).fill_(self.sos_idx).to(input_ids.device)

        for step in range(max_token_seq_len):
            non_pad_mask = get_non_pad_mask(tgt_seq, self.pad_idx)

            slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq, pad_idx=self.pad_idx)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
            dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=input_ids, seq_q=tgt_seq, pad_idx=self.pad_idx)

            dec_output = self.embedding(tgt_seq) + self.post_word_emb(tgt_seq)

            for layer in self.dec_layer_stack:
                dec_output, dec_slf_attn, dec_enc_attn = layer(
                    dec_output, enc_output,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask,
                    dec_enc_attn_mask=dec_enc_attn_mask)

            dec_output = dec_output[:, -1, :].unsqueeze(1)
            
            copy_gate = torch.sigmoid(self.gate(dec_output))

            in_vocab_prob = torch.softmax(self.proj(dec_output), -1)
            full_vocab_prob = (1 - copy_gate) * in_vocab_prob

            scores = torch.bmm(dec_output, enc_output.transpose(2, 1))
            scores = scores.masked_fill(dec_enc_attn_mask, -np.inf)

            oov_vocab_prob = torch.softmax(scores, -1)

            full_vocab_prob = full_vocab_prob.scatter_add(2, input_ids.unsqueeze(1).repeat(1, full_vocab_prob.shape[1], 1), oov_vocab_prob * copy_gate)

            decoded = torch.argmax(full_vocab_prob, -1)

            tgt_seq = torch.cat([tgt_seq, decoded], -1)

        return tgt_seq[:, 1:]

    def beam_search(self, inputs, tokenizer, n_bm, max_token_seq_len=30, banwords=[]):
        with torch.no_grad():
            #-- Repeat data for beam search
            n_inst, len_s = inputs[0].size()

            new_inputs = []
            for _ in inputs:
                if len(_.shape) == 2:
                    new_inputs.append(_.repeat(1, n_bm).view(n_inst * n_bm, len_s))
                else:
                    new_inputs.append(_.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, len_s))
            inputs = new_inputs

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, device=inputs[0].device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, max_token_seq_len + 1):
                active_inst_idx_list = beam_decode_step(self, inst_dec_beams, len_dec_seq, active_inst_idx_list,
                                                        inputs, inst_idx_to_position_map, n_bm, banwords)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                inputs, inst_idx_to_position_map = collate_active_info(inputs, inst_idx_to_position_map, active_inst_idx_list, n_bm)

            batch_hyp, batch_scores = collect_hypothesis_and_scores(tokenizer.pad_token_id, inst_dec_beams, n_bm)

            result = []
            for _ in batch_hyp:
                finished = False
                for r in _:
                    if len(r) >= 8:
                        result.append(r)
                        finished = True
                        break
                if not finished:
                    result.append(_[0])

            return result


def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    ''' Collect tensor parts associated to active instances. '''
    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * n_bm, *d_hs)

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)

    return beamed_tensor

def collate_active_info(inputs, inst_idx_to_position_map, active_inst_idx_list, n_bm):
    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(inputs[0].device)

    active_inputs = [collect_active_part(_, active_inst_idx, n_prev_active_inst, n_bm) for _ in inputs]
    
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    return active_inputs, active_inst_idx_to_position_map

def beam_decode_step(model, inst_dec_beams, len_dec_seq, active_inst_idx_list, inputs, \
                     inst_idx_to_position_map, n_bm, banwords):
    ''' Decode and update beam status, and then return active beam idx '''
    n_active_inst = len(inst_idx_to_position_map)

    #dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
    dec_partial_seq = [inst_dec_beams[idx].get_current_state() 
                       for idx in active_inst_idx_list if not inst_dec_beams[idx].done]
    dec_partial_seq = torch.stack(dec_partial_seq).to(inputs[0].device)
    dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
    
    combined_inputs = inputs + [dec_partial_seq]
    logits = model.forward(*combined_inputs)[:, -1, :] / 1.

    if len_dec_seq <= 5:
        logits[:, banwords] = -np.inf
    
    word_prob = F.log_softmax(logits, dim=1)
    word_prob = word_prob.view(n_active_inst, n_bm, -1)
    
    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = []
    for inst_idx, inst_position in inst_idx_to_position_map.items():
        is_inst_complete = inst_dec_beams[inst_idx].advance(word_prob[inst_position])
        if not is_inst_complete:
            active_inst_idx_list += [inst_idx]

    return active_inst_idx_list     

def collect_hypothesis_and_scores(pad_idx, inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for beam in inst_dec_beams:
        scores = beam.scores
        
        hyps = np.array([beam.get_hypothesis(i) for i in range(beam.size)], 'long')
        lengths = (hyps != pad_idx).sum(-1)
        normed_scores = [scores[i].item()/lengths[i] for i, hyp in enumerate(hyps)]
        idxs = np.argsort(normed_scores)[::-1]
        
        all_hyp.append([hyps[idx] for idx in idxs])
        all_scores.append([normed_scores[idx] for idx in idxs])

    return all_hyp, all_scores

class GPT2Wrapper(object):
    def __init__(self, model):
        self.model = model

    def greedy_search(self, inputs, tokenizer, max_len):
        batch_size = inputs.shape[0]
        prefix_length = inputs.shape[1]
        sos_token = tokenizer.bos_token_id
        device = inputs.device

        tgt_seq = torch.LongTensor(batch_size, 1).fill_(sos_token).to(device)
        for _ in range(max_len):
            logits = self.forward(inputs, tgt_seq)[:, -1, :]
            decoded_token = torch.argmax(logits, -1).unsqueeze(-1)
            tgt_seq = torch.cat([tgt_seq, decoded_token], -1)

        return tgt_seq[:, 1:]

    def beam_search(self, inputs, tokenizer, n_bm, max_token_seq_len=30, banwords=[]):
        #prefix_length = inputs[0].shape[1]
        with torch.no_grad():
            #-- Repeat data for beam search
            n_inst, len_s = inputs[0].size()
            
            inputs = [_.repeat(1, n_bm).view(n_inst * n_bm, len_s) for _ in inputs]             

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, device=inputs[0].device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, max_token_seq_len + 1):
                active_inst_idx_list = beam_decode_step(self, inst_dec_beams, len_dec_seq, active_inst_idx_list,
                                                        inputs, inst_idx_to_position_map, n_bm, banwords)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                inputs, inst_idx_to_position_map = collate_active_info(inputs, inst_idx_to_position_map, active_inst_idx_list, n_bm)
        
            batch_hyp, batch_scores = collect_hypothesis_and_scores(tokenizer.pad_token_id, inst_dec_beams, n_bm)
            
            result = []
            for _ in batch_hyp:
                finished = False
                for r in _:
                    if len(r) >= 8:
                        result.append(r)
                        finished = True
                        break
                if not finished:
                    result.append(_[0])
            
            return result

    def forward(self, prefix, tgt_seq):
        inputs = torch.cat([prefix, tgt_seq], -1)
        logits = self.model(inputs)[0]
        return logits

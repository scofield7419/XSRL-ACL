from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from DataUtils.Common import *
from Engines.initialize import *
from allennlp.modules.elmo import Elmo
from Engines.PGNLSTM import *

torch.manual_seed(seed_num)
random.seed(seed_num)


class Encoder(nn.Module):

    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        total_dim = 0

        self.elmo = Elmo(self.options_file, self.weight_file, 3, vocab_to_cache=self.elmo_vocab,
                         requires_grad=False, dropout=0)
        total_dim += 1024

        if self.is_predicate_position:
            total_dim += 1

        if self.is_predicate:
            prd_paddingId = self.prd_paddingId
            prd_embed_num = self.prd_embed_num
            prd_embed_dim = self.prd_embed_dim
            total_dim += prd_embed_dim
            self.prd_embed = nn.Embedding(prd_embed_num, prd_embed_dim, padding_idx=prd_paddingId)

        if self.is_pos:
            pos_paddingId = self.pos_paddingId
            pos_embed_num = self.pos_embed_num
            pos_embed_dim = self.pos_embed_dim
            total_dim += pos_embed_dim
            init_embedding(self.pos_embed.weight)
            self.pos_embed = nn.Embedding(pos_embed_num, pos_embed_dim, padding_idx=pos_paddingId)

        self.lang_embed = nn.Embedding(len(self.lang2id) + 1, self.lang_emb_dim, padding_idx=len(self.lang2id))

        C = self.label_num

        self.dropout_embed = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout_emb)

        self.pgn = PGNLSTM(input_size=total_dim, hidden_size=self.lstm_hiddens, num_layers=self.lstm_layers,
                            task_dim_size=self.lang_emb_dim, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(in_features=self.lstm_hiddens * 2, out_features=C, bias=True)
        init_linear(self.linear)

    def forward(self, elmo_char_seqs, elmo_word_seqs, word, lang,pos,  prd, x_prd_posi, mask, sentence_length):

        if self.is_predicate_position:
            x_prd_posi = x_prd_posi.float()
            x_prd_posi = x_prd_posi.unsqueeze(2)
            x = x_prd_posi

        if self.is_predicate:
            x_prd = self.prd_embed(prd)
            x_prd = self.dropout_embed(x_prd)
            x = torch.cat((x, x_prd), 2)

        if self.is_pos:
            x_pos = self.pos_embed(pos)
            x_pos = self.dropout_embed(x_pos)
            x = torch.cat((x, x_pos), 2)

        x_elmo_pack = self.elmo(elmo_char_seqs, elmo_word_seqs)
        x_elmo_embeddings = x_elmo_pack['elmo_representations']
        x_elmo = (x_elmo_embeddings[0] + x_elmo_embeddings[1] + x_elmo_embeddings[2]) / 3.0
        x = torch.cat((x, x_elmo), 2)

        x_lang = self.lang_embed(lang)

        x, _ = self.pgn(x_lang, x, mask)
        x = x.transpose(1, 0)
        x = self.dropout(x)
        x = torch.tanh(x)
        logit = self.linear(x)

        return logit

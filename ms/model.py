from operator import length_hint
import torch
import math
from torch import nn
from torch import Tensor
from .layers import *
from torch.nn.functional import normalize
import torch.nn.functional as F
DIM = 128
DROPOUT = 0.1


class Compose_single(nn.Module):
    def __init__(self, spect_model, irt_model, spect_dim=174) -> None:
        super().__init__()
        self.spect_model = spect_model
        self.irt_model = irt_model
        self.classify = nn.Sequential(
            nn.Linear(spect_dim * 2 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, data):
        spect = self.spect_model(data)
        irt = self.irt_model(data)

        spect_mask = (data['intensities_raw'] >= 0).float()
        spect_ov = data['intensities_raw'] * spect_mask
        spect_pr = spect * spect_mask
        spect_feat = torch.cat(
            (spect_pr, spect_ov.float(), irt, data['irt'].float()), dim=1)
        return self.classify(spect_feat).squeeze()


class pDeep2_nomod(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_size = 256
        self.peptide_dim = kwargs.pop('peptide_dim', 22)
        self.instrument_size = 8
        self.input_size = self.peptide_dim * 4 + 2 + 1 + 3
        self.ions_dim = kwargs.pop('ions_dim', 6)
        self.instrument_ce_scope = "instrument_nce"
        self.rnn_dropout = 0.2
        self.output_dropout = 0.2
        self.init_layers()

    def init_layers(self):
        self.lstm_layer1 = nn.LSTM(
            self.input_size, self.layer_size, batch_first=True, bidirectional=True)
        self.lstm_layer2 = nn.LSTM(
            self.layer_size * 2 + 1 + 3, self.layer_size, batch_first=True, bidirectional=True)

        self.lstm_output_layer = nn.LSTM(
            self.layer_size * 2 + 1 + 3, self.ions_dim, bidirectional=True, batch_first=True
        )
        self.linear_inst_proj = nn.Linear(
            self.instrument_size + 1, 3, bias=False)
        self.dropout = nn.Dropout(p=self.output_dropout)

    def comment(self):
        return "pDeep2"

    def pdeep2_long_feature(self, data):
        peptides = F.one_hot(data['sequence_integer'],
                             num_classes=self.peptide_dim)
        peptides_mask = data['peptide_mask']
        peptides_length = torch.sum(peptides_mask, dim=1)
        pep_dim = peptides.shape[2]
        assert pep_dim == self.peptide_dim
        long_feature = peptides.new_zeros(
            (peptides.shape[0], peptides.shape[1] - 1, pep_dim * 4 + 2))
        long_feature[:, :, :pep_dim] = peptides[:, :-1, :]
        long_feature[:, :, pep_dim:2 * pep_dim] = peptides[:, 1:, :]
        for i in range(peptides.shape[1] - 1):
            long_feature[:, i, 2 * pep_dim:3 * pep_dim] = torch.sum(
                peptides[:, :i, :], dim=1) if i != 0 else peptides.new_zeros((peptides.shape[0], pep_dim))
            long_feature[:, i, 3 * pep_dim:4 * pep_dim] = torch.sum(peptides[:, (i + 2):, :], dim=1) if i == (
                peptides.shape[1] - 2) else peptides.new_zeros((peptides.shape[0], pep_dim))
            long_feature[:, i, 4 * pep_dim] = 1 if (i == 0) else 0
            long_feature[:, i, 4 * pep_dim + 1] = ((peptides_length - 2) == i)
        return long_feature

    def add_leng_dim(self, x, length):
        x = x.unsqueeze(dim=1)
        shape_repeat = [1] * len(x.shape)
        shape_repeat[1] = length
        return x.repeat(*shape_repeat)

    def forward(self, data, **kwargs):
        peptides = self.pdeep2_long_feature(data)  # n-1 input

        nce = data['collision_energy_aligned_normed'].float()
        charge = data['precursor_charge_onehot'].float()
        charge = torch.argmax(charge, dim=1).unsqueeze(-1)

        B = peptides.shape[0]
        peptides_length = peptides.shape[1]
        inst_feat = charge.new_zeros((B, self.instrument_size))
        # ['QE', 'Velos', 'Elite', 'Fusion', 'Lumos', 'unknown']
        inst_feat[: 5] = 1
        charge = self.add_leng_dim(charge, peptides_length)
        nce = self.add_leng_dim(nce, peptides_length)
        inst_feat = self.add_leng_dim(inst_feat, peptides_length)

        # print(peptides.shape, inst_feat.shape, nce.shape, charge.shape)

        proj_inst = self.linear_inst_proj(torch.cat([inst_feat, nce], dim=2))
        x = torch.cat([peptides, charge, proj_inst], dim=2)

        x, _ = self.lstm_layer1(x)
        x = self.dropout(x)
        x = torch.cat([x, charge, proj_inst], dim=2)
        x, _ = self.lstm_layer2(x)
        x = self.dropout(x)
        x = torch.cat([x, charge, proj_inst], dim=2)
        output, _ = self.lstm_output_layer(x)
        output = (output[:, :, :self.ions_dim] + output[:, :, self.ions_dim:])

        return output.reshape(B, -1)


class TransPro(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_num = kwargs.pop('layer_num', 6)
        self.frag_layer_num = kwargs.pop('frag_layer_num', 3)
        self.in_frag_layer_num = kwargs.pop('frag_layer_num_in', 2)
        # self.layer_num = kwargs.pop('layer_num', 1)
        self.peptide_dim = kwargs.pop('peptide_dim', 21)
        self.peptide_embed_dim = kwargs.pop('peptide_embed_dim', DIM)
        self.percursor_dim = kwargs.pop('peptide_embed_dim', 6)
        self.inner_dim = kwargs.pop('inner_dim', 256)

        self.pos_dim = kwargs.pop('pos_dim', 30)
        self.ions_dim = kwargs.pop('ions_dim', 6)
        self.n_head = kwargs.pop('n_head', 8)
        self.init()

    def comment(self):
        return f"TransPro-cls-{self.layer_num}-{self.frag_layer_num}-{self.peptide_embed_dim}-{DROPOUT}-{self.inner_dim}"

    def init(self):
        self.p_embed_token = TokenEmbedding(
            self.peptide_dim + 1, self.peptide_embed_dim)
        self.p_embed_pos = PositionalEncoding_fix(
            self.pos_dim + 1, self.peptide_embed_dim)
        # assert peptide id starts from 1, cls for zeros
        self.c_embed = nn.Linear(self.percursor_dim, self.peptide_embed_dim)
        self.n_embed = nn.Linear(1, self.peptide_embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.peptide_embed_dim, nhead=8, dropout=DROPOUT, dim_feedforward=self.inner_dim)
        self.p_trans = nn.TransformerEncoder(
            encoder_layer, num_layers=self.layer_num)

        self.frag_trans = nn.TransformerEncoder(
            encoder_layer, num_layers=self.frag_layer_num)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.peptide_embed_dim * 2, nhead=8, dropout=DROPOUT, dim_feedforward=self.inner_dim * 2)
        self.in_frag_trans = nn.TransformerEncoder(
            encoder_layer, num_layers=self.in_frag_layer_num)

        # self.in_frag = nn.Linear(self.pos_dim-1, self.pos_dim-1)

        self.irt_decoder = nn.Sequential(
            # nn.Linear(
            #     self.peptide_embed_dim, self.peptide_embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(
                self.peptide_embed_dim, 1)
        )

        self.frag_tower = FragAttention()
        self.frag_final_decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(
                self.peptide_embed_dim * 2, self.ions_dim),
        )

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=DROPOUT)

    def add_cls(self, peptides: Tensor, mask: Tensor):
        batch_len = peptides.shape[0]
        clss = peptides.new_full((batch_len, 1), 0)
        clss_mask = mask.new_full((batch_len, 1), False)
        return torch.cat([clss, peptides], dim=1).long(), torch.cat([clss_mask, mask], dim=1)

    def forward(self, x, choice="frag"):
        assert choice in ['frag', 'irt', 'both']
        peptides = x['sequence_integer']
        peptides_mask = x['peptide_mask']
        B = peptides.shape[0]

        peptides, peptides_mask = self.add_cls(peptides, peptides_mask)
        # B, S+1

        pepti_embed = self.p_embed_token(peptides)
        pepti_embed = self.p_embed_pos(
            pepti_embed, peptides_mask).transpose(0, 1)
        # B, S+1, D
        trans_embed = self.p_trans(
            pepti_embed, src_key_padding_mask=peptides_mask)
        if choice == 'irt':
            irt_embed = trans_embed[0]
            irt = self.irt_decoder(irt_embed)
            return irt
        else:
            frag_embed = trans_embed[1:]
            frag_mask = peptides_mask[:, 1:]

            nce = x['collision_energy_aligned_normed'].float()
            charge = x['precursor_charge_onehot'].float()

            charge_embed = self.c_embed(charge)
            nce_embed = self.n_embed(nce)

            frag_embed = frag_embed + nce_embed + charge_embed
            # frag_embed = frag_embed * nce_embed * charge_embed

            frag_embed = self.frag_trans(
                frag_embed, src_key_padding_mask=frag_mask)

            frag_embed = self.frag_tower(frag_embed, frag_mask).transpose(0, 1)
            in_frag_mask = frag_mask[:, 1:]
            frag_embed = self.in_frag_trans(
                frag_embed, src_key_padding_mask=in_frag_mask).transpose(0, 1)

            frag_ions = self.frag_final_decoder(frag_embed)
            frag_ions = frag_ions.reshape(B, -1)
            return frag_ions


class TransProDeep(TransPro):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frag_tower = Deepmatch()

    def comment(self):
        return f"TransProDeep-cls-{self.layer_num}-{self.frag_layer_num}-{self.peptide_embed_dim}-{DROPOUT}-{self.inner_dim}"


class PrositIRT(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.peptide_dim = kwargs.pop('peptide_dim', 22)
        self.peptide_embed_dim = kwargs.pop('peptide_embed_dim', 32)
        self.hidden_size = kwargs.pop('bi_dim', 256)

        self.embedding = nn.Embedding(self.peptide_dim, self.peptide_embed_dim)
        self.bi = nn.GRU(input_size=self.peptide_embed_dim,
                         hidden_size=self.hidden_size,
                         bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        self.gru = nn.GRU(input_size=self.hidden_size * 2,
                          hidden_size=self.hidden_size * 2)
        self.agg = AttentalSum(self.hidden_size * 2)
        self.leaky = nn.LeakyReLU()
        self.drop2 = nn.Dropout(p=0.1)
        self.decoder = nn.Linear(self.hidden_size * 2, 1)

    def comment(self):
        return "PrositIRT"

    def forward(self, x, **kwargs):
        peptides = x['sequence_integer']
        # peptides_mask = x['peptide_mask']
        # S = peptides_mask.shape[1]
        x = self.embedding(peptides)
        x = x.transpose(0, 1)
        x, _ = self.bi(x)
        x = self.drop(x)
        x, _ = self.gru(x)
        x = self.drop(x)
        x = self.agg(x)
        x = self.drop2(self.leaky(x))
        return self.decoder(x)


class PrositFrag(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.peptide_dim = kwargs.pop('peptide_dim', 22)
        self.peptide_embed_dim = kwargs.pop('peptide_embed_dim', 32)
        self.percursor_dim = kwargs.pop('peptide_embed_dim', 6)
        self.hidden_size = kwargs.pop('bi_dim', 256)
        self.max_sequence = kwargs.pop('max_lenght', 30)

        self.embedding = nn.Embedding(self.peptide_dim, self.peptide_embed_dim)
        self.bi = nn.GRU(input_size=self.peptide_embed_dim,
                         hidden_size=self.hidden_size,
                         bidirectional=True)
        self.drop3 = nn.Dropout(p=0.3)
        self.gru = nn.GRU(input_size=self.hidden_size * 2,
                          hidden_size=self.hidden_size * 2)
        self.agg = AttentalSum(self.hidden_size * 2)
        self.leaky = nn.LeakyReLU()

        self.side_encoder = nn.Linear(
            self.percursor_dim + 1, self.hidden_size * 2)

        self.gru_decoder = nn.GRU(input_size=self.hidden_size * 2,
                                  hidden_size=self.hidden_size * 2)
        self.in_frag = nn.Linear(self.max_sequence - 1, self.max_sequence - 1)
        self.final_decoder = nn.Linear(self.hidden_size * 2, 6)

    def comment(self):
        return "PrositFrag"

    def forward(self, x, **kwargs):
        self.bi.flatten_parameters()
        self.gru.flatten_parameters()
        self.gru_decoder.flatten_parameters()

        peptides = x['sequence_integer']
        nce = x['collision_energy_aligned_normed'].float()
        charge = x['precursor_charge_onehot'].float()
        B = peptides.shape[0]
        x = self.embedding(peptides)
        x = x.transpose(0, 1)
        x, _ = self.bi(x)
        x = self.drop3(x)
        x, _ = self.gru(x)
        x = self.drop3(x)
        x = self.agg(x)

        side_input = torch.cat([charge, nce], dim=1)
        side_info = self.side_encoder(side_input)
        side_info = self.drop3(side_info)

        x = x * side_info
        x = x.expand(self.max_sequence - 1, x.shape[0], x.shape[1])
        x, _ = self.gru_decoder(x)
        x = self.drop3(x)
        x_d = self.in_frag(x.transpose(0, 2))

        x = x * x_d.transpose(0, 2)
        x = self.final_decoder(x)
        x = self.leaky(x)
        x = x.transpose(0, 1).reshape(B, -1)
        return x


class pDeep2_nomod(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_size = 256
        self.peptide_dim = kwargs.pop('peptide_dim', 22)
        self.instrument_size = 8
        self.input_size = self.peptide_dim * 4 + 2 + 1 + 3
        self.ions_dim = kwargs.pop('ions_dim', 6)
        self.instrument_ce_scope = "instrument_nce"
        self.rnn_dropout = 0.2
        self.output_dropout = 0.2
        self.init_layers()

    def init_layers(self):
        self.lstm_layer1 = nn.LSTM(
            self.input_size, self.layer_size, batch_first=True, bidirectional=True)
        self.lstm_layer2 = nn.LSTM(
            self.layer_size * 2 + 1 + 3, self.layer_size, batch_first=True, bidirectional=True)

        self.lstm_output_layer = nn.LSTM(
            self.layer_size * 2 + 1 + 3, self.ions_dim, bidirectional=True, batch_first=True
        )
        self.linear_inst_proj = nn.Linear(
            self.instrument_size + 1, 3, bias=False)
        self.dropout = nn.Dropout(p=self.output_dropout)

    def comment(self):
        return "pDeep2"

    def pdeep2_long_feature(self, data):
        peptides = F.one_hot(data['sequence_integer'],
                             num_classes=self.peptide_dim)
        peptides_mask = data['peptide_mask']
        peptides_length = torch.sum(peptides_mask, dim=1)
        pep_dim = peptides.shape[2]
        assert pep_dim == self.peptide_dim
        long_feature = peptides.new_zeros(
            (peptides.shape[0], peptides.shape[1] - 1, pep_dim * 4 + 2))
        long_feature[:, :, :pep_dim] = peptides[:, :-1, :]
        long_feature[:, :, pep_dim:2 * pep_dim] = peptides[:, 1:, :]
        for i in range(peptides.shape[1] - 1):
            long_feature[:, i, 2 * pep_dim:3 * pep_dim] = torch.sum(
                peptides[:, :i, :], dim=1) if i != 0 else peptides.new_zeros((peptides.shape[0], pep_dim))
            long_feature[:, i, 3 * pep_dim:4 * pep_dim] = torch.sum(peptides[:, (i + 2):, :], dim=1) if i == (
                peptides.shape[1] - 2) else peptides.new_zeros((peptides.shape[0], pep_dim))
            long_feature[:, i, 4 * pep_dim] = 1 if (i == 0) else 0
            long_feature[:, i, 4 * pep_dim + 1] = ((peptides_length - 2) == i)
        return long_feature

    def add_leng_dim(self, x, length):
        x = x.unsqueeze(dim=1)
        shape_repeat = [1] * len(x.shape)
        shape_repeat[1] = length
        return x.repeat(*shape_repeat)

    def forward(self, data, **kwargs):
        peptides = self.pdeep2_long_feature(data)  # n-1 input

        nce = data['collision_energy_aligned_normed'].float()
        charge = data['precursor_charge_onehot'].float()
        charge = torch.argmax(charge, dim=1).unsqueeze(-1)

        B = peptides.shape[0]
        peptides_length = peptides.shape[1]
        inst_feat = charge.new_zeros((B, self.instrument_size))
        # ['QE', 'Velos', 'Elite', 'Fusion', 'Lumos', 'unknown']
        inst_feat[: 5] = 1
        charge = self.add_leng_dim(charge, peptides_length)
        nce = self.add_leng_dim(nce, peptides_length)
        inst_feat = self.add_leng_dim(inst_feat, peptides_length)

        # print(peptides.shape, inst_feat.shape, nce.shape, charge.shape)

        proj_inst = self.linear_inst_proj(torch.cat([inst_feat, nce], dim=2))
        x = torch.cat([peptides, charge, proj_inst], dim=2)

        x, _ = self.lstm_layer1(x)
        x = self.dropout(x)
        x = torch.cat([x, charge, proj_inst], dim=2)
        x, _ = self.lstm_layer2(x)
        x = self.dropout(x)
        x = torch.cat([x, charge, proj_inst], dim=2)
        output, _ = self.lstm_output_layer(x)
        output = (output[:, :, :self.ions_dim] + output[:, :, self.ions_dim:])

        return output.reshape(B, -1)


class TransProBest(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_num = kwargs.pop('layer_num', 6)
        self.frag_layer_num = kwargs.pop('frag_layer_num', 3)
        self.in_frag_layer_num = kwargs.pop('frag_layer_num', 1)
        # self.layer_num = kwargs.pop('layer_num', 1)
        self.peptide_dim = kwargs.pop('peptide_dim', 21)
        self.peptide_embed_dim = kwargs.pop('peptide_embed_dim', 128)
        self.percursor_dim = kwargs.pop('peptide_embed_dim', 6)
        self.inner_dim = kwargs.pop('inner_dim', 256)

        self.pos_dim = kwargs.pop('pos_dim', 30)
        self.ions_dim = kwargs.pop('ions_dim', 6)
        self.n_head = kwargs.pop('n_head', 8)
        self.init()

    def comment(self):
        return f"TransProBest-{self.layer_num}-{self.frag_layer_num}-{self.peptide_embed_dim}-{DROPOUT}-{self.inner_dim}"

    def init(self):
        self.p_embed_token = TokenEmbedding(
            self.peptide_dim + 1, self.peptide_embed_dim)
        self.p_embed_pos = PositionalEncoding_fix(
            self.pos_dim + 1, self.peptide_embed_dim)
        # assert peptide id starts from 1, cls for zeros
        self.c_embed = nn.Linear(self.percursor_dim, self.peptide_embed_dim)
        self.n_embed = nn.Linear(1, self.peptide_embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.peptide_embed_dim, nhead=8, dropout=DROPOUT, dim_feedforward=self.inner_dim)
        self.p_trans = nn.TransformerEncoder(
            encoder_layer, num_layers=self.layer_num)

        self.frag_trans = nn.TransformerEncoder(
            encoder_layer, num_layers=self.frag_layer_num)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.peptide_embed_dim * 2, nhead=8, dropout=DROPOUT, dim_feedforward=self.inner_dim)

        self.in_frag = nn.Linear(self.pos_dim - 1, self.pos_dim - 1)

        # self.in_frag_trans = nn.TransformerEncoder(
        #     encoder_layer, num_layers=self.in_frag_layer_num)
        self.irt_decoder = nn.Sequential(
            # nn.Linear(
            #     self.peptide_embed_dim, self.peptide_embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(
                self.peptide_embed_dim, 1)
        )

        self.frag_tower = FragAttention()
        self.frag_final_decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(
                self.peptide_embed_dim * 2, self.ions_dim),
        )

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=DROPOUT)

    def add_cls(self, peptides: Tensor, mask: Tensor):
        batch_len = peptides.shape[0]
        clss = peptides.new_full((batch_len, 1), 0)
        clss_mask = mask.new_full((batch_len, 1), False)
        return torch.cat([clss, peptides], dim=1).long(), torch.cat([clss_mask, mask], dim=1)

    def forward(self, x, choice="frag"):
        assert choice in ['frag', 'irt', 'both']
        peptides = x['sequence_integer']
        peptides_mask = x['peptide_mask']
        B = peptides.shape[0]

        peptides, peptides_mask = self.add_cls(peptides, peptides_mask)
        # B, S+1

        pepti_embed = self.p_embed_token(peptides)
        pepti_embed = self.p_embed_pos(
            pepti_embed, peptides_mask).transpose(0, 1)
        # B, S+1, D
        trans_embed = self.p_trans(
            pepti_embed, src_key_padding_mask=peptides_mask)
        if choice == 'irt':
            irt_embed = trans_embed[0]
            irt = self.irt_decoder(irt_embed)
            return irt
        else:
            frag_embed = trans_embed[1:]
            frag_mask = peptides_mask[:, 1:]

            nce = x['collision_energy_aligned_normed'].float()
            charge = x['precursor_charge_onehot'].float()

            charge_embed = self.c_embed(charge)
            nce_embed = self.n_embed(nce)

            frag_embed = frag_embed + nce_embed + charge_embed
            frag_embed = self.frag_trans(
                frag_embed, src_key_padding_mask=frag_mask)
            # frag_embed = frag_embed * nce_embed * charge_embed

            # B, S-1, D*2
            frag_embed = self.frag_tower(frag_embed, frag_mask)
            # frag_embed = self.in_frag(
            #     frag_embed.transpose(1, 2)).transpose(1, 2)
            frag_ions = self.frag_final_decoder(frag_embed)
            # reverse y ions loc
            frag_ions[:, :, :3] = torch.flip(frag_ions[:, :, :3], dims=(1, ))
            return frag_ions.reshape(B, -1)


class TransProIRT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_num = kwargs.pop('layer_num', 9)

        # self.layer_num = kwargs.pop('layer_num', 1)
        self.peptide_dim = kwargs.pop('peptide_dim', 21)
        self.peptide_embed_dim = kwargs.pop('peptide_embed_dim', 64)
        self.inner_dim = kwargs.pop('inner_dim', 64)

        self.pos_dim = kwargs.pop('pos_dim', 30)
        self.ions_dim = kwargs.pop('ions_dim', 6)

        self.init()

    def comment(self):
        return f"TransPro-irt-{self.layer_num}-{self.peptide_embed_dim}-{DROPOUT}-{self.inner_dim}"

    def init(self):
        self.p_embed_token = TokenEmbedding(
            self.peptide_dim + 1, self.peptide_embed_dim)
        self.p_embed_pos = PositionalEncoding_fix(
            self.pos_dim + 1, self.peptide_embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.peptide_embed_dim, nhead=8, dropout=DROPOUT, dim_feedforward=self.inner_dim)
        self.p_trans = nn.TransformerEncoder(
            encoder_layer, num_layers=self.layer_num)
        # self.in_frag = nn.Linear(self.pos_dim-1, self.pos_dim-1)

        self.irt_decoder = nn.Sequential(
            # nn.Linear(
            #     self.peptide_embed_dim, self.peptide_embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(
                self.peptide_embed_dim, 1)
        )
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=DROPOUT)
        self.pool = MaskMean()

    def add_cls(self, peptides: Tensor, mask: Tensor):
        batch_len = peptides.shape[0]
        clss = peptides.new_full((batch_len, 1), 0)
        clss_mask = mask.new_full((batch_len, 1), False)
        return torch.cat([clss, peptides], dim=1).long(), torch.cat([clss_mask, mask], dim=1)

    def forward(self, x, choice="frag"):
        assert choice in ['frag', 'irt', 'both']
        peptides = x['sequence_integer']
        peptides_mask = x['peptide_mask']
        B = peptides.shape[0]

        peptides, peptides_mask = self.add_cls(peptides, peptides_mask)
        # B, S+1

        pepti_embed = self.p_embed_token(peptides)
        pepti_embed = self.p_embed_pos(
            pepti_embed, peptides_mask).transpose(0, 1)
        # B, S+1, D
        trans_embed = self.p_trans(
            pepti_embed, src_key_padding_mask=peptides_mask)
        # irt_embed = self.pool(trans_embed.transpose(0, 1), peptides_mask)
        irt_embed = trans_embed[0]
        irt = self.irt_decoder(irt_embed)
        return irt

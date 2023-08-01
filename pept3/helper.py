import pandas as pd
from turtle import forward
import torch
from torch import nn
import copy
import torch.nn.functional as F
from .tools import *
from . import bio_helper

EPS = 1e-9


class EarlyStop:
    def __init__(self, pat=10) -> None:
        self._best = torch.inf
        self._count = 0
        self._pat = pat

    def step(self, metrics):
        if metrics < self._best:
            self._count = 0
            self._best = metrics
            return False
        else:
            self._count += 1
            self._best = (self._best * self._count +
                          metrics) / (self._count + 1)
            if self._count > self._pat:
                return True
            return False


def spectral_distance(true, pred):
    true_mask = (true >= 0).float()

    pred2com = pred * true_mask
    true2com = true * true_mask

    pred2com = F.normalize(pred2com, dim=1)
    true2com = F.normalize(true2com, dim=1)

    re = torch.sum(pred2com * true2com, dim=-1)
    return torch.mean((2 / torch.pi) * torch.arccos(re))


class InfoNCE(nn.Module):
    def __init__(self, temp=0.1, on_spectral=True, lambdaa=0):
        super().__init__()
        self.temp = temp
        self.on_spectral = on_spectral
        self.labmbda = lambdaa

    def forward(self, true, pred):
        B = true.shape[0]
        if self.on_spectral:
            pred_d = pred.repeat_interleave(B, 0)
            true_d = true.repeat(B, 1)
        else:
            true_d = true.repeat_interleave(B, 0)
            pred_d = pred.repeat(B, 1)
        sas = spectral_angle(true_d, pred_d)
        scores_mat = sas.reshape(B, B)
        nce = scores_mat / self.temp
        labels = torch.arange(
            start=0, end=B, dtype=torch.long, device=nce.device)
        # print([(i.item(), j.item())
        #       for i, j in zip(scores_mat[labels, labels][:10], torch.softmax(nce, dim=1)[labels, labels][:10])])
        loss_nce = F.cross_entropy(nce, labels)

        if self.labmbda > 0:
            loss_sa = spectral_distance(true, pred)
            loss = self.labmbda * loss_sa + (1 - self.labmbda) * loss_nce
            return loss, loss_nce.item(), loss_sa.item()
        else:
            return loss_nce, loss_nce.item(), 0


class FinetunePairSALoss(nn.Module):
    def __init__(self, l1_lambda=0.00005):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.act = nn.Softplus()

    def forward(self, true, pred, neg_true, neg_pred):
        l1_v = (torch.abs(pred).sum(1).mean() +
                torch.abs(neg_pred).sum(1).mean())
        sas = spectral_angle(true, pred)
        sas_neg = spectral_angle(neg_true, neg_pred)
        # base = torch.mean(self.act(sas_neg - sas))
        base = (1 - (sas - sas_neg))
        base[base < 0] = 0
        base = torch.mean(base)
        return base + self.l1_lambda * l1_v, base.item(), l1_v.item()


class FinetuneSALoss(nn.Module):
    def __init__(self, l1_lambda=0.00000, pearson=False):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.if_pearson = pearson

    def forward(self, true, pred, label):
        true_mask = (true >= 0).float()
        pred = pred * true_mask
        l1_v = torch.abs(pred).sum(1).mean()
        if not self.if_pearson:
            sas = spectral_angle(true, pred)
            base = torch.mean((sas - label)**2)
        else:
            pears = pearson_coff(true, pred)
            base = torch.mean((pears - label)**2)
        return base + self.l1_lambda * l1_v, base.item(), l1_v.item()


class FinetuneSALossNoneg(nn.Module):
    def __init__(self, l1_lambda=0.00000, pearson=False):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.if_pearson = pearson

    def filter_negs(self, true, pred, label):
        index = (label > 0)
        return true[index], pred[index], label[index]

    def forward(self, true, pred, label):
        true, pred, label = self.filter_negs(true, pred, label)
        true_mask = (true >= 0).float()
        pred = pred * true_mask
        l1_v = torch.abs(pred).sum(1).mean()
        if not self.if_pearson:
            sas = spectral_angle(true, pred)
            base = torch.mean((sas - label)**2)
        else:
            pears = pearson_coff(true, pred)
            base = torch.mean((pears - label)**2)
        return base + self.l1_lambda * l1_v, base.item(), l1_v.item()


class FinetuneComplexSALoss(nn.Module):
    def __init__(self, l1_lambda=0.0001):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.act = nn.Sigmoid()

    def forward(self, score, label):
        sas = self.act(score)
        label[label < 0] = 0
        base = torch.mean((sas - label)**2)
        return base


class FinetuneRTLoss(nn.Module):
    def __init__(self, gap=0.1):
        super().__init__()
        self.gap = gap

    def forward(self, true, pred, label):
        l1_v = torch.abs(pred).sum(1).mean()
        sas = (true - pred)**2
        base1 = sas[label == 1].mean()
        base2 = (self.gap - sas[label == -1])
        base2[base2 < 0] = 0
        base2 = base2.mean()
        return base1 + base2, base1.item(), base2.item()


class L1Loss(nn.Module):
    def __init__(self, loss_fn, lambdaa=0.001):
        super().__init__()
        self.lambdaa = lambdaa
        self.loss_fn = loss_fn

    def forward(self, true, pred):
        true_mask = (true >= 0).float()
        pred = pred * true_mask
        l1_v = torch.abs(pred).sum(1).mean()
        base = self.loss_fn(true, pred)
        return base + self.lambdaa * l1_v, base.item(), l1_v.item()


def reshape_dims(array, MAX_SEQUENCE=30, ION_TYPES="yb", MAX_FRAG_CHARGE=3):
    n, dims = array.shape
    assert dims == 174
    return array.reshape(
        *[array.shape[0], MAX_SEQUENCE - 1,
            len(ION_TYPES), MAX_FRAG_CHARGE]
    )


def mask_outofrange(array, lengths, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        array[i, lengths[i] - 1:, :, :] = mask
    return array


def mask_outofcharge(array, charges, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        if charges[i] < 3:
            array[i, :, :, charges[i]:] = mask
    return array


def predict_sa(true, pred, data):
    # pred[pred < 0] = 0
    pred = pred / pred.max()
    true = true / true.max()
    B = pred.shape[0]
    lengths = torch.count_nonzero(data['sequence_integer'], dim=1)
    charges = torch.argmax(data["precursor_charge_onehot"], dim=1) + 1

    tide_pred = reshape_dims(pred)
    tide_pred = mask_outofrange(tide_pred, lengths)
    tide_pred = mask_outofcharge(tide_pred, charges)
    pred = tide_pred.view(B, -1)
    return spectral_angle(true, pred), pred


def predict_pearson(true, pred, data):
    pred = pred / pred.max()
    B = pred.shape[0]
    lengths = torch.count_nonzero(data['sequence_integer'], dim=1)
    charges = torch.argmax(data["precursor_charge_onehot"], dim=1) + 1

    tide_pred = reshape_dims(pred)
    tide_pred = mask_outofrange(tide_pred, lengths)
    tide_pred = mask_outofcharge(tide_pred, charges)
    pred = tide_pred.view(B, -1)
    return pearson_coff(true, pred), pred


def predict_sa_scale(true, pred, data, scale):
    pred[pred < 0] = 0
    pred = pred / pred.max()
    B = pred.shape[0]
    lengths = torch.count_nonzero(data['sequence_integer'], dim=1)
    charges = torch.argmax(data["precursor_charge_onehot"], dim=1) + 1

    tide_pred = reshape_dims(pred)
    tide_pred = mask_outofrange(tide_pred, lengths)
    tide_pred = mask_outofcharge(tide_pred, charges)
    pred = tide_pred.view(B, -1)
    return spectral_angle(true, pred, scale=scale), pred


def median_spectral_angle(true, pred):
    re = spectral_angle(true, pred)
    return torch.median(re)


def spectral_angle(true, pred, scale=None):
    true_mask = (true >= 0).float()

    pred2com = pred * true_mask
    true2com = true * true_mask

    pred2com = F.normalize(pred2com)
    if scale is None:
        true2com = F.normalize(true2com)
    else:
        true2com /= (scale + 1e-9)

    re = torch.sum(pred2com * true2com, dim=-1)
    re[re > 1] = 1
    re[re < -1] = -1
    return 1 - (2 / torch.pi) * torch.arccos(re)


def pearson_coff(true, pred):
    true_mask = (true >= 0).float()

    pred2com = pred * true_mask
    true2com = true * true_mask

    pred2com -= torch.mean(pred2com, dim=1).unsqueeze(-1)
    true2com -= torch.mean(true2com, dim=1).unsqueeze(-1)

    pred2com = F.normalize(pred2com, dim=1)
    true2com = F.normalize(true2com, dim=1)

    return torch.sum(pred2com * true2com, dim=-1)
# def spectral_angle_nonormal(true, pred):
#     true_mask = (true >= 0).float()

#     pred2com = pred*true_mask
#     true2com = true*true_mask

#     pred2com = F.normalize(pred2com)
#     true2com = F.normalize(true2com)

#     re = torch.sum(pred2com*true2com, dim=-1)
#     return - re


def mse_distance(true, pred):
    return F.mse_loss(true, pred)


def create_mask(peptides, pad_id=0):
    return (peptides == pad_id)


def set_seed(seed):
    import numpy as np
    import torch
    import random
    import pandas as pd
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def fixed_features(msms_file, raw_dir, save_tab, over_write=False):
    table2save = f"{save_tab}/fixed_features.tab"
    if os.path.exists(table2save) and not over_write:
        return table2save
    name, msms_data = read_msms(
        msms_file)
    msms_data = filter_msms(name, msms_data)
    save2 = os.path.splitext(msms_file)[0] + "_peaks.txt"
    m_r = loc_msms_in_raw(msms_data, raw_dir)
    m_r = sorted(m_r, key=lambda x: int(x[0][name.index("id")]))
    if not os.path.exists(save2):
        print("Ions generating")
        matched_ions_pre = generate_matched_ions(m_r)
        # m_i = generate_peaks_info(matched_ions_pre)
        m_i = [bio_helper.match_all(i, 'yb') for i in tqdm(matched_ions_pre)]
        with open(save2, 'w') as f:
            for pack in m_i:
                peak = ";".join(pack[0])
                intens = ";".join([str(i) for i in pack[1]])
                ratios = str(pack[4])
                scale = str(pack[6])
                f.write('\t'.join([peak, intens, ratios, scale]) + '\n')
        del m_i, matched_ions_pre
    peak_infos = []
    with open(save2) as f:
        for line in f:
            line = line.strip('\n').split('\t')
            peak_infos.append(
                (line[0], line[1],
                 float(line[2]), float(line[3]))
            )
    assert len(m_r) == len(peak_infos)
    need_col = ['id', "Raw file", 'Scan number', "Reverse", "Mass",
                "Sequence", "Charge", "Missed cleavages", "Length", "Mass Error [ppm]",
                "Score", "Delta score", "All modified sequences", "Retention time"]
    i_d = {}
    for c in need_col:
        i_d[c] = name.index(c)
    pack = [(m[0], m[1], p) for m, p in zip(m_r, peak_infos)]
    Features = OrderedDict()

    def add_id(pack):
        Features['SpecId'] = [int(m[0][i_d['id']]) for m in pack]

    def add_scannr(pack):
        Features['ScanNr'] = [hash(
            m[0][i_d['Raw file']] + "|" + m[0][i_d['Scan number']] + "|" + m[0][i_d['id']]) for m in pack]

    def add_label(pack):
        Features['Label'] = [-1 if m[0]
                             [i_d['Reverse']].strip() else 1 for m in pack]

    def add_expmass(pack):
        Features['ExpMass'] = 1000

    def add_mass(pack):
        Features['Mass'] = [float(m[0][i_d['Mass']]) for m in pack]

    def add_peptide(pack):
        Features['Peptide'] = [m[0][i_d['Sequence']].strip("_") for m in pack]

    def add_protein(pack):
        Features['Protein'] = [m[0][i_d['Sequence']].strip("_") for m in pack]

    def add_charge(pack):
        Features['Charge'] = [int(m[0][i_d['Charge']]) for m in pack]

    def add_missedCleavages(pack):
        Features['missedCleavages'] = [
            int(m[0][i_d["Missed cleavages"]]) for m in pack]

    def add_seqlength(pack):
        Features['sequence_length'] = np.array(
            [int(m[0][i_d['Length']]) for m in pack])

    def add_deltaM_ppm(pack):
        Features['deltaM_ppm'] = [
            float(m[0][i_d['Mass Error [ppm]']]) for m in pack]
        Features['deltaM_ppm'] = [0. if math.isnan(
            i) else i for i in Features['deltaM_ppm']]

    def add_absDeltaM_ppm(pack):
        Features['absDeltaM_ppm'] = [
            float(m[0][i_d['Mass Error [ppm]']]) for m in pack]
        Features['absDeltaM_ppm'] = [0. if math.isnan(
            i) else abs(i) for i in Features['absDeltaM_ppm']]

    def add_deltaM_da(pack):
        Features['deltaM_da'] = [
            p / 1e6 * m for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
        ]

    def add_absDeltaM_da(pack):
        Features['absDeltaM_da'] = [
            abs(p / 1e6 * m) for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
        ]

    def add_andromeda(pack):
        Features['andromeda'] = [float(m[0][i_d['Score']]) for m in pack]

    def add_delta_score(pack):
        Features['delta_score'] = [
            float(m[0][i_d["Delta score"]]) for m in pack]

    def add_KR(pack):
        Features['KR'] = [
            sum(map(lambda x: 1 if x in "KR" else 0, m[0][i_d['Sequence']])) for m in pack]

    def add_rt(pack):
        Features['retention_time'] = [
            float(m[0][i_d['Retention time']]) for m in pack]

    def add_collision_energy_aligned_normed(pack):
        Features['collision_energy_aligned_normed'] = 0.33

    def add_p(pack):
        Features['peak_ratio'] = [
            m[2][2] for m in pack
        ]
        Features['peak_scale'] = [
            m[2][3] for m in pack
        ]
        Features['peak_inten'] = [
            m[2][1] for m in pack
        ]
        Features['peak_ions'] = [
            m[2][0] for m in pack
        ]
    add_id(pack)
    add_label(pack)
    add_scannr(pack)
    add_mass(pack)
    add_deltaM_ppm(pack)
    add_deltaM_da(pack)
    add_absDeltaM_ppm(pack)
    add_absDeltaM_da(pack)
    add_missedCleavages(pack)
    add_seqlength(pack)
    add_collision_energy_aligned_normed(pack)
    add_andromeda(pack)
    add_delta_score(pack)
    add_KR(pack)
    add_rt(pack)
    add_charge(pack)
    add_p(pack)
    add_peptide(pack)

    table = pd.DataFrame(Features)
    table.to_csv(table2save, sep='\t', index=False)
    return table2save


def one_pack_all_twofold(msms_file, raw_dir, model1, model2, sample_size=None, irt_model=None, id2remove=None, pearson=False):

    name = read_name(msms_file)
    ions_save = os.path.splitext(msms_file)[0] + "_ions.txt"
    if sample_size is not None:
        ions_save = os.path.splitext(msms_file)[0] + f"_{sample_size}_ions.txt"
    if not os.path.exists(ions_save):
        print("Computing matched ions from scratch", ions_save)
        save_m_r_ions(msms_file, raw_dir, sample_size=sample_size)
    m_r, m_i_delta, m_i = read_m_r_ions(ions_save)
    print(f"Before Len: {len(m_r)}")
    target_index = [i for i in range(len(m_r)) if len(
        m_r[i][0][name.index('Reverse')]) == 0]
    decoy_1_index = [i for i in range(
        len(m_r)) if int(m_r[i][0][name.index('id')]) in id2remove]
    decoy_2_index = [i for i in range(
        len(m_r)) if (len(m_r[i][0][name.index("Reverse")]) == 1 and int(
            m_r[i][0][name.index('id')]) not in id2remove)]

    msms_data_target = [m_r[i][0] for i in target_index]
    msms_data_decoy1 = [m_r[i][0] for i in decoy_1_index]
    msms_data_decoy2 = [m_r[i][0] for i in decoy_2_index]

    # --------------------------------------------
    with torch.no_grad():
        frag_msms_delta = [bio_helper.reverse_annotation(
            *i[:4]) for i in [m_i_delta[i] for i in target_index]]
        data_nce_cand_delta = generate_from_msms_delta(
            msms_data_target, name, nces=33)
        sas_delta1, _ = get_sa_all(
            model1, data_nce_cand_delta, frag_msms_delta, pearson=pearson)
        sas_delta2, _ = get_sa_all(
            model2, data_nce_cand_delta, frag_msms_delta, pearson=pearson)
        sas_delta_target = ((sas_delta1 + sas_delta2) / 2).cpu().numpy()
        # --------------------------------------------
        frag_msms_delta = [bio_helper.reverse_annotation(
            *i[:4]) for i in [m_i_delta[i] for i in decoy_1_index]]
        data_nce_cand_delta = generate_from_msms_delta(
            msms_data_decoy1, name, nces=33)
        sas_delta_decoy1, _ = get_sa_all(
            model2, data_nce_cand_delta, frag_msms_delta, pearson=pearson)
        sas_delta_decoy1 = sas_delta_decoy1.cpu().numpy()
        # --------------------------------------------
        frag_msms_delta = [bio_helper.reverse_annotation(
            *i[:4]) for i in [m_i_delta[i] for i in decoy_2_index]]
        data_nce_cand_delta = generate_from_msms_delta(
            msms_data_decoy2, name, nces=33)
        sas_delta_decoy2, _ = get_sa_all(
            model1, data_nce_cand_delta, frag_msms_delta, pearson=pearson)
        sas_delta_decoy2 = sas_delta_decoy2.cpu().numpy()

        # --------------------------------------------
        frag_msms_target = [bio_helper.reverse_annotation(
            *i[:4]) for i in [m_i[i] for i in target_index]]
        data_nce_cand = generate_from_msms(msms_data_target, name, nces=33)

        sas1, sa_tensors1 = get_sa_all(
            model1, data_nce_cand, frag_msms_target, pearson=pearson)
        sas2, sa_tensors2 = get_sa_all(
            model1, data_nce_cand, frag_msms_target, pearson=pearson)
        sas_target = ((sas1 + sas2) / 2).cpu().numpy()
        sa_tensors_target = ((sa_tensors1 + sa_tensors2) / 2).cpu().numpy()
        # --------------------------------------------
        frag_msms_decoy1 = [bio_helper.reverse_annotation(
            *i[:4]) for i in [m_i[i] for i in decoy_1_index]]
        data_nce_cand = generate_from_msms(
            msms_data_decoy1, name, nces=33)
        sas_decoy1, sa_tensors_decoy1 = get_sa_all(
            model2, data_nce_cand, frag_msms_decoy1, pearson=pearson)
        sas_decoy1 = sas_decoy1.cpu().numpy()
        sa_tensors_decoy1 = sa_tensors_decoy1.cpu().numpy()
        # --------------------------------------------
        frag_msms_decoy2 = [bio_helper.reverse_annotation(
            *i[:4]) for i in [m_i[i] for i in decoy_2_index]]
        data_nce_cand = generate_from_msms(
            msms_data_decoy2, name, nces=33)
        sas_decoy2, sa_tensors_decoy2 = get_sa_all(
            model1, data_nce_cand, frag_msms_decoy2, pearson=pearson)
        sas_decoy2 = sas_decoy2.cpu().numpy()
        sa_tensors_decoy2 = sa_tensors_decoy2.cpu().numpy()
        # --------------------------------------------
        reorder_index = []
        reorder_index.extend(target_index)
        reorder_index.extend(decoy_1_index)
        reorder_index.extend(decoy_2_index)

        m_r = [m_r[i] for i in reorder_index]
        sas = np.concatenate([sas_target, sas_decoy1, sas_decoy2], axis=0)
        sa_tensors = np.concatenate(
            [sa_tensors_target, sa_tensors_decoy1, sa_tensors_decoy2], axis=0)
        sas_delta = np.concatenate(
            [sas_delta_target, sas_delta_decoy1, sas_delta_decoy2], axis=0)
        frag_msms = np.concatenate(
            [frag_msms_target, frag_msms_decoy1, frag_msms_decoy2], axis=0)
        frag_msms = [i.reshape(-1) for i in frag_msms]

        if irt_model is not None:
            msms_data = [m[0] for m in m_r]
            data_nce_cand = generate_from_msms(msms_data, name, nces=33)
            irts = get_irt_all(irt_model, data_nce_cand)
            pack = [(m[0], m[1], sa, sat, sa_d, frag, irt) for m, sa, sat, sa_d,
                    frag, irt in zip(m_r, sas, sa_tensors, sas_delta, frag_msms, irts)]
        else:
            pack = [(m[0], m[1], sa, sat, sa_d, frag) for m, sa, sat, sa_d,
                    frag in zip(m_r, sas, sa_tensors, sas_delta, frag_msms)]
        print(f"Assemle Len: {len(pack)}")
        return pack, name


def fdr_test_twofold(run_model1, run_model2, msms_file, raw_dir, save_tab, sample_size=300000, irt_model=None, need_all=False, id2remove=None, totest=None, pearson=False):
    # pack, msms_name = one_pack_all(
    #     msms_file,
    #     raw_dir,
    #     run_model, sample_size=sample_size, irt_model=irt_model)
    if need_all:
        print("All sprectral used for features")
    pack, msms_name = one_pack_all_twofold(
        msms_file,
        raw_dir,
        run_model1, run_model2, sample_size=sample_size, irt_model=irt_model, id2remove=id2remove, pearson=pearson)

    # %%
    need_col = ['id', "Raw file", 'Scan number', "Reverse", "Mass",
                "Sequence", "Charge", "Missed cleavages", "Length", "Mass Error [ppm]",
                "Score", "Delta score", "All modified sequences", "Retention time"]
    i_d = {}
    for c in need_col:
        i_d[c] = msms_name.index(c)

    # %%
    Features = {}

    def add_id(pack):
        Features['SpecId'] = [int(m[0][i_d['id']]) for m in pack]

    def add_scannr(pack):
        Features['ScanNr'] = [hash(
            m[0][i_d['Raw file']] + "|" + m[0][i_d['Scan number']] + "|" + m[0][i_d['id']]) for m in pack]

    def add_label(pack):
        Features['Label'] = [-1 if m[0]
                             [i_d['Reverse']].strip() else 1 for m in pack]

    def add_expmass(pack):
        Features['ExpMass'] = 1000

    def add_mass(pack):
        Features['Mass'] = [float(m[0][i_d['Mass']]) for m in pack]

    def add_peptide(pack):
        Features['Peptide'] = ["_." + m[0]
                               [i_d['Sequence']].strip("_") + "._" for m in pack]

    def add_protein(pack):
        Features['Protein'] = [m[0][i_d['Sequence']].strip("_") for m in pack]

    def add_charge2(pack):
        Features['Charge2'] = [
            1 if int(m[0][i_d['Charge']]) == 2 else 0 for m in pack]

    def add_charge3(pack):
        Features['Charge3'] = [
            1 if int(m[0][i_d['Charge']]) == 3 else 0 for m in pack]

    def add_missedCleavages(pack):
        Features['missedCleavages'] = [
            int(m[0][i_d["Missed cleavages"]]) for m in pack]

    def add_seqlength(pack):
        Features['sequence_length'] = np.array(
            [int(m[0][i_d['Length']]) for m in pack])

    def add_deltaM_ppm(pack):
        Features['deltaM_ppm'] = [
            float(m[0][i_d['Mass Error [ppm]']]) for m in pack]
        Features['deltaM_ppm'] = [0. if math.isnan(
            i) else i for i in Features['deltaM_ppm']]

    def add_absDeltaM_ppm(pack):
        Features['absDeltaM_ppm'] = [
            float(m[0][i_d['Mass Error [ppm]']]) for m in pack]
        Features['absDeltaM_ppm'] = [0. if math.isnan(
            i) else abs(i) for i in Features['absDeltaM_ppm']]

    def add_deltaM_da(pack):
        Features['deltaM_da'] = [
            p / 1e6 * m for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
        ]

    def add_absDeltaM_da(pack):
        Features['absDeltaM_da'] = [
            abs(p / 1e6 * m) for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
        ]

    def add_sa(pack):
        Features['spectral_angle'] = [float(m[2]) for m in pack]

    def add_delta_sa(pack):
        Features['delta_sa'] = [float(m[2]) - float(m[4]) for m in pack]

    def add_andromeda(pack):
        Features['andromeda'] = [float(m[0][i_d['Score']]) for m in pack]

    def add_delta_score(pack):
        Features['delta_score'] = [
            float(m[0][i_d["Delta score"]]) for m in pack]

    def add_irt(pack):
        Features['irt'] = [float(m[6]) for m in pack]
        Features['retention_time'] = [
            float(m[0][i_d['Retention time']]) for m in pack]

    def add_collision_energy_aligned_normed(pack):
        Features['collision_energy_aligned_normed'] = 0.33

    def add_KR(pack):
        Features['KR'] = [
            sum(map(lambda x: 1 if x in "KR" else 0, m[0][i_d['Sequence']])) for m in pack]

    def add_sprectral(pack):
        spectral_len = len(pack[0][3])
        ids = []
        for i in range(spectral_len):
            Features[f"exp_{i}"] = np.array([m[5][i] for m in pack])
            ids.append(f"exp_{i}")
        for i in range(spectral_len):
            Features[f"pred_{i}"] = np.array([m[3][i] for m in pack])
            ids.append(f"pred_{i}")
        return ids

    def add_ratio(pack):
        Features['count_peak'] = np.array(
            [np.sum(m[5] > 0) for m in pack]) / np.array([len(m[1][3].split(' ')) for m in pack])
        Features['sum_peak'] = np.array([np.sum(
            m[5][m[5] > 0]) for m in pack]) / np.array([sum(map(float, m[1][3].split(' '))) for m in pack])

    def add_pred(pack):
        def b(tensor):
            return tensor.reshape(29, 2, 3)[:, 1, :]

        def y(tensor):
            return tensor.reshape(29, 2, 3)[:, 0, :]

        Features['not_pred_seen'] = [np.sum(m[3][m[5] > 0] == 0) for m in pack]
        Features['not_pred_seen_b'] = [
            np.sum(b(m[3])[b(m[5]) > 0] == 0) for m in pack]
        Features['not_pred_seen_y'] = [
            np.sum(y(m[3])[y(m[5]) > 0] == 0) for m in pack]
        Features['pred_nonZero_fragments'] = [np.sum(m[3] > 0) for m in pack]
        Features['pred_nonZero_b'] = [np.sum(b(m[3]) > 0) for m in pack]
        Features['pred_nonZero_y'] = [np.sum(y(m[3]) > 0) for m in pack]
        Features['pred_not_seen'] = [np.sum(m[5][m[3] > 0] == 0) for m in pack]
        Features['pred_not_seen_b'] = [
            np.sum(b(m[5])[b(m[3]) > 0] == 0) for m in pack]
        Features['pred_not_seen_y'] = [
            np.sum(y(m[5])[y(m[3]) > 0] == 0) for m in pack]
        Features['pred_seen_nonzero'] = [
            np.sum(m[5][m[3] > 0] > 0) for m in pack]
        Features['pred_seen_nonzero_y'] = [
            np.sum(y(m[5])[y(m[3]) > 0] > 0) for m in pack]
        Features['pred_seen_nonzero_b'] = [
            np.sum(b(m[5])[b(m[3]) > 0] > 0) for m in pack]
        Features['pred_seen_zero'] = [
            np.sum(m[5][m[3] == 0] == 0) for m in pack]
        Features['pred_seen_zero_b'] = [
            np.sum(b(m[5])[b(m[3]) == 0] == 0) for m in pack]
        Features['pred_seen_zero_y'] = [
            np.sum(y(m[5])[y(m[3]) == 0] == 0) for m in pack]
        Features['raw_nonZero_fragments'] = [np.sum(m[5] > 0) for m in pack]
        Features['raw_nonZero_b'] = [np.sum(b(m[5]) > 0) for m in pack]
        Features['raw_nonZero_y'] = [np.sum(y(m[5]) > 0) for m in pack]

        theoretically = Features['sequence_length'] * 2 * \
            np.array([int(m[0][i_d['Charge']]) for m in pack]) + 1e-9
        Features['rel_not_pred_seen'] = np.array(
            Features['not_pred_seen']) / theoretically
        Features['rel_not_pred_seen_b'] = np.array(
            Features['not_pred_seen_b']) / theoretically * 2
        Features['rel_not_pred_seen_y'] = np.array(
            Features['not_pred_seen_y']) / theoretically * 2
        Features['rel_pred_nonZero_b'] = np.array(
            Features['pred_nonZero_b']) / theoretically * 2
        Features['rel_pred_nonZero_y'] = np.array(
            Features['pred_nonZero_y']) / theoretically * 2
        Features['rel_pred_not_seen'] = np.array(
            Features['pred_not_seen']) / theoretically
        Features['rel_pred_not_seen_b'] = np.array(
            Features['pred_not_seen_b']) / theoretically * 2
        Features['rel_pred_not_seen_y'] = np.array(
            Features['pred_not_seen_y']) / theoretically * 2
        Features['rel_pred_seen_nonzero'] = np.array(
            Features['pred_seen_nonzero']) / theoretically
        Features['rel_pred_seen_nonzero_b'] = np.array(
            Features['pred_seen_nonzero_b']) / theoretically * 2
        Features['rel_pred_seen_nonzero_y'] = np.array(
            Features['pred_seen_nonzero_y']) / theoretically * 2
        Features['rel_pred_seen_zero'] = np.array(
            Features['pred_seen_zero']) / theoretically
        Features['rel_pred_seen_zero_b'] = np.array(
            Features['pred_seen_zero_b']) / theoretically * 2
        Features['rel_pred_seen_zero_y'] = np.array(
            Features['pred_seen_zero_y']) / theoretically * 2
        Features['rel_raw_nonZero_fragments'] = np.array(
            Features['raw_nonZero_fragments']) / theoretically
        Features['rel_raw_nonZero_b'] = np.array(
            Features['raw_nonZero_b']) / theoretically * 2
        Features['rel_raw_nonZero_y'] = np.array(
            Features['raw_nonZero_y']) / theoretically * 2

        Features['relpred_not_pred_seen2pred_nonZero_fragments'] = np.array(
            Features['not_pred_seen']) / (np.array(Features['pred_nonZero_fragments']) + 1e-9)
        Features['relpred_not_pred_seen_b2pred_nonZero_b'] = np.array(
            Features['not_pred_seen_b']) / (np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_not_pred_seen_y2pred_nonZero_y'] = np.array(
            Features['not_pred_seen_y']) / (np.array(Features['pred_nonZero_y']) + 1e-9)
        Features['relpred_pred_not_seen_b2pred_nonZero_b'] = np.array(
            Features['pred_not_seen_b']) / (np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_pred_not_seen_y2pred_nonZero_y'] = np.array(
            Features['pred_not_seen_y']) / (np.array(Features['pred_nonZero_y']) + 1e-9)

        Features['relpred_pred_not_seen2pred_nonZero_fragments'] = np.array(
            Features['pred_not_seen']) / (np.array(Features['pred_nonZero_fragments']) + 1e-9)
        Features['relpred_pred_seen_nonzero_b2pred_nonZero_b'] = np.array(
            Features['pred_seen_nonzero_b']) / (np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_pred_seen_nonzero_y2pred_nonZero_y'] = np.array(
            Features['pred_seen_nonzero_y']) / (np.array(Features['pred_nonZero_y']) + 1e-9)

        Features['relpred_pred_seen_nonzero2pred_nonZero_fragments'] = np.array(
            Features['pred_seen_nonzero']) / (np.array(Features['pred_nonZero_fragments']) + 1e-9)
        Features['relpred_pred_seen_zero_b2pred_nonZero_b'] = np.array(
            Features['pred_seen_zero_b']) / (np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_pred_seen_zero_y2pred_nonZero_y'] = np.array(
            Features['pred_seen_zero_y']) / (np.array(Features['pred_nonZero_y']) + 1e-9)

        Features['relpred_pred_seen_zero2pred_nonZero_fragments'] = np.array(
            Features['pred_seen_zero']) / (np.array(Features['pred_nonZero_fragments']) + 1e-9)

    # %%
    add_id(pack)
    add_label(pack)
    add_scannr(pack)
    add_expmass(pack)
    add_mass(pack)
    add_deltaM_ppm(pack)
    add_deltaM_da(pack)
    add_absDeltaM_ppm(pack)
    add_absDeltaM_da(pack)
    add_missedCleavages(pack)
    add_seqlength(pack)
    add_andromeda(pack)
    add_delta_score(pack)

    add_charge2(pack)
    add_charge3(pack)

    add_peptide(pack)
    add_protein(pack)

    add_sa(pack)
    add_delta_sa(pack)

    add_KR(pack)
    add_collision_energy_aligned_normed(pack)
    add_pred(pack)
    add_ratio(pack)
    # %%
    table = pd.DataFrame(Features)

    if totest is None or "andromeda" in totest:
        order_and = "SpecId  Label   ScanNr  ExpMass Mass    deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da    missedCleavages        sequence_length andromeda       delta_score     Charge2 Charge3 Peptide Protein".split()
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/andromeda.tab", sep='\t', index=False)
    if totest is None or "sa" in totest:
        order_and = "SpecId  Label   ScanNr  ExpMass Mass    deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da    missedCleavages        sequence_length spectral_angle delta_sa  Charge2 Charge3 Peptide Protein".split()
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/sa.tab", sep='\t', index=False)
    if totest is None or "combined" in totest:
        order_and = "SpecId  Label ScanNr  ExpMass Mass deltaM_ppm deltaM_da absDeltaM_ppm absDeltaM_da missedCleavages sequence_length spectral_angle  delta_sa andromeda delta_score Charge2 Charge3 Peptide Protein".split()
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/combined.tab", sep='\t', index=False)

    if totest is None or "prosit" in totest:
        order_and = "SpecId  Label   ScanNr  ExpMass Mass    deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da  missedCleavages  sequence_length collision_energy_aligned_normed spectral_angle  KR      raw_nonZero_fragments  raw_nonZero_y   raw_nonZero_b   pred_nonZero_fragments  pred_nonZero_y  pred_nonZero_b  pred_not_seen  pred_not_seen_y pred_not_seen_b pred_seen_zero  pred_seen_zero_y        pred_seen_zero_b      pred_seen_nonzero        pred_seen_nonzero_y     pred_seen_nonzero_b     not_pred_seen   not_pred_seen_y not_pred_seen_b rel_pred_nonZero_y      rel_pred_nonZero_b      rel_pred_not_seen       rel_pred_not_seen_y    rel_pred_not_seen_b     rel_pred_seen_zero      rel_pred_seen_zero_y    rel_pred_seen_zero_b  rel_pred_seen_nonzero    rel_pred_seen_nonzero_y rel_pred_seen_nonzero_b rel_not_pred_seen       rel_not_pred_seen_y    rel_not_pred_seen_b     relpred_pred_not_seen2pred_nonZero_fragments    relpred_pred_not_seen_y2pred_nonZero_y relpred_pred_not_seen_b2pred_nonZero_b  relpred_pred_seen_zero2pred_nonZero_fragments  relpred_pred_seen_zero_y2pred_nonZero_y relpred_pred_seen_zero_b2pred_nonZero_b relpred_pred_seen_nonzero2pred_nonZero_fragments       relpred_pred_seen_nonzero_y2pred_nonZero_y      relpred_pred_seen_nonzero_b2pred_nonZero_b     relpred_not_pred_seen2pred_nonZero_fragments    relpred_not_pred_seen_y2pred_nonZero_y relpred_not_pred_seen_b2pred_nonZero_b  rel_raw_nonZero_b       rel_raw_nonZero_y     rel_raw_nonZero_fragments      Charge2 Charge3 Peptide Protein".split()
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/prosit.tab", sep='\t', index=False)
    if totest is None or "prosit_combined" in totest:
        order_and = "SpecId  Label   ScanNr  ExpMass Mass    deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da  missedCleavages  sequence_length collision_energy_aligned_normed spectral_angle delta_sa andromeda delta_score  KR      raw_nonZero_fragments  raw_nonZero_y   raw_nonZero_b   pred_nonZero_fragments  pred_nonZero_y  pred_nonZero_b  pred_not_seen  pred_not_seen_y pred_not_seen_b pred_seen_zero  pred_seen_zero_y        pred_seen_zero_b      pred_seen_nonzero        pred_seen_nonzero_y     pred_seen_nonzero_b     not_pred_seen   not_pred_seen_y not_pred_seen_b rel_pred_nonZero_y      rel_pred_nonZero_b      rel_pred_not_seen       rel_pred_not_seen_y    rel_pred_not_seen_b     rel_pred_seen_zero      rel_pred_seen_zero_y    rel_pred_seen_zero_b  rel_pred_seen_nonzero    rel_pred_seen_nonzero_y rel_pred_seen_nonzero_b rel_not_pred_seen       rel_not_pred_seen_y    rel_not_pred_seen_b     relpred_pred_not_seen2pred_nonZero_fragments    relpred_pred_not_seen_y2pred_nonZero_y relpred_pred_not_seen_b2pred_nonZero_b  relpred_pred_seen_zero2pred_nonZero_fragments  relpred_pred_seen_zero_y2pred_nonZero_y relpred_pred_seen_zero_b2pred_nonZero_b relpred_pred_seen_nonzero2pred_nonZero_fragments       relpred_pred_seen_nonzero_y2pred_nonZero_y      relpred_pred_seen_nonzero_b2pred_nonZero_b     relpred_not_pred_seen2pred_nonZero_fragments    relpred_not_pred_seen_y2pred_nonZero_y relpred_not_pred_seen_b2pred_nonZero_b  rel_raw_nonZero_b       rel_raw_nonZero_y     rel_raw_nonZero_fragments      Charge2 Charge3 Peptide Protein".split()
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/prosit_combined.tab", sep='\t', index=False)
    if totest is None or "prosit_ratio" in totest:
        order_and = "SpecId  Label   ScanNr  ExpMass Mass    deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da count_peak sum_peak  missedCleavages  sequence_length collision_energy_aligned_normed spectral_angle delta_sa andromeda delta_score  KR      raw_nonZero_fragments  raw_nonZero_y   raw_nonZero_b   pred_nonZero_fragments  pred_nonZero_y  pred_nonZero_b  pred_not_seen  pred_not_seen_y pred_not_seen_b pred_seen_zero  pred_seen_zero_y        pred_seen_zero_b      pred_seen_nonzero        pred_seen_nonzero_y     pred_seen_nonzero_b     not_pred_seen   not_pred_seen_y not_pred_seen_b rel_pred_nonZero_y      rel_pred_nonZero_b      rel_pred_not_seen       rel_pred_not_seen_y    rel_pred_not_seen_b     rel_pred_seen_zero      rel_pred_seen_zero_y    rel_pred_seen_zero_b  rel_pred_seen_nonzero    rel_pred_seen_nonzero_y rel_pred_seen_nonzero_b rel_not_pred_seen       rel_not_pred_seen_y    rel_not_pred_seen_b     relpred_pred_not_seen2pred_nonZero_fragments    relpred_pred_not_seen_y2pred_nonZero_y relpred_pred_not_seen_b2pred_nonZero_b  relpred_pred_seen_zero2pred_nonZero_fragments  relpred_pred_seen_zero_y2pred_nonZero_y relpred_pred_seen_zero_b2pred_nonZero_b relpred_pred_seen_nonzero2pred_nonZero_fragments       relpred_pred_seen_nonzero_y2pred_nonZero_y      relpred_pred_seen_nonzero_b2pred_nonZero_b     relpred_not_pred_seen2pred_nonZero_fragments    relpred_not_pred_seen_y2pred_nonZero_y relpred_not_pred_seen_b2pred_nonZero_b  rel_raw_nonZero_b       rel_raw_nonZero_y     rel_raw_nonZero_fragments      Charge2 Charge3 Peptide Protein".split()
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/prosit_ratio.tab", sep='\t', index=False)

    if irt_model is not None:
        del table
        add_irt(pack)
        table = pd.DataFrame(Features)
        if totest is None or "sa_rich" in totest:
            order_and = "SpecId  Label ScanNr  ExpMass Mass deltaM_ppm deltaM_da absDeltaM_ppm absDeltaM_da count_peak sum_peak missedCleavages sequence_length spectral_angle  delta_sa andromeda delta_score Charge2 Charge3 Peptide Protein".split()
            andre_table = table[order_and]
            andre_table.to_csv(
                f"{save_tab}/sa_rich.tab", sep='\t', index=False)
        if totest is None or "prosit_best" in totest:
            order_and = "SpecId  Label   ScanNr  ExpMass Mass retention_time irt count_peak sum_peak deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da  missedCleavages  sequence_length collision_energy_aligned_normed spectral_angle delta_sa andromeda delta_score  KR      raw_nonZero_fragments  raw_nonZero_y   raw_nonZero_b   pred_nonZero_fragments  pred_nonZero_y  pred_nonZero_b  pred_not_seen  pred_not_seen_y pred_not_seen_b pred_seen_zero  pred_seen_zero_y        pred_seen_zero_b      pred_seen_nonzero        pred_seen_nonzero_y     pred_seen_nonzero_b     not_pred_seen   not_pred_seen_y not_pred_seen_b rel_pred_nonZero_y      rel_pred_nonZero_b      rel_pred_not_seen       rel_pred_not_seen_y    rel_pred_not_seen_b     rel_pred_seen_zero      rel_pred_seen_zero_y    rel_pred_seen_zero_b  rel_pred_seen_nonzero    rel_pred_seen_nonzero_y rel_pred_seen_nonzero_b rel_not_pred_seen       rel_not_pred_seen_y    rel_not_pred_seen_b     relpred_pred_not_seen2pred_nonZero_fragments    relpred_pred_not_seen_y2pred_nonZero_y relpred_pred_not_seen_b2pred_nonZero_b  relpred_pred_seen_zero2pred_nonZero_fragments  relpred_pred_seen_zero_y2pred_nonZero_y relpred_pred_seen_zero_b2pred_nonZero_b relpred_pred_seen_nonzero2pred_nonZero_fragments       relpred_pred_seen_nonzero_y2pred_nonZero_y      relpred_pred_seen_nonzero_b2pred_nonZero_b     relpred_not_pred_seen2pred_nonZero_fragments    relpred_not_pred_seen_y2pred_nonZero_y relpred_not_pred_seen_b2pred_nonZero_b  rel_raw_nonZero_b       rel_raw_nonZero_y     rel_raw_nonZero_fragments      Charge2 Charge3 Peptide Protein".split()
            andre_table = table[order_and]
            andre_table.to_csv(
                f"{save_tab}/prosit_best.tab", sep='\t', index=False)
    if need_all:
        Features = {}
        add_id(pack)
        add_label(pack)
        add_scannr(pack)
        add_expmass(pack)
        add_mass(pack)
        add_deltaM_ppm(pack)
        add_deltaM_da(pack)
        add_absDeltaM_ppm(pack)
        add_absDeltaM_da(pack)
        add_missedCleavages(pack)
        add_seqlength(pack)
        add_andromeda(pack)
        add_delta_score(pack)

        add_charge2(pack)
        add_charge3(pack)

        add_peptide(pack)
        add_protein(pack)
        add_sa(pack)
        add_delta_sa(pack)

        add_KR(pack)
        add_collision_energy_aligned_normed(pack)
        spect_ids = add_sprectral(pack)
        table = pd.DataFrame(Features)

        order_and = "SpecId  Label ScanNr  ExpMass Mass deltaM_ppm deltaM_da absDeltaM_ppm absDeltaM_da missedCleavages sequence_length spectral_angle  delta_sa andromeda delta_score Charge2 Charge3 Peptide Protein".split()
        order_and = order_and[:11] + spect_ids + order_and[11:]
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/spectral_all.tab", sep='\t', index=False)


def one_pack_all_nfold(msms_file,
                       raw_dir,
                       models,
                       ids2selects,
                       sample_size=None,
                       irt_model=None,
                       pearson=False):
    assert len(models) == len(ids2selects), "Mismatch models and ids"
    print(f">>Running {len(models)} folds-------------")
    name = read_name(msms_file)
    ions_save = os.path.splitext(msms_file)[0] + "_ions.txt"
    if sample_size is not None:
        ions_save = os.path.splitext(msms_file)[0] + f"_{sample_size}_ions.txt"
    if not os.path.exists(ions_save):
        print("Computing matched ions from scratch", ions_save)
        save_m_r_ions(msms_file, raw_dir, sample_size=sample_size)
    m_r, m_i_delta, m_i = read_m_r_ions(ions_save)
    
    with torch.no_grad():
        delta_sas = []
        sas = []
        sas_tensor = []
        frag_msms_list = []
        reorder_index = []
        for now_i, (model, ids) in enumerate(zip(models, ids2selects)):
            print(f">> generating fold-{now_i} [{len(ids)}]...")
            ids_all = set(ids.values)
            ids_index = [i for i in range(len(m_r)) if int(m_r[i][0][name.index('id')]) in ids_all]
            msms_data = [m_r[i][0] for i in ids_index]
            
            frag_msms_delta = [bio_helper.reverse_annotation(*i[:4]) for i in [m_i_delta[i] for i in ids_index]]
            data_nce_cand_delta = generate_from_msms_delta(msms_data, name, nces=33)    
            sas_delta, _ = get_sa_all(model, data_nce_cand_delta, frag_msms_delta, pearson=pearson)    
            delta_sas.append(sas_delta)
            
            frag_msms = [bio_helper.reverse_annotation(*i[:4]) for i in [m_i[i] for i in ids_index]]
            data_nce_cand = generate_from_msms(msms_data, name, nces=33)    
            sa, sa_tensor = get_sa_all(model, data_nce_cand, frag_msms, pearson=pearson)    
            
            reorder_index.extend(ids_index)
            sas.append(sa)
            sas_tensor.append(sa_tensor)
            frag_msms_list.append(frag_msms)
        m_r = [m_r[i] for i in reorder_index]
        sas = np.concatenate(sas, axis=0)
        sas_delta = np.concatenate(delta_sas, axis=0)
        sas_tensor = np.concatenate(sas_tensor, axis=0)
        frag_msms = np.concatenate(frag_msms_list, axis=0)
        frag_msms = [i.reshape(-1) for i in frag_msms]
        if irt_model is not None:
            msms_data = [m[0] for m in m_r]
            data_nce_cand = generate_from_msms(msms_data, name, nces=33)
            irts = get_irt_all(irt_model, data_nce_cand)
            pack = [(m[0], m[1], sa, sat, sa_d, frag, irt) for m, sa, sat, sa_d,
                    frag, irt in zip(m_r, sas, sas_tensor, sas_delta, frag_msms, irts)]
        else:
            pack = [(m[0], m[1], sa, sat, sa_d, frag) for m, sa, sat, sa_d,
                    frag in zip(m_r, sas, sas_tensor, sas_delta, frag_msms)]
    return pack, name

def fdr_test_nfold(models, 
                   msms_file, 
                   raw_dir, save_tab, 
                   id2selects,
                   sample_size=300000, irt_model=None, need_all=False, totest=None, pearson=False):
    # pack, msms_name = one_pack_all(
    #     msms_file,
    #     raw_dir,
    #     run_model, sample_size=sample_size, irt_model=irt_model)
    if need_all:
        print("All sprectral used for features")
    pack, msms_name = one_pack_all_nfold(
        msms_file,
        raw_dir,
        models, 
        id2selects,
        sample_size=sample_size, irt_model=irt_model, pearson=pearson)

    # %%
    need_col = ['id', "Raw file", 'Scan number', "Reverse", "Mass",
                "Sequence", "Charge", "Missed cleavages", "Length", "Mass Error [ppm]",
                "Score", "Delta score", "All modified sequences", "Retention time"]
    i_d = {}
    for c in need_col:
        i_d[c] = msms_name.index(c)

    # %%
    Features = {}

    def add_id(pack):
        Features['SpecId'] = [int(m[0][i_d['id']]) for m in pack]

    def add_scannr(pack):
        Features['ScanNr'] = [hash(
            f"{m[0][i_d['Raw file']]}|{m[0][i_d['Scan number']]}|{m[0][i_d['id']]}") for m in pack]

    def add_label(pack):
        Features['Label'] = [-1 if m[0]
                             [i_d['Reverse']].strip() else 1 for m in pack]

    def add_expmass(pack):
        Features['ExpMass'] = 1000

    def add_mass(pack):
        Features['Mass'] = [float(m[0][i_d['Mass']]) for m in pack]

    def add_peptide(pack):
        Features['Peptide'] = ["_." + m[0]
                               [i_d['Sequence']].strip("_") + "._" for m in pack]

    def add_protein(pack):
        Features['Protein'] = [m[0][i_d['Sequence']].strip("_") for m in pack]

    def add_charge2(pack):
        Features['Charge2'] = [
            1 if int(m[0][i_d['Charge']]) == 2 else 0 for m in pack]

    def add_charge3(pack):
        Features['Charge3'] = [
            1 if int(m[0][i_d['Charge']]) == 3 else 0 for m in pack]

    def add_missedCleavages(pack):
        Features['missedCleavages'] = [
            int(m[0][i_d["Missed cleavages"]]) for m in pack]

    def add_seqlength(pack):
        Features['sequence_length'] = np.array(
            [int(m[0][i_d['Length']]) for m in pack])

    def add_deltaM_ppm(pack):
        Features['deltaM_ppm'] = [
            float(m[0][i_d['Mass Error [ppm]']]) for m in pack]
        Features['deltaM_ppm'] = [0. if math.isnan(
            i) else i for i in Features['deltaM_ppm']]

    def add_absDeltaM_ppm(pack):
        Features['absDeltaM_ppm'] = [
            float(m[0][i_d['Mass Error [ppm]']]) for m in pack]
        Features['absDeltaM_ppm'] = [0. if math.isnan(
            i) else abs(i) for i in Features['absDeltaM_ppm']]

    def add_deltaM_da(pack):
        Features['deltaM_da'] = [
            p / 1e6 * m for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
        ]

    def add_absDeltaM_da(pack):
        Features['absDeltaM_da'] = [
            abs(p / 1e6 * m) for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
        ]

    def add_sa(pack):
        Features['spectral_angle'] = [float(m[2]) for m in pack]

    def add_delta_sa(pack):
        Features['delta_sa'] = [float(m[2]) - float(m[4]) for m in pack]

    def add_andromeda(pack):
        Features['andromeda'] = [float(m[0][i_d['Score']]) for m in pack]

    def add_delta_score(pack):
        Features['delta_score'] = [
            float(m[0][i_d["Delta score"]]) for m in pack]

    def add_irt(pack):
        Features['irt'] = [float(m[6]) for m in pack]
        Features['retention_time'] = [
            float(m[0][i_d['Retention time']]) for m in pack]

    def add_collision_energy_aligned_normed(pack):
        Features['collision_energy_aligned_normed'] = 0.33

    def add_KR(pack):
        Features['KR'] = [
            sum(map(lambda x: 1 if x in "KR" else 0, m[0][i_d['Sequence']])) for m in pack]

    def add_sprectral(pack):
        spectral_len = len(pack[0][3])
        ids = []
        for i in range(spectral_len):
            Features[f"exp_{i}"] = np.array([m[5][i] for m in pack])
            ids.append(f"exp_{i}")
        for i in range(spectral_len):
            Features[f"pred_{i}"] = np.array([m[3][i] for m in pack])
            ids.append(f"pred_{i}")
        return ids

    def add_ratio(pack):
        Features['count_peak'] = np.array(
            [np.sum(m[5] > 0) for m in pack]) / np.array([len(m[1][3].split(' ')) for m in pack])
        Features['sum_peak'] = np.array([np.sum(
            m[5][m[5] > 0]) for m in pack]) / np.array([sum(map(float, m[1][3].split(' '))) for m in pack])

    def add_pred(pack):
        def b(tensor):
            return tensor.reshape(29, 2, 3)[:, 1, :]

        def y(tensor):
            return tensor.reshape(29, 2, 3)[:, 0, :]

        Features['not_pred_seen'] = [np.sum(m[3][m[5] > 0] == 0) for m in pack]
        Features['not_pred_seen_b'] = [
            np.sum(b(m[3])[b(m[5]) > 0] == 0) for m in pack]
        Features['not_pred_seen_y'] = [
            np.sum(y(m[3])[y(m[5]) > 0] == 0) for m in pack]
        Features['pred_nonZero_fragments'] = [np.sum(m[3] > 0) for m in pack]
        Features['pred_nonZero_b'] = [np.sum(b(m[3]) > 0) for m in pack]
        Features['pred_nonZero_y'] = [np.sum(y(m[3]) > 0) for m in pack]
        Features['pred_not_seen'] = [np.sum(m[5][m[3] > 0] == 0) for m in pack]
        Features['pred_not_seen_b'] = [
            np.sum(b(m[5])[b(m[3]) > 0] == 0) for m in pack]
        Features['pred_not_seen_y'] = [
            np.sum(y(m[5])[y(m[3]) > 0] == 0) for m in pack]
        Features['pred_seen_nonzero'] = [
            np.sum(m[5][m[3] > 0] > 0) for m in pack]
        Features['pred_seen_nonzero_y'] = [
            np.sum(y(m[5])[y(m[3]) > 0] > 0) for m in pack]
        Features['pred_seen_nonzero_b'] = [
            np.sum(b(m[5])[b(m[3]) > 0] > 0) for m in pack]
        Features['pred_seen_zero'] = [
            np.sum(m[5][m[3] == 0] == 0) for m in pack]
        Features['pred_seen_zero_b'] = [
            np.sum(b(m[5])[b(m[3]) == 0] == 0) for m in pack]
        Features['pred_seen_zero_y'] = [
            np.sum(y(m[5])[y(m[3]) == 0] == 0) for m in pack]
        Features['raw_nonZero_fragments'] = [np.sum(m[5] > 0) for m in pack]
        Features['raw_nonZero_b'] = [np.sum(b(m[5]) > 0) for m in pack]
        Features['raw_nonZero_y'] = [np.sum(y(m[5]) > 0) for m in pack]

        theoretically = Features['sequence_length'] * 2 * \
            np.array([int(m[0][i_d['Charge']]) for m in pack]) + 1e-9
        Features['rel_not_pred_seen'] = np.array(
            Features['not_pred_seen']) / theoretically
        Features['rel_not_pred_seen_b'] = np.array(
            Features['not_pred_seen_b']) / theoretically * 2
        Features['rel_not_pred_seen_y'] = np.array(
            Features['not_pred_seen_y']) / theoretically * 2
        Features['rel_pred_nonZero_b'] = np.array(
            Features['pred_nonZero_b']) / theoretically * 2
        Features['rel_pred_nonZero_y'] = np.array(
            Features['pred_nonZero_y']) / theoretically * 2
        Features['rel_pred_not_seen'] = np.array(
            Features['pred_not_seen']) / theoretically
        Features['rel_pred_not_seen_b'] = np.array(
            Features['pred_not_seen_b']) / theoretically * 2
        Features['rel_pred_not_seen_y'] = np.array(
            Features['pred_not_seen_y']) / theoretically * 2
        Features['rel_pred_seen_nonzero'] = np.array(
            Features['pred_seen_nonzero']) / theoretically
        Features['rel_pred_seen_nonzero_b'] = np.array(
            Features['pred_seen_nonzero_b']) / theoretically * 2
        Features['rel_pred_seen_nonzero_y'] = np.array(
            Features['pred_seen_nonzero_y']) / theoretically * 2
        Features['rel_pred_seen_zero'] = np.array(
            Features['pred_seen_zero']) / theoretically
        Features['rel_pred_seen_zero_b'] = np.array(
            Features['pred_seen_zero_b']) / theoretically * 2
        Features['rel_pred_seen_zero_y'] = np.array(
            Features['pred_seen_zero_y']) / theoretically * 2
        Features['rel_raw_nonZero_fragments'] = np.array(
            Features['raw_nonZero_fragments']) / theoretically
        Features['rel_raw_nonZero_b'] = np.array(
            Features['raw_nonZero_b']) / theoretically * 2
        Features['rel_raw_nonZero_y'] = np.array(
            Features['raw_nonZero_y']) / theoretically * 2

        Features['relpred_not_pred_seen2pred_nonZero_fragments'] = np.array(
            Features['not_pred_seen']) / (np.array(Features['pred_nonZero_fragments']) + 1e-9)
        Features['relpred_not_pred_seen_b2pred_nonZero_b'] = np.array(
            Features['not_pred_seen_b']) / (np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_not_pred_seen_y2pred_nonZero_y'] = np.array(
            Features['not_pred_seen_y']) / (np.array(Features['pred_nonZero_y']) + 1e-9)
        Features['relpred_pred_not_seen_b2pred_nonZero_b'] = np.array(
            Features['pred_not_seen_b']) / (np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_pred_not_seen_y2pred_nonZero_y'] = np.array(
            Features['pred_not_seen_y']) / (np.array(Features['pred_nonZero_y']) + 1e-9)

        Features['relpred_pred_not_seen2pred_nonZero_fragments'] = np.array(
            Features['pred_not_seen']) / (np.array(Features['pred_nonZero_fragments']) + 1e-9)
        Features['relpred_pred_seen_nonzero_b2pred_nonZero_b'] = np.array(
            Features['pred_seen_nonzero_b']) / (np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_pred_seen_nonzero_y2pred_nonZero_y'] = np.array(
            Features['pred_seen_nonzero_y']) / (np.array(Features['pred_nonZero_y']) + 1e-9)

        Features['relpred_pred_seen_nonzero2pred_nonZero_fragments'] = np.array(
            Features['pred_seen_nonzero']) / (np.array(Features['pred_nonZero_fragments']) + 1e-9)
        Features['relpred_pred_seen_zero_b2pred_nonZero_b'] = np.array(
            Features['pred_seen_zero_b']) / (np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_pred_seen_zero_y2pred_nonZero_y'] = np.array(
            Features['pred_seen_zero_y']) / (np.array(Features['pred_nonZero_y']) + 1e-9)

        Features['relpred_pred_seen_zero2pred_nonZero_fragments'] = np.array(
            Features['pred_seen_zero']) / (np.array(Features['pred_nonZero_fragments']) + 1e-9)

    # %%
    add_id(pack)
    add_label(pack)
    add_scannr(pack)
    add_expmass(pack)
    add_mass(pack)
    add_deltaM_ppm(pack)
    add_deltaM_da(pack)
    add_absDeltaM_ppm(pack)
    add_absDeltaM_da(pack)
    add_missedCleavages(pack)
    add_seqlength(pack)
    add_andromeda(pack)
    add_delta_score(pack)

    add_charge2(pack)
    add_charge3(pack)

    add_peptide(pack)
    add_protein(pack)

    add_sa(pack)
    add_delta_sa(pack)

    add_KR(pack)
    add_collision_energy_aligned_normed(pack)
    add_pred(pack)
    add_ratio(pack)
    # %%
    table = pd.DataFrame(Features)

    if totest is None or "andromeda" in totest:
        order_and = "SpecId  Label   ScanNr  ExpMass Mass    deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da    missedCleavages        sequence_length andromeda       delta_score     Charge2 Charge3 Peptide Protein".split()
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/andromeda.tab", sep='\t', index=False)
    if totest is None or "sa" in totest:
        order_and = "SpecId  Label   ScanNr  ExpMass Mass    deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da    missedCleavages        sequence_length spectral_angle delta_sa  Charge2 Charge3 Peptide Protein".split()
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/sa.tab", sep='\t', index=False)
    if totest is None or "combined" in totest:
        order_and = "SpecId  Label ScanNr  ExpMass Mass deltaM_ppm deltaM_da absDeltaM_ppm absDeltaM_da missedCleavages sequence_length spectral_angle  delta_sa andromeda delta_score Charge2 Charge3 Peptide Protein".split()
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/combined.tab", sep='\t', index=False)

    if totest is None or "prosit" in totest:
        order_and = "SpecId  Label   ScanNr  ExpMass Mass    deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da  missedCleavages  sequence_length collision_energy_aligned_normed spectral_angle  KR      raw_nonZero_fragments  raw_nonZero_y   raw_nonZero_b   pred_nonZero_fragments  pred_nonZero_y  pred_nonZero_b  pred_not_seen  pred_not_seen_y pred_not_seen_b pred_seen_zero  pred_seen_zero_y        pred_seen_zero_b      pred_seen_nonzero        pred_seen_nonzero_y     pred_seen_nonzero_b     not_pred_seen   not_pred_seen_y not_pred_seen_b rel_pred_nonZero_y      rel_pred_nonZero_b      rel_pred_not_seen       rel_pred_not_seen_y    rel_pred_not_seen_b     rel_pred_seen_zero      rel_pred_seen_zero_y    rel_pred_seen_zero_b  rel_pred_seen_nonzero    rel_pred_seen_nonzero_y rel_pred_seen_nonzero_b rel_not_pred_seen       rel_not_pred_seen_y    rel_not_pred_seen_b     relpred_pred_not_seen2pred_nonZero_fragments    relpred_pred_not_seen_y2pred_nonZero_y relpred_pred_not_seen_b2pred_nonZero_b  relpred_pred_seen_zero2pred_nonZero_fragments  relpred_pred_seen_zero_y2pred_nonZero_y relpred_pred_seen_zero_b2pred_nonZero_b relpred_pred_seen_nonzero2pred_nonZero_fragments       relpred_pred_seen_nonzero_y2pred_nonZero_y      relpred_pred_seen_nonzero_b2pred_nonZero_b     relpred_not_pred_seen2pred_nonZero_fragments    relpred_not_pred_seen_y2pred_nonZero_y relpred_not_pred_seen_b2pred_nonZero_b  rel_raw_nonZero_b       rel_raw_nonZero_y     rel_raw_nonZero_fragments      Charge2 Charge3 Peptide Protein".split()
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/prosit.tab", sep='\t', index=False)
    if totest is None or "prosit_combined" in totest:
        order_and = "SpecId  Label   ScanNr  ExpMass Mass    deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da  missedCleavages  sequence_length collision_energy_aligned_normed spectral_angle delta_sa andromeda delta_score  KR      raw_nonZero_fragments  raw_nonZero_y   raw_nonZero_b   pred_nonZero_fragments  pred_nonZero_y  pred_nonZero_b  pred_not_seen  pred_not_seen_y pred_not_seen_b pred_seen_zero  pred_seen_zero_y        pred_seen_zero_b      pred_seen_nonzero        pred_seen_nonzero_y     pred_seen_nonzero_b     not_pred_seen   not_pred_seen_y not_pred_seen_b rel_pred_nonZero_y      rel_pred_nonZero_b      rel_pred_not_seen       rel_pred_not_seen_y    rel_pred_not_seen_b     rel_pred_seen_zero      rel_pred_seen_zero_y    rel_pred_seen_zero_b  rel_pred_seen_nonzero    rel_pred_seen_nonzero_y rel_pred_seen_nonzero_b rel_not_pred_seen       rel_not_pred_seen_y    rel_not_pred_seen_b     relpred_pred_not_seen2pred_nonZero_fragments    relpred_pred_not_seen_y2pred_nonZero_y relpred_pred_not_seen_b2pred_nonZero_b  relpred_pred_seen_zero2pred_nonZero_fragments  relpred_pred_seen_zero_y2pred_nonZero_y relpred_pred_seen_zero_b2pred_nonZero_b relpred_pred_seen_nonzero2pred_nonZero_fragments       relpred_pred_seen_nonzero_y2pred_nonZero_y      relpred_pred_seen_nonzero_b2pred_nonZero_b     relpred_not_pred_seen2pred_nonZero_fragments    relpred_not_pred_seen_y2pred_nonZero_y relpred_not_pred_seen_b2pred_nonZero_b  rel_raw_nonZero_b       rel_raw_nonZero_y     rel_raw_nonZero_fragments      Charge2 Charge3 Peptide Protein".split()
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/prosit_combined.tab", sep='\t', index=False)
    if totest is None or "prosit_ratio" in totest:
        order_and = "SpecId  Label   ScanNr  ExpMass Mass    deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da count_peak sum_peak  missedCleavages  sequence_length collision_energy_aligned_normed spectral_angle delta_sa andromeda delta_score  KR      raw_nonZero_fragments  raw_nonZero_y   raw_nonZero_b   pred_nonZero_fragments  pred_nonZero_y  pred_nonZero_b  pred_not_seen  pred_not_seen_y pred_not_seen_b pred_seen_zero  pred_seen_zero_y        pred_seen_zero_b      pred_seen_nonzero        pred_seen_nonzero_y     pred_seen_nonzero_b     not_pred_seen   not_pred_seen_y not_pred_seen_b rel_pred_nonZero_y      rel_pred_nonZero_b      rel_pred_not_seen       rel_pred_not_seen_y    rel_pred_not_seen_b     rel_pred_seen_zero      rel_pred_seen_zero_y    rel_pred_seen_zero_b  rel_pred_seen_nonzero    rel_pred_seen_nonzero_y rel_pred_seen_nonzero_b rel_not_pred_seen       rel_not_pred_seen_y    rel_not_pred_seen_b     relpred_pred_not_seen2pred_nonZero_fragments    relpred_pred_not_seen_y2pred_nonZero_y relpred_pred_not_seen_b2pred_nonZero_b  relpred_pred_seen_zero2pred_nonZero_fragments  relpred_pred_seen_zero_y2pred_nonZero_y relpred_pred_seen_zero_b2pred_nonZero_b relpred_pred_seen_nonzero2pred_nonZero_fragments       relpred_pred_seen_nonzero_y2pred_nonZero_y      relpred_pred_seen_nonzero_b2pred_nonZero_b     relpred_not_pred_seen2pred_nonZero_fragments    relpred_not_pred_seen_y2pred_nonZero_y relpred_not_pred_seen_b2pred_nonZero_b  rel_raw_nonZero_b       rel_raw_nonZero_y     rel_raw_nonZero_fragments      Charge2 Charge3 Peptide Protein".split()
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/prosit_ratio.tab", sep='\t', index=False)

    if irt_model is not None:
        del table
        add_irt(pack)
        table = pd.DataFrame(Features)
        if totest is None or "sa_rich" in totest:
            order_and = "SpecId  Label ScanNr  ExpMass Mass deltaM_ppm deltaM_da absDeltaM_ppm absDeltaM_da count_peak sum_peak missedCleavages sequence_length spectral_angle  delta_sa andromeda delta_score Charge2 Charge3 Peptide Protein".split()
            andre_table = table[order_and]
            andre_table.to_csv(
                f"{save_tab}/sa_rich.tab", sep='\t', index=False)
        if totest is None or "prosit_best" in totest:
            order_and = "SpecId  Label   ScanNr  ExpMass Mass retention_time irt count_peak sum_peak deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da  missedCleavages  sequence_length collision_energy_aligned_normed spectral_angle delta_sa andromeda delta_score  KR      raw_nonZero_fragments  raw_nonZero_y   raw_nonZero_b   pred_nonZero_fragments  pred_nonZero_y  pred_nonZero_b  pred_not_seen  pred_not_seen_y pred_not_seen_b pred_seen_zero  pred_seen_zero_y        pred_seen_zero_b      pred_seen_nonzero        pred_seen_nonzero_y     pred_seen_nonzero_b     not_pred_seen   not_pred_seen_y not_pred_seen_b rel_pred_nonZero_y      rel_pred_nonZero_b      rel_pred_not_seen       rel_pred_not_seen_y    rel_pred_not_seen_b     rel_pred_seen_zero      rel_pred_seen_zero_y    rel_pred_seen_zero_b  rel_pred_seen_nonzero    rel_pred_seen_nonzero_y rel_pred_seen_nonzero_b rel_not_pred_seen       rel_not_pred_seen_y    rel_not_pred_seen_b     relpred_pred_not_seen2pred_nonZero_fragments    relpred_pred_not_seen_y2pred_nonZero_y relpred_pred_not_seen_b2pred_nonZero_b  relpred_pred_seen_zero2pred_nonZero_fragments  relpred_pred_seen_zero_y2pred_nonZero_y relpred_pred_seen_zero_b2pred_nonZero_b relpred_pred_seen_nonzero2pred_nonZero_fragments       relpred_pred_seen_nonzero_y2pred_nonZero_y      relpred_pred_seen_nonzero_b2pred_nonZero_b     relpred_not_pred_seen2pred_nonZero_fragments    relpred_not_pred_seen_y2pred_nonZero_y relpred_not_pred_seen_b2pred_nonZero_b  rel_raw_nonZero_b       rel_raw_nonZero_y     rel_raw_nonZero_fragments      Charge2 Charge3 Peptide Protein".split()
            andre_table = table[order_and]
            andre_table.to_csv(
                f"{save_tab}/prosit_best.tab", sep='\t', index=False)
    if need_all:
        Features = {}
        add_id(pack)
        add_label(pack)
        add_scannr(pack)
        add_expmass(pack)
        add_mass(pack)
        add_deltaM_ppm(pack)
        add_deltaM_da(pack)
        add_absDeltaM_ppm(pack)
        add_absDeltaM_da(pack)
        add_missedCleavages(pack)
        add_seqlength(pack)
        add_andromeda(pack)
        add_delta_score(pack)

        add_charge2(pack)
        add_charge3(pack)

        add_peptide(pack)
        add_protein(pack)
        add_sa(pack)
        add_delta_sa(pack)

        add_KR(pack)
        add_collision_energy_aligned_normed(pack)
        spect_ids = add_sprectral(pack)
        table = pd.DataFrame(Features)

        order_and = "SpecId  Label ScanNr  ExpMass Mass deltaM_ppm deltaM_da absDeltaM_ppm absDeltaM_da missedCleavages sequence_length spectral_angle  delta_sa andromeda delta_score Charge2 Charge3 Peptide Protein".split()
        order_and = order_and[:11] + spect_ids + order_and[11:]
        andre_table = table[order_and]
        andre_table.to_csv(
            f"{save_tab}/spectral_all.tab", sep='\t', index=False)


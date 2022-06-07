import os
import seaborn as sns
import numpy as np
import torch
import re
import pickle
from tqdm import tqdm
from importlib import reload
import bio_helper
import math
import pandas as pd
import enum
from collections import defaultdict, OrderedDict
import ms
import matplotlib.pyplot as plt
import tools
from tools import *
from random import choices
from ms import model
from sklearn.preprocessing import scale
import sys
sys.path.append("..")
sys.path.append("./figs")


def fixed_features(msms_file, raw_dir, save_tab, over_write=False):
    table2save = f"{save_tab}/fixed_features.tab"
    if os.path.exists(table2save) and not over_write:
        return table2save
    name, msms_data = read_msms(
        msms_file)
    msms_data = filter_msms(name, msms_data)
    save2 = os.path.splitext(msms_file)[0]+"_peaks.txt"
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
            p/1e6 * m for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
        ]

    def add_absDeltaM_da(pack):
        Features['absDeltaM_da'] = [
            abs(p/1e6 * m) for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
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


def fixed_features_random(msms_file, raw_dir, save_tab, over_write=False):
    table2save = f"{save_tab}/fixed_features_shuffled.tab"
    table2shuffle = f"{save_tab}/shuffle_order.pkl"
    if os.path.exists(table2save) and not over_write:
        return table2save, pickle.load(open(table2shuffle, 'rb'))
    name, msms_data = read_msms(
        msms_file)
    msms_data = filter_msms(name, msms_data)
    save2 = os.path.splitext(msms_file)[0]+"_random_peaks.txt"
    m_r = loc_msms_in_raw(msms_data, raw_dir)
    m_r = sorted(m_r, key=lambda x: int(x[0][name.index("id")]))
    
    print(save2, os.path.exists(save2))
    if not os.path.exists(save2):
        print("Ions generating[random version]")
        all_ids = np.array([int(p[0][name.index("id")]) for p in m_r])
        shuffle_order = np.arange(len(all_ids))
        np.random.shuffle(shuffle_order)
        shuffle_matches = {
            i: j for i, j in zip(all_ids, all_ids[shuffle_order])
        }

        # TODO
        replace_attr = [
            'Modified sequence',
            'All modified sequences',
            "Charge",
            "Reverse",
            'Length',
            'Mass Error [ppm]',
            'Missed cleavages'
        ]
        all_ids_map = {int(p[0][name.index('id')]): [
            p[0][name.index(i)] for i in replace_attr] for p in m_r}
        for p in m_r:
            shuffle_id = shuffle_matches[int(p[0][name.index('id')])]
            for i, ms_attr in enumerate(replace_attr):
                p[0][name.index(ms_attr)] = all_ids_map[shuffle_id][i]

        pickle.dump(shuffle_matches, open(table2shuffle, 'wb'))
        matched_ions_pre = generate_matched_ions(m_r)
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
    shuffle_matches = pickle.load(open(table2shuffle, 'rb'))
    assert isinstance(shuffle_matches, dict)
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
            p/1e6 * m for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
        ]

    def add_absDeltaM_da(pack):
        Features['absDeltaM_da'] = [
            abs(p/1e6 * m) for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
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
    return table2save, shuffle_matches


def one_pack_all(msms_file, raw_dir, model, sample_size=None, irt_model=None, id2remove=None, pearson=False, decoyid2keep=None):

    name = read_name(msms_file)
    ions_save = os.path.splitext(msms_file)[0]+"_ions.txt"
    if sample_size is not None:
        ions_save = os.path.splitext(msms_file)[0]+f"_{sample_size}_ions.txt"
    if not os.path.exists(ions_save):
        print("Computing matched ions from scratch", ions_save)
        save_m_r_ions(msms_file, raw_dir, sample_size=sample_size)
    m_r, m_i_delta, m_i = read_m_r_ions(ions_save)
    print(f"Samples Len: {len(m_r)}")
    if id2remove is not None:
        len_before = len(m_r)
        m_i = [m_i[i] for i in range(len(m_r)) if int(
            m_r[i][0][name.index('id')]) not in id2remove]
        m_i_delta = [m_i_delta[i] for i in range(len(m_r)) if int(
            m_r[i][0][name.index('id')]) not in id2remove]
        m_r = [m_r[i] for i in range(len(m_r)) if int(
            m_r[i][0][name.index('id')]) not in id2remove]
        print(
            f"Remove {len(id2remove)} seen decoys... {len_before}->{len(m_r)}")
    elif decoyid2keep is not None:
        len_before = len(m_r)
        m_i = [m_i[i] for i in range(len(m_r)) if (len(m_r[i][0][name.index(
            'reverse')]) == 0 or int(m_r[i][0][name.index('id')]) in decoyid2keep)]
        m_i_delta = [m_i_delta[i] for i in range(len(m_r)) if (len(m_r[i][0][name.index(
            'reverse')]) == 0 or int(m_r[i][0][name.index('id')]) in decoyid2keep)]
        m_r = [m_r[i] for i in range(len(m_r)) if (len(m_r[i][0][name.index(
            'reverse')]) == 0 or int(m_r[i][0][name.index('id')]) in decoyid2keep)]
        print(
            f"Keeping {len(decoyid2keep)} decoys... {len_before}->{len(m_r)}")
    msms_data = [i[0] for i in m_r]
    # --------------------------------------------
    with torch.no_grad():
        # m_i_delta = [bio_helper.match_all(i, 'yb')
        #             for i in tqdm(matched_ions_pre_delta)]
        frag_msms_delta = [bio_helper.reverse_annotation(
            *i[:4]) for i in m_i_delta]
        data_nce_cand_delta = generate_from_msms_delta(
            msms_data, name, nces=33)

        sas_delta, _ = get_sa_all(
            model, data_nce_cand_delta, frag_msms_delta, pearson=pearson)
        sas_delta = sas_delta.cpu().numpy()

        # --------------------------------------------

        # m_i = [bio_helper.match_all(i, 'yb') for i in tqdm(matched_ions_pre)]
        frag_msms = [bio_helper.reverse_annotation(*i[:4]) for i in m_i]
        data_nce_cand = generate_from_msms(msms_data, name, nces=33)

        sas, sa_tensors = get_sa_all(
            model, data_nce_cand, frag_msms, pearson=pearson)
        sas = sas.cpu().numpy()
        sa_tensors = sa_tensors.cpu().numpy()
        # --------------------------------------------

        frag_msms = [i.reshape(-1) for i in frag_msms]
        if irt_model is not None:
            irts = get_irt_all(irt_model, data_nce_cand)
            pack = [(m[0], m[1], sa, sat, sa_d, frag, irt) for m, sa, sat, sa_d,
                    frag, irt in zip(m_r, sas, sa_tensors, sas_delta, frag_msms, irts)]
        else:
            pack = [(m[0], m[1], sa, sat, sa_d, frag) for m, sa, sat, sa_d,
                    frag in zip(m_r, sas, sa_tensors, sas_delta, frag_msms)]
        return pack, name


def one_pack_random(msms_file, raw_dir, model, sample_size=None, irt_model=None, id2remove=None, pearson=False, shuffle_matches=None):

    name, msms_data = read_msms(
        msms_file)
    msms_data = filter_msms(name, msms_data)
    m_r = loc_msms_in_raw(msms_data, raw_dir)
    m_r = sorted(m_r, key=lambda x: int(x[0][name.index("id")]))
    if id2remove is not None:
        len_before = len(m_r)
        m_r = [m_r[i] for i in range(len(m_r)) if int(
            m_r[i][0][name.index('id')]) not in id2remove]
        print(
            f"Remove {len(id2remove)} seen decoys... {len_before}->{len(m_r)}")
    if shuffle_matches is not None:
        all_ids_map = {int(p[0][name.index('id')]): (
            p[0][name.index('Modified sequence')],
            p[0][name.index('All modified sequences')],
            p[0][name.index('Charge')]
        ) for p in m_r}
        for p in m_r:
            shuffle_id = shuffle_matches[int(p[0][name.index('id')])]
            p[0][name.index('Modified sequence')] = all_ids_map[shuffle_id][0]
            p[0][name.index('All modified sequences')
                 ] = all_ids_map[shuffle_id][1]
            p[0][name.index('Charge')] = all_ids_map[shuffle_id][2]
    else:
        print("Randomly shuffle is not activated! Wrong function call")
        exit()
    msms_data = [i[0] for i in m_r]
    # --------------------------------------------
    with torch.no_grad():
        matched_ions_pre_delta = generate_matched_ions_delta(m_r)
        m_i_delta = [bio_helper.match_all(i, 'yb')
                     for i in tqdm(matched_ions_pre_delta)]
        frag_msms_delta = [bio_helper.reverse_annotation(
            *i[:4]) for i in m_i_delta]
        data_nce_cand_delta = generate_from_msms_delta(
            msms_data, name, nces=33)

        sas_delta, _ = get_sa_all(
            model, data_nce_cand_delta, frag_msms_delta, pearson=pearson)
        sas_delta = sas_delta.cpu().numpy()

        # --------------------------------------------
        matched_ions_pre = generate_matched_ions(m_r)
        m_i = [bio_helper.match_all(i, 'yb') for i in tqdm(matched_ions_pre)]
        frag_msms = [bio_helper.reverse_annotation(*i[:4]) for i in m_i]
        data_nce_cand = generate_from_msms(msms_data, name, nces=33)

        sas, sa_tensors = get_sa_all(
            model, data_nce_cand, frag_msms, pearson=pearson)
        sas = sas.cpu().numpy()
        sa_tensors = sa_tensors.cpu().numpy()
        # --------------------------------------------

        frag_msms = [i.reshape(-1) for i in frag_msms]
        if irt_model is not None:
            irts = get_irt_all(irt_model, data_nce_cand)
            pack = [(m[0], m[1], sa, sat, sa_d, frag, irt) for m, sa, sat, sa_d,
                    frag, irt in zip(m_r, sas, sa_tensors, sas_delta, frag_msms, irts)]
        else:
            pack = [(m[0], m[1], sa, sat, sa_d, frag) for m, sa, sat, sa_d,
                    frag in zip(m_r, sas, sa_tensors, sas_delta, frag_msms)]
        return pack, name


def one_pack_all_twofold(msms_file, raw_dir, model1, model2, sample_size=None, irt_model=None, id2remove=None, pearson=False):

    name = read_name(msms_file)
    ions_save = os.path.splitext(msms_file)[0]+"_ions.txt"
    if sample_size is not None:
        ions_save = os.path.splitext(msms_file)[0]+f"_{sample_size}_ions.txt"
    if not os.path.exists(ions_save):
        print("Computing matched ions from scratch", ions_save)
        save_m_r_ions(msms_file, raw_dir, sample_size=sample_size)
    m_r, m_i_delta, m_i = read_m_r_ions(ions_save)
    print(f"Before Len: {len(m_r)}")
    target_index = [i for i in range(len(m_r)) if len(m_r[i][0][name.index('Reverse')]) == 0]
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
        sas_delta_target = ((sas_delta1 + sas_delta2)/2).cpu().numpy()
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
        frag_msms_target = [bio_helper.reverse_annotation(*i[:4]) for i in [m_i[i] for i in target_index]]
        data_nce_cand = generate_from_msms(msms_data_target, name, nces=33)

        sas1, sa_tensors1 = get_sa_all(
            model1, data_nce_cand, frag_msms_target, pearson=pearson)
        sas2, sa_tensors2 = get_sa_all(
            model1, data_nce_cand, frag_msms_target, pearson=pearson)
        sas_target = ((sas1 + sas2)/2).cpu().numpy()
        sa_tensors_target = ((sa_tensors1 + sa_tensors2)/2).cpu().numpy()
        # --------------------------------------------
        frag_msms_decoy1 = [bio_helper.reverse_annotation(
            *i[:4]) for i in [m_i[i] for i in decoy_1_index]]
        data_nce_cand = generate_from_msms(
            msms_data_decoy1, name, nces=33)
        sas_decoy1,sa_tensors_decoy1 = get_sa_all(
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
        sa_tensors = np.concatenate([sa_tensors_target, sa_tensors_decoy1, sa_tensors_decoy2], axis=0)
        sas_delta = np.concatenate([sas_delta_target, sas_delta_decoy1, sas_delta_decoy2], axis=0)
        frag_msms = np.concatenate([frag_msms_target, frag_msms_decoy1, frag_msms_decoy2], axis=0)
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



def fdr_test(run_model, msms_file, raw_dir, save_tab, sample_size=300000, irt_model=None, need_all=False, id2remove=None, totest=None, pearson=False):
    # pack, msms_name = one_pack_all(
    #     msms_file,
    #     raw_dir,
    #     run_model, sample_size=sample_size, irt_model=irt_model)
    if need_all:
        print("All sprectral used for features")
    pack, msms_name = one_pack_all(
        msms_file,
        raw_dir,
        run_model, sample_size=sample_size, irt_model=irt_model, id2remove=id2remove, pearson=pearson)

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
            p/1e6 * m for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
        ]

    def add_absDeltaM_da(pack):
        Features['absDeltaM_da'] = [
            abs(p/1e6 * m) for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
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

        theoretically = Features['sequence_length']*2 * \
            np.array([int(m[0][i_d['Charge']]) for m in pack]) + 1e-9
        Features['rel_not_pred_seen'] = np.array(
            Features['not_pred_seen'])/theoretically
        Features['rel_not_pred_seen_b'] = np.array(
            Features['not_pred_seen_b'])/theoretically*2
        Features['rel_not_pred_seen_y'] = np.array(
            Features['not_pred_seen_y'])/theoretically*2
        Features['rel_pred_nonZero_b'] = np.array(
            Features['pred_nonZero_b'])/theoretically*2
        Features['rel_pred_nonZero_y'] = np.array(
            Features['pred_nonZero_y'])/theoretically*2
        Features['rel_pred_not_seen'] = np.array(
            Features['pred_not_seen'])/theoretically
        Features['rel_pred_not_seen_b'] = np.array(
            Features['pred_not_seen_b'])/theoretically*2
        Features['rel_pred_not_seen_y'] = np.array(
            Features['pred_not_seen_y'])/theoretically*2
        Features['rel_pred_seen_nonzero'] = np.array(
            Features['pred_seen_nonzero'])/theoretically
        Features['rel_pred_seen_nonzero_b'] = np.array(
            Features['pred_seen_nonzero_b'])/theoretically*2
        Features['rel_pred_seen_nonzero_y'] = np.array(
            Features['pred_seen_nonzero_y'])/theoretically*2
        Features['rel_pred_seen_zero'] = np.array(
            Features['pred_seen_zero'])/theoretically
        Features['rel_pred_seen_zero_b'] = np.array(
            Features['pred_seen_zero_b'])/theoretically*2
        Features['rel_pred_seen_zero_y'] = np.array(
            Features['pred_seen_zero_y'])/theoretically*2
        Features['rel_raw_nonZero_fragments'] = np.array(
            Features['raw_nonZero_fragments'])/theoretically
        Features['rel_raw_nonZero_b'] = np.array(
            Features['raw_nonZero_b'])/theoretically*2
        Features['rel_raw_nonZero_y'] = np.array(
            Features['raw_nonZero_y'])/theoretically*2

        Features['relpred_not_pred_seen2pred_nonZero_fragments'] = np.array(
            Features['not_pred_seen'])/(np.array(Features['pred_nonZero_fragments']) + 1e-9)
        Features['relpred_not_pred_seen_b2pred_nonZero_b'] = np.array(
            Features['not_pred_seen_b'])/(np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_not_pred_seen_y2pred_nonZero_y'] = np.array(
            Features['not_pred_seen_y'])/(np.array(Features['pred_nonZero_y']) + 1e-9)
        Features['relpred_pred_not_seen_b2pred_nonZero_b'] = np.array(
            Features['pred_not_seen_b'])/(np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_pred_not_seen_y2pred_nonZero_y'] = np.array(
            Features['pred_not_seen_y'])/(np.array(Features['pred_nonZero_y']) + 1e-9)

        Features['relpred_pred_not_seen2pred_nonZero_fragments'] = np.array(
            Features['pred_not_seen'])/(np.array(Features['pred_nonZero_fragments']) + 1e-9)
        Features['relpred_pred_seen_nonzero_b2pred_nonZero_b'] = np.array(
            Features['pred_seen_nonzero_b'])/(np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_pred_seen_nonzero_y2pred_nonZero_y'] = np.array(
            Features['pred_seen_nonzero_y'])/(np.array(Features['pred_nonZero_y']) + 1e-9)

        Features['relpred_pred_seen_nonzero2pred_nonZero_fragments'] = np.array(
            Features['pred_seen_nonzero'])/(np.array(Features['pred_nonZero_fragments']) + 1e-9)
        Features['relpred_pred_seen_zero_b2pred_nonZero_b'] = np.array(
            Features['pred_seen_zero_b'])/(np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_pred_seen_zero_y2pred_nonZero_y'] = np.array(
            Features['pred_seen_zero_y'])/(np.array(Features['pred_nonZero_y']) + 1e-9)

        Features['relpred_pred_seen_zero2pred_nonZero_fragments'] = np.array(
            Features['pred_seen_zero'])/(np.array(Features['pred_nonZero_fragments']) + 1e-9)

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
            p/1e6 * m for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
        ]

    def add_absDeltaM_da(pack):
        Features['absDeltaM_da'] = [
            abs(p/1e6 * m) for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
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

        theoretically = Features['sequence_length']*2 * \
            np.array([int(m[0][i_d['Charge']]) for m in pack]) + 1e-9
        Features['rel_not_pred_seen'] = np.array(
            Features['not_pred_seen'])/theoretically
        Features['rel_not_pred_seen_b'] = np.array(
            Features['not_pred_seen_b'])/theoretically*2
        Features['rel_not_pred_seen_y'] = np.array(
            Features['not_pred_seen_y'])/theoretically*2
        Features['rel_pred_nonZero_b'] = np.array(
            Features['pred_nonZero_b'])/theoretically*2
        Features['rel_pred_nonZero_y'] = np.array(
            Features['pred_nonZero_y'])/theoretically*2
        Features['rel_pred_not_seen'] = np.array(
            Features['pred_not_seen'])/theoretically
        Features['rel_pred_not_seen_b'] = np.array(
            Features['pred_not_seen_b'])/theoretically*2
        Features['rel_pred_not_seen_y'] = np.array(
            Features['pred_not_seen_y'])/theoretically*2
        Features['rel_pred_seen_nonzero'] = np.array(
            Features['pred_seen_nonzero'])/theoretically
        Features['rel_pred_seen_nonzero_b'] = np.array(
            Features['pred_seen_nonzero_b'])/theoretically*2
        Features['rel_pred_seen_nonzero_y'] = np.array(
            Features['pred_seen_nonzero_y'])/theoretically*2
        Features['rel_pred_seen_zero'] = np.array(
            Features['pred_seen_zero'])/theoretically
        Features['rel_pred_seen_zero_b'] = np.array(
            Features['pred_seen_zero_b'])/theoretically*2
        Features['rel_pred_seen_zero_y'] = np.array(
            Features['pred_seen_zero_y'])/theoretically*2
        Features['rel_raw_nonZero_fragments'] = np.array(
            Features['raw_nonZero_fragments'])/theoretically
        Features['rel_raw_nonZero_b'] = np.array(
            Features['raw_nonZero_b'])/theoretically*2
        Features['rel_raw_nonZero_y'] = np.array(
            Features['raw_nonZero_y'])/theoretically*2

        Features['relpred_not_pred_seen2pred_nonZero_fragments'] = np.array(
            Features['not_pred_seen'])/(np.array(Features['pred_nonZero_fragments']) + 1e-9)
        Features['relpred_not_pred_seen_b2pred_nonZero_b'] = np.array(
            Features['not_pred_seen_b'])/(np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_not_pred_seen_y2pred_nonZero_y'] = np.array(
            Features['not_pred_seen_y'])/(np.array(Features['pred_nonZero_y']) + 1e-9)
        Features['relpred_pred_not_seen_b2pred_nonZero_b'] = np.array(
            Features['pred_not_seen_b'])/(np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_pred_not_seen_y2pred_nonZero_y'] = np.array(
            Features['pred_not_seen_y'])/(np.array(Features['pred_nonZero_y']) + 1e-9)

        Features['relpred_pred_not_seen2pred_nonZero_fragments'] = np.array(
            Features['pred_not_seen'])/(np.array(Features['pred_nonZero_fragments']) + 1e-9)
        Features['relpred_pred_seen_nonzero_b2pred_nonZero_b'] = np.array(
            Features['pred_seen_nonzero_b'])/(np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_pred_seen_nonzero_y2pred_nonZero_y'] = np.array(
            Features['pred_seen_nonzero_y'])/(np.array(Features['pred_nonZero_y']) + 1e-9)

        Features['relpred_pred_seen_nonzero2pred_nonZero_fragments'] = np.array(
            Features['pred_seen_nonzero'])/(np.array(Features['pred_nonZero_fragments']) + 1e-9)
        Features['relpred_pred_seen_zero_b2pred_nonZero_b'] = np.array(
            Features['pred_seen_zero_b'])/(np.array(Features['pred_nonZero_b']) + 1e-9)
        Features['relpred_pred_seen_zero_y2pred_nonZero_y'] = np.array(
            Features['pred_seen_zero_y'])/(np.array(Features['pred_nonZero_y']) + 1e-9)

        Features['relpred_pred_seen_zero2pred_nonZero_fragments'] = np.array(
            Features['pred_seen_zero'])/(np.array(Features['pred_nonZero_fragments']) + 1e-9)

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


if __name__ == "__main__":
    which = "trypsin"
    save_tab = f"/data/prosit/figs/fig235/{which}/percolator_up/try/autos"
    msms_file = f"/data/prosit/figs/fig235/{which}/maxquant/combined/txt/msms.txt"
    raw_dir = f"/data/prosit/figs/fig235/{which}/raw"

    fixed_features(msms_file, raw_dir, save_tab)

# def fdr_test_reverse(run_model, msms_file, raw_dir, save_tab, sample_size=300000, irt_model=None):
#     # pack, msms_name = one_pack_all(
#     #     msms_file,
#     #     raw_dir,
#     #     run_model, sample_size=sample_size, irt_model=irt_model)

#     pack, msms_name = one_pack_all_reverse(
#         msms_file,
#         raw_dir,
#         run_model, sample_size=sample_size, irt_model=irt_model)

#     # %%
#     need_col = ['id', "Raw file", 'Scan number', "Reverse", "Mass",
#                 "Sequence", "Charge", "Missed cleavages", "Length", "Mass Error [ppm]",
#                 "Score", "Delta score", "All modified sequences"]
#     i_d = {}
#     for c in need_col:
#         i_d[c] = msms_name.index(c)

#     # %%
#     Features = {}

#     def add_id(pack):
#         Features['SpecId'] = [int(m[0][i_d['id']]) for m in pack]

#     def add_scannr(pack):
#         Features['ScanNr'] = [hash(
#             m[0][i_d['Raw file']] + "|" + m[0][i_d['Scan number']] + "|" + m[0][i_d['id']]) for m in pack]

#     def add_label(pack):
#         Features['Label'] = [-1 if m[0]
#                              [i_d['Reverse']].strip() else 1 for m in pack]

#     def add_expmass(pack):
#         Features['ExpMass'] = 1000

#     def add_mass(pack):
#         Features['Mass'] = [float(m[0][i_d['Mass']]) for m in pack]

#     def add_peptide(pack):
#         Features['Peptide'] = ["_." + m[0]
#                                [i_d['Sequence']].strip("_") + "._" for m in pack]

#     def add_protein(pack):
#         Features['Protein'] = [m[0][i_d['Sequence']].strip("_") for m in pack]

#     def add_charge2(pack):
#         Features['Charge2'] = [
#             1 if int(m[0][i_d['Charge']]) == 2 else 0 for m in pack]

#     def add_charge3(pack):
#         Features['Charge3'] = [
#             1 if int(m[0][i_d['Charge']]) == 3 else 0 for m in pack]

#     def add_missedCleavages(pack):
#         Features['missedCleavages'] = [
#             int(m[0][i_d["Missed cleavages"]]) for m in pack]

#     def add_seqlength(pack):
#         Features['sequence_length'] = [int(m[0][i_d['Length']]) for m in pack]

#     def add_deltaM_ppm(pack):
#         Features['deltaM_ppm'] = [
#             float(m[0][i_d['Mass Error [ppm]']]) for m in pack]
#         Features['deltaM_ppm'] = [0. if math.isnan(
#             i) else i for i in Features['deltaM_ppm']]

#     def add_absDeltaM_ppm(pack):
#         Features['absDeltaM_ppm'] = [
#             float(m[0][i_d['Mass Error [ppm]']]) for m in pack]
#         Features['absDeltaM_ppm'] = [0. if math.isnan(
#             i) else abs(i) for i in Features['absDeltaM_ppm']]

#     def add_deltaM_da(pack):
#         Features['deltaM_da'] = [
#             p/1e6 * m for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
#         ]

#     def add_absDeltaM_da(pack):
#         Features['absDeltaM_da'] = [
#             abs(p/1e6 * m) for m, p in zip(Features['Mass'], Features['deltaM_ppm'])
#         ]

#     def add_sa(pack):
#         Features['spectral_angle'] = [float(m[2]) - float(m[6]) for m in pack]

#     def add_delta_sa(pack):
#         Features['delta_sa'] = [float(m[2]) - float(m[4]) for m in pack]

#     def add_andromeda(pack):
#         Features['andromeda'] = [float(m[0][i_d['Score']]) for m in pack]

#     def add_delta_score(pack):
#         Features['delta_score'] = [
#             float(m[0][i_d["Delta score"]]) for m in pack]

#     def add_irt(pack):
#         Features['irt'] = [float(m[6]) for m in pack]

#     # def add_reverse(pack):
#     #     Features['reverse'] =

#     def add_collision_energy_aligned_normed(pack):
#         Features['collision_energy_aligned_normed'] = 0.33

#     def add_pred(pack):
#         Features['pred_seen_nonzero'] = [np.count_nonzero(
#             np.clip(m[3][m[5] > 0], 0, None)) for m in pack]
#         Features['not_pred_seen'] = [np.sum(
#             np.clip(m[3][m[5] > 0], 0, None) == 0) for m in pack]

#         # Features['pred_seen_nonzero'] = [np.sum(m[3] > 0) for m in pack]
#         # Features['delta_score'] = [0. if math.isnan(
#         #     i) else i for i in Features['delta_score']]

#     # %%
#     add_id(pack)
#     add_label(pack)
#     add_scannr(pack)
#     add_expmass(pack)
#     add_mass(pack)
#     add_deltaM_ppm(pack)
#     add_deltaM_da(pack)
#     add_absDeltaM_ppm(pack)
#     add_absDeltaM_da(pack)
#     add_missedCleavages(pack)
#     add_seqlength(pack)
#     add_andromeda(pack)
#     add_delta_score(pack)

#     add_charge2(pack)
#     add_charge3(pack)

#     add_peptide(pack)
#     add_protein(pack)

#     add_sa(pack)
#     add_delta_sa(pack)

#     # %%
#     table = pd.DataFrame(Features)

#     order_and = "SpecId  Label   ScanNr  ExpMass Mass    deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da    missedCleavages        sequence_length andromeda       delta_score     Charge2 Charge3 Peptide Protein".split()
#     andre_table = table[order_and]
#     andre_table.to_csv(
#         f"{save_tab}/andromeda.tab", sep='\t', index=False)

#     order_and = "SpecId  Label   ScanNr  ExpMass Mass    deltaM_ppm      deltaM_da       absDeltaM_ppm   absDeltaM_da    missedCleavages        sequence_length spectral_angle delta_sa  Charge2 Charge3 Peptide Protein".split()
#     andre_table = table[order_and]
#     andre_table.to_csv(
#         f"{save_tab}/sa.tab", sep='\t', index=False)

#     order_and = "SpecId  Label ScanNr  ExpMass Mass deltaM_ppm deltaM_da absDeltaM_ppm absDeltaM_da missedCleavages sequence_length spectral_angle  delta_sa andromeda delta_score Charge2 Charge3 Peptide Protein".split()
#     andre_table = table[order_and]
#     andre_table.to_csv(
#         f"{save_tab}/combined.tab", sep='\t', index=False)

#     if irt_model is not None:
#         add_irt(pack)
#         order_and = "SpecId  Label ScanNr  ExpMass Mass deltaM_ppm deltaM_da absDeltaM_ppm absDeltaM_da missedCleavages sequence_length spectral_angle  delta_sa andromeda delta_score irt Charge2 Charge3 Peptide Protein".split()
#         andre_table = table[order_and]
#         andre_table.to_csv(
#             f"{save_tab}/combined_irt.tab", sep='\t', index=False)

# def one_pack_all_reverse(msms_file, raw_dir, model, sample_size=None, irt_model=None):
#     print("Reverse Score")

#     name = read_name(msms_file)
#     ions_save = os.path.splitext(msms_file)[0]+"_ions_reverse.txt"
#     if sample_size is not None:
#         ions_save = os.path.splitext(msms_file)[0]+f"_{sample_size}_ions_reverse.txt"
#     if not os.path.exists(ions_save):
#         print("Computing matched ions from scratch", ions_save)
#         save_m_r_ions_reverse(msms_file, raw_dir, sample_size=sample_size)
#     m_r, m_i_delta, m_i, m_i_rever = read_m_r_ions_reverse(ions_save)
#     msms_data = [i[0] for i in m_r]
#     # --------------------------------------------
#     with torch.no_grad():
#         frag_msms_delta = [bio_helper.reverse_annotation(*i[:4]) for i in m_i_delta]

#         data_nce_cand_delta = generate_from_msms_delta(
#             msms_data, name, nces=33)

#         sas_delta, _ = get_sa_all(model, data_nce_cand_delta, frag_msms_delta)
#         sas_delta = sas_delta.cpu().numpy()

#         del data_nce_cand_delta, frag_msms_delta, m_i_delta, _
#         # --------------------------------------------
#         frag_msms = [bio_helper.reverse_annotation(*i[:4]) for i in m_i]
#         data_nce_cand = generate_from_msms(msms_data, name, nces=33)
#         sas, sa_tensors = get_sa_all(model, data_nce_cand, frag_msms)
#         sas = sas.cpu().numpy()
#         sa_tensors = sa_tensors.cpu().numpy()

#         del data_nce_cand
#         # --------------------------------------------
#         frag_msms_rever = [bio_helper.reverse_annotation(*i[:4]) for i in m_i_rever]
#         data_nce_cand_rever = generate_from_msms_reverse(msms_data, name, nces=33)
#         sas_rever, _ = get_sa_all(model, data_nce_cand_rever, frag_msms_rever)
#         sas_rever = sas_rever.cpu().numpy()

#         del frag_msms_rever, data_nce_cand_rever, _
#         # --------------------------------------------

#         frag_msms = [i.reshape(-1) for i in frag_msms]
#         if irt_model is not None:
#             irts = get_irt_all(irt_model, data_nce_cand)
#             pack = [(m[0], m[1], sa, sat, sa_d, frag, irt) for m, sa, sat, sa_d,
#                     frag, irt in zip(m_r, sas, sa_tensors, sas_delta, frag_msms, irts)]
#         else:
#             pack = [(m[0], m[1], sa, sat, sa_d, frag, sa_rever) for m, sa, sat, sa_d,
#                     frag, sa_rever in zip(m_r, sas, sa_tensors, sas_delta, frag_msms, sas_rever)]
#         return pack, name

# def one_pack_all_ratio(msms_file, raw_dir, model, sample_size=None, irt_model=None):
#     print("Modified SA with ratio")
#     name = read_name(msms_file)
#     ions_save = os.path.splitext(msms_file)[0]+"_ions.txt"
#     if sample_size is not None:
#         ions_save = os.path.splitext(msms_file)[0]+f"_{sample_size}_ions.txt"
#     if not os.path.exists(ions_save):
#         print("Computing matched ions from scratch", ions_save)
#         save_m_r_ions(msms_file, raw_dir, sample_size=sample_size)
#     m_r, m_i_delta, m_i = read_m_r_ions(ions_save)
#     msms_data = [i[0] for i in m_r]
#     all_peaks = all_intensities_len(m_r)
#     all_peaks_len = [len(i) for i in all_peaks]
#     # --------------------------------------------
#     with torch.no_grad():
#         # m_i_delta = [bio_helper.match_all(i, 'yb')
#         #             for i in tqdm(matched_ions_pre_delta)]
#         frag_msms_delta = [bio_helper.reverse_annotation(
#             *i[:4]) for i in m_i_delta]
#         ratio_match = np.array(
#             [len(i[1])/j for i, j in zip(m_i_delta, all_peaks_len)])
#         data_nce_cand_delta = generate_from_msms_delta(
#             msms_data, name, nces=33)

#         sas_delta, _ = get_sa_all(model, data_nce_cand_delta, frag_msms_delta)
#         sas_delta = sas_delta.cpu().numpy()
#         sas_delta = ratio_match * sas_delta
#         # --------------------------------------------

#         # m_i = [bio_helper.match_all(i, 'yb') for i in tqdm(matched_ions_pre)]
#         frag_msms = [bio_helper.reverse_annotation(*i[:4]) for i in m_i]
#         ratio_match = np.array(
#             [len(i[1])/j for i, j in zip(m_i, all_peaks_len)])
#         ratio_sum_match = np.array(
#             [sum(i[1])/sum(j) for i, j in zip(m_i, all_peaks)])

#         data_nce_cand = generate_from_msms(msms_data, name, nces=33)

#         sas, sa_tensors = get_sa_all(model, data_nce_cand, frag_msms)
#         sas = sas.cpu().numpy()
#         sas = sas*ratio_match
#         sa_tensors = sa_tensors.cpu().numpy()
#         # --------------------------------------------

#         frag_msms = [i.reshape(-1) for i in frag_msms]
#         if irt_model is not None:
#             irts = get_irt_all(irt_model, data_nce_cand)
#             pack = [(m[0], m[1], sa, sat, sa_d, frag, irt) for m, sa, sat, sa_d,
#                     frag, irt in zip(m_r, sas, sa_tensors, sas_delta, frag_msms, irts)]
#         else:
#             pack = [(m[0], m[1], sa, sat, sa_d, frag) for m, sa, sat, sa_d,
#                     frag in zip(m_r, sas, sa_tensors, sas_delta, frag_msms)]
#         return pack, name

# def one_pack_all_scale(msms_file, raw_dir, model, sample_size=None, irt_model=None):
#     # name, msms_data = tools.read_msms(
#     #     msms_file)
#     # msms_data = tools.filter_msms(name, msms_data)
#     # if sample_size is not None:
#     #     msms_data = choices(msms_data, k=sample_size)
#     # msms_data.sort(key=lambda x: int(x[name.index("id")]))
#     # m_r = loc_msms_in_raw(msms_data, raw_dir)
#     # m_r = sorted(m_r, key=lambda x: int(x[0][name.index("id")]))

#     # matched_ions_pre = generate_matched_ions(m_r)
#     # matched_ions_pre_delta = generate_matched_ions_delta(m_r)
#     print("Modified SA with scale")
#     name = read_name(msms_file)
#     ions_save = os.path.splitext(msms_file)[0]+"_ions.txt"
#     if sample_size is not None:
#         ions_save = os.path.splitext(msms_file)[0]+f"_{sample_size}_ions.txt"
#     if not os.path.exists(ions_save):
#         print("Computing matched ions from scratch", ions_save)
#         save_m_r_ions(msms_file, raw_dir, sample_size=sample_size)
#     m_r, m_i_delta, m_i = read_m_r_ions(ions_save)
#     msms_data = [i[0] for i in m_r]
#     all_peaks = all_intensities_len(m_r)
#     all_peaks_scale = np.array([i[6] for i in m_i])
#     # --------------------------------------------
#     with torch.no_grad():
#         # m_i_delta = [bio_helper.match_all(i, 'yb')
#         #             for i in tqdm(matched_ions_pre_delta)]

#         frag_msms_delta = [bio_helper.reverse_annotation(
#             *i[:4]) for i in m_i_delta]
#         # ratio_match = np.array(
#         #     [len(i[1])/j for i, j in zip(m_i_delta, all_peaks_len)])
#         # ratio_sum_match = np.array(
#         #     [sum(i[1])/sum(j) for i, j in zip(m_i_delta, all_peaks)])
#         data_nce_cand_delta = generate_from_msms_delta(
#             msms_data, name, nces=33)

#         sas_delta, _ = get_sa_all_scale(model, data_nce_cand_delta, frag_msms_delta, all_peaks_scale)
#         sas_delta = sas_delta.cpu().numpy()
#         # --------------------------------------------

#         # m_i = [bio_helper.match_all(i, 'yb') for i in tqdm(matched_ions_pre)]
#         frag_msms = [bio_helper.reverse_annotation(*i[:4]) for i in m_i]
#         data_nce_cand = generate_from_msms(msms_data, name, nces=33)

#         sas, sa_tensors = get_sa_all_scale(model, data_nce_cand, frag_msms, all_peaks_scale)
#         sas = sas.cpu().numpy()
#         sa_tensors = sa_tensors.cpu().numpy()
#         # --------------------------------------------

#         frag_msms = [i.reshape(-1) for i in frag_msms]
#         if irt_model is not None:
#             irts = get_irt_all(irt_model, data_nce_cand)
#             pack = [(m[0], m[1], sa, sat, sa_d, frag, irt) for m, sa, sat, sa_d,
#                     frag, irt in zip(m_r, sas, sa_tensors, sas_delta, frag_msms, irts)]
#         else:
#             pack = [(m[0], m[1], sa, sat, sa_d, frag) for m, sa, sat, sa_d,
#                     frag in zip(m_r, sas, sa_tensors, sas_delta, frag_msms)]
#         return pack, name

# def one_pack_no_sa(msms_file, raw_dir, sample_size=None):
#     name, msms_data = tools.read_msms(
#         msms_file)
#     msms_data = tools.filter_msms(name, msms_data)
#     if sample_size is not None:
#         msms_data = choices(msms_data, k=sample_size)
#     msms_data.sort(key=lambda x: int(x[name.index("id")]))
#     m_r = loc_msms_in_raw(msms_data, raw_dir)
#     m_r = sorted(m_r, key=lambda x: int(x[0][name.index("id")]))
#     return m_r, name

# def one_pack_single(msms_file, raw_dir, model, sample_size=None, irt_model=None, id2remove=None):

#     name = read_name(msms_file)
#     ions_save = os.path.splitext(msms_file)[0]+"_ions.txt"
#     if sample_size is not None:
#         ions_save = os.path.splitext(msms_file)[0]+f"_{sample_size}_ions.txt"
#     if not os.path.exists(ions_save):
#         print("Computing matched ions from scratch", ions_save)
#         save_m_r_ions(msms_file, raw_dir, sample_size=sample_size)
#     m_r, m_i_delta, m_i = read_m_r_ions(ions_save)
#     if id2remove is not None:
#         len_before = len(m_r)
#         m_i = [m_i[i] for i in range(len(m_r)) if int(
#             m_r[i][0][name.index('id')]) not in id2remove]
#         m_i_delta = [m_i_delta[i] for i in range(len(m_r)) if int(
#             m_r[i][0][name.index('id')]) not in id2remove]
#         m_r = [m_r[i] for i in range(len(m_r)) if int(
#             m_r[i][0][name.index('id')]) not in id2remove]
#         print(
#             f"Remove {len(id2remove)} seen decoys... {len_before}->{len(m_r)}")
#     msms_data = [i[0] for i in m_r]
#     # --------------------------------------------
#     with torch.no_grad():
#         # frag_msms_delta = [bio_helper.reverse_annotation(
#         #     *i[:4]) for i in m_i_delta]
#         # data_nce_cand_delta = generate_from_msms_delta(
#         #     msms_data, name, nces=33)

#         # # TODO: for single score implementation
#         # sas_delta = get_single_score_all(model, data_nce_cand_delta, frag_msms_delta)
#         # sas_delta = sas_delta.cpu().numpy()

#         # # --------------------------------------------

#         frag_msms = [bio_helper.reverse_annotation(*i[:4]) for i in m_i]
#         data_nce_cand = generate_from_msms_rt(msms_data, name, nces=33)

#         sas = get_single_score_all(model, data_nce_cand, frag_msms)
#         sas = sas.cpu().numpy()
#         # --------------------------------------------

#         if irt_model is not None:
#             irts = get_irt_all(irt_model, data_nce_cand)
#             pack = [(m[0], m[1], sa, frag, irt) for m, sa, sa_d,
#                     frag, irt in zip(m_r, sas, frag_msms, irts)]
#         else:
#             pack = [(m[0], m[1], sa, frag) for m, sa,
#                     frag in zip(m_r, sas, frag_msms)]
#         return pack, name

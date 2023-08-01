import os
import math
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("../")
from pept3.tools import read_name, save_m_r_ions, read_m_r_ions
from pept3 import bio_helper
from pept3 import helper

def ms2pip_result_convert(ms2pipdir, max_length=30, min_length=7, max_charge=6):
    peprec = os.path.join(ms2pipdir, "filter_from_maxquant.peprec")
    result = os.path.join(ms2pipdir, "spectrum.predict_predictions.csv")
    
    peprec : pd.DataFrame = pd.read_csv(peprec, sep=' ')
    result : pd.DataFrame = pd.read_csv(result)
    result = result[result['ionnumber'] <= max_length]
    result = result[result['charge'] <= max_charge]
    
    specid_dict = {}
    counts = [0, 0]
    for pack in tqdm(result.groupby("spec_id")):
        id_key = pack[0]
        bio_data = peprec[peprec['spec_id'] == id_key]
        assert len(bio_data) == 1
        bio_data = bio_data.squeeze()
        counts[1] += 1

        if len(bio_data['peptide']) > max_length or len(bio_data['peptide']) < min_length:
            counts[0] += 1
            continue
        if bio_data['charge'] > max_charge:
            counts[0] += 1
            continue
        
        predict_flag = (bio_data['modified_peptide'], bio_data['charge'], )
        if predict_flag in specid_dict:
            continue
        ions_t = pack[1]
        frag_msms = np.zeros((29,2,3))
        frag_num = ions_t['ionnumber'].to_numpy() - 1

        ion_num = ions_t['ion'].apply(lambda x: 0 if x == "Y" else 1).to_numpy()
        charge_num = ions_t['charge'].to_numpy() - 1
        # unlog intensity, refer to https://github.com/compomics/ms2pip#output
        intens = 2**(ions_t['prediction'].to_numpy()) - 0.001
        
        charge_index = (charge_num <= 2)
        frag_msms[frag_num[charge_index], 
                  ion_num[charge_index], 
                  charge_num[charge_index]] = intens[charge_index]
        specid_dict[predict_flag] = (frag_msms, bio_data)
    
    print("[remove, total]:", counts)
    return specid_dict

def get_sa_pack(lookup_dict, name, m_r, m_i_delta, m_i, need_irt=True,
                fixed_modifications=[('C', 'cm')]):
    irts = []
    sas = []
    sas_tensor = []
    frag_msms_list = []
    ids_index = range(len(m_r))
    
    msms_data = [m_r[i][0] for i in ids_index]
    frag_msms_delta = [bio_helper.reverse_annotation(*i[:4]) for i in [m_i_delta[i] for i in ids_index]]
    frag_msms = [bio_helper.reverse_annotation(*i[:4]) for i in [m_i[i] for i in ids_index]]    
    
    def get_pack(seq, charge, frag_msms):
        seq = seq.replace("_", "")
        seq = seq.replace(".", "")
        for aa, mod in fixed_modifications:
            seq = seq.replace(aa, f"{aa}({mod})")
        ms2pip_pred, bio_data = lookup_dict[(seq, charge)]
        ms2pip_pred = torch.from_numpy(ms2pip_pred)
        ms2pip_max = ms2pip_pred/(ms2pip_pred.max() + 1e-9)
        frag_msms = torch.from_numpy(frag_msms)
        ms2pip_max[frag_msms == -1] = -1
        
        ms2pip_max = ms2pip_max.reshape(1, -1)
        frag_msms = frag_msms.reshape(1, -1)
        sa = helper.spectral_angle(frag_msms, ms2pip_max)
        import ipdb
        ipdb.set_trace()
        return sa, frag_msms, ms2pip_max, bio_data
    
    loss_c = 0
    bar = tqdm(zip(msms_data, frag_msms_delta, frag_msms))
    for m, mfd, mf in bar:
        charge = int(m[name.index('Charge')])
        seq = m[name.index('Modified sequence')]
        try:
            sa, mf, pip_mf, bio_data = get_pack(seq, charge, mf)
        except KeyError:
            loss_c += 1
            bar.set_description(f"loss:{loss_c}")    
            continue    
        sas.append(sa.numpy())
        sas_tensor.append(pip_mf.numpy())
        frag_msms_list.append(mf.numpy())
        if need_irt:
            irts.append(bio_data['rt'])
    sas = np.concatenate(sas, axis=0)
    sas_tensor = np.concatenate(sas_tensor, axis=0)
    sas_delta = sas
    frag_msms = np.concatenate(frag_msms_list, axis=0)
    if need_irt:
        irts = np.array(irts)
        pack = [(m[0], m[1], sa, sat, sa_d, frag, irt) for m, sa, sat, sa_d,
                    frag, irt in zip(m_r, sas, sas_tensor, sas_delta, frag_msms, irts)]
    else:
        pack = [(m[0], m[1], sa, sat, sa_d, frag) for m, sa, sat, sa_d,
                    frag in zip(m_r, sas, sas_tensor, sas_delta, frag_msms)]
    return pack, name
    
def one_pack_raw(msms_file,
                 raw_dir,
                 lookup_dict,
                 need_irt=True):
    name = read_name(msms_file)
    ions_save = os.path.splitext(msms_file)[0] + "_ions.txt"
    if not os.path.exists(ions_save):
        print("Computing matched ions from scratch", ions_save)
        save_m_r_ions(msms_file, raw_dir, sample_size=None)
    m_r, m_i_delta, m_i = read_m_r_ions(ions_save, sample_size=100)
    return get_sa_pack(lookup_dict, name, m_r, m_i_delta, m_i, need_irt=need_irt)
    

def fdr_test(lookup_dict, msms_file, raw_dir, save_tab, need_irt=True, totest=None, need_all=True):
    if need_all:
        print("All sprectral used for features")
    pack, msms_name = one_pack_raw(msms_file, 
                                   raw_dir,
                                   lookup_dict,
                                   need_irt=need_irt)
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

    if need_irt:
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
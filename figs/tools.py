import os
from collections import defaultdict, OrderedDict
import re
import sys
import torch
sys.path.append("..")
import numpy as np
import constants
from ms import helper
import ms
import math
import random
from tqdm import tqdm
import bio_helper
from bio_helper import peptide_parser
from random import choices, sample

MSMS_NAME=None

def read_msms(file):
    global MSMS_NAME
    with open(file, "r") as f:
        data = []
        for i, l in enumerate(f):
            if i == 0:
                l = l.strip().split("\t")
                head = l
            if i != 0:
                l = l.strip().split("\t")
                data.append(l)
    MSMS_NAME = head
    return head, data

def has(hint, alist):
    for i in alist:
        if i in hint:
            return True
    return False


def filter_peptide(pe: str):
    # pat = re.compile(r".+(M\(ox\))+.*")
    pat = re.compile(r".+(M\(ox\))*.*")
    if pe.startswith("_(ac)"):
        return False
    elif has(pe, ["U", "X", "O"]):
        return False
    elif not re.match(pat, pe):
        return False
    return True


def head(data, i, top=10):
    return [data[_][i] for _ in range(top)]

def filter_msms(name, row_data):
    frag_index = name.index("Fragmentation")
    data = [i for i in row_data if i[frag_index] == "HCD"]
    data = [i for i in data if int(i[name.index("Charge")]) < 7]
    data = [i for i in data if 7 <= int(i[name.index("Length")]) <= 30]
    data = [i for i in data if filter_peptide(i[name.index("Modified sequence")])]
    data = [i for i in data if len(i[name.index("All modified sequences")].split(';')) > 1]
    data = [i for i in data if filter_peptide(
        i[name.index("All modified sequences")].split(';')[1])]
    data = [i for i in data if 7 <= len(
        i[name.index("All modified sequences")].split(';')[1].strip("_")) <= 30]
    # data = [i for i in data if not math.isnan(
    #     float(i[name.index("Mass Error [ppm]")]))]
    return data


def loc_msms_in_raw(data, raws_dir):
    csv_files = [f for f in os.listdir(raws_dir) if f.endswith('.csv')]
    csv_files.sort()

    find_raw = defaultdict(list)
    for t in data:
        find_raw[t[0]].append(t)

    for f, scans in find_raw.items():
        scans.sort(key=lambda x: int(x[1]))

    matches = []
    total_spect = 0
    for f in sorted(find_raw.keys()):
        scans = find_raw[f]
        index_now = 0
        file2read = os.path.join(raws_dir, f+".csv")
        log_raw = []
        log_scan = []
        with open(file2read) as read:
            all_lines = []
            for i, line in enumerate(read):
                if i == 0:
                    continue
                fields = line.strip().split(',')
                try:
                    scan_number = int(fields[0])
                except:
                    print(fields[0], "unable to parse")
                    continue
                all_lines.append(fields)
            all_lines.sort(key=lambda x: int(x[0]))
        # print([x[0] for x in all_lines][-20:])
        # print([x[1] for x in scans][29486:29486+20])
        for i, line in enumerate(all_lines):
            # if i == 0:
            #     continue
            # line = line.strip().split(',')
            # log_raw.append(int(line[0]))
            # log_scan.append(int(scans[index_now][1]))
            while index_now < len(scans) and int(scans[index_now][1]) <= int(line[0]):
                if int(scans[index_now][1]) == int(line[0]):
                    matches.append((scans[index_now], line))
                index_now += 1
            if index_now == len(scans):
                break
        total_spect += len(scans)
        if index_now < len(scans):
            print(f"{f} -- Not match spect: No.{index_now}-{scans[index_now][1]} in {len(scans)}")
        # print("Done", f)
    return matches

# def loc_m_r_in_percolator(ms_name, m_r, tab_file):
#     """Make sure m_r is sorted by msms id"""
#     ms_seq_index = ms_name.index("Modified sequence")
#     ms_score_index = ms_name.index("Score")
#     tab_name, tab_data = read_msms(tab_file)
#     tab_seq_index = tab_name.index("Protein")
#     tab_score_index = tab_name.index("andromeda")
#     def match_a_pair(ms_a, tab_b):
#         print(ms_a[ms_seq_index].strip("_"), tab_b[tab_seq_index], ms_a[ms_score_index], tab_b[tab_score_index] )
#         return (ms_a[ms_seq_index].strip("_") == tab_b[tab_seq_index]) \
#             and abs(float(ms_a[ms_score_index]) - float(tab_b[tab_score_index])) < 1e-7
#     m_r_i = 0
#     tab_i = 0
#     matches = []
#     while m_r_i < len(m_r) and tab_i < len(tab_data):
#         ms_a = m_r[m_r_i][0]
#         tab_b = tab_data[tab_i]
#         if match_a_pair(ms_a, tab_b):
#             matches.append((m_r[m_r_i][0], m_r[m_r_i][1], tab_data[tab_i]))
#             m_r_i += 1
#             tab_i += 1
#         else:
#             m_r_i += 1
#         if m_r_i > 100:
#             break
#     try:
#         assert tab_i == len(tab_data)
#     except:
#         print(tab_i, m_r_i)
#         raise
#     return matches


def combine_m_r_p(msms_file, raw_dir, percolator_tab):
    ms_name, ms_data = read_msms(msms_file)
    ms_data = filter_msms(ms_name, ms_data)
    key2sort = ms_name.index("id")

    print(f"Load {len(ms_data)} from {msms_file}")
    m_r = loc_msms_in_raw(ms_data, raw_dir)
    m_r.sort(key=lambda x: int(x[0][key2sort]))
    print([i[0][key2sort]] for i in m_r[:100])
    return
    print(f"Load {len(m_r)} pair from {msms_file} and {raw_dir}")

    m_r_p = loc_m_r_in_percolator(ms_name, m_r, percolator_tab)
    return m_r_p


def test_model_frag(model):
    import json
    from ms.dataset import FragDataset
    from torch.utils.data import DataLoader
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    config_data = json.load(open("../checkpoints/data.json"))
    frag_dir = config_data['frag']
    holdout = os.path.join(frag_dir, "holdout_hcd.hdf5")
    test_data = FragDataset(holdout, ratio=1)
    test_loader = DataLoader(
        test_data.train(), batch_size=1024, shuffle=True)
    with torch.no_grad():
        loss_test = 0
        test_count = 0
        model = model.eval()
        for i, data in tqdm(enumerate(test_loader)):
            test_count += 1

            data = {k: v.to(device) for k, v in data.items()}
            data["peptide_mask"] = helper.create_mask(data['sequence_integer'])

            pred = model(data, choice='frag')
            loss_b = helper.spectral_distance(data['intensities_raw'], pred)
            loss_test += loss_b.item()
        loss_test /= test_count
    print(f"----Test Loss: {loss_test:.5f}----")

def filter_m_r(m_r):
    def inten_mass_match(row):
        return len(row[1][2].split(' ')) == len(row[1][3].split(' '))
    data = [m for m in m_r if inten_mass_match(m)]
    return data
    
def generate_matched_ions(matches):
    re = []
    for m in matches:
        pack = {}
        msms = m[0]
        raw = m[1]
        pack['intensities_raw'] = raw[3]
        pack['masses_raw'] = raw[2]
        pack['mass_analyzer'] = raw[1].split(" ")[0]
        pack['modified_sequence'] = msms[MSMS_NAME.index(
            "Modified sequence")].strip("_")
        pack['charge'] = int(msms[MSMS_NAME.index("Charge")])
        pack['id'] = int(msms[MSMS_NAME.index("id")])
        re.append(pack)
    return re

def all_intensities_len(matches):
    re = []
    for m in matches:
        raw = m[1]
        re.append([float(i) for i in raw[3].split(' ')])
    return re

def generate_matched_ions_delta(matches):
    re = []
    for m in matches:
        pack = {}
        msms = m[0]
        raw = m[1]
        pack['intensities_raw'] = raw[3]
        pack['masses_raw'] = raw[2]
        pack['mass_analyzer'] = raw[1].split(" ")[0]
        pack['modified_sequence'] = msms[MSMS_NAME.index(
            "All modified sequences")].split(";")[1].strip("_")
        pack['charge'] = int(msms[MSMS_NAME.index("Charge")])
        pack['id'] = int(msms[MSMS_NAME.index("id")])
        re.append(pack)
    return re


# def generate_matched_ions_reverse(matches):
#     re = []
#     for m in matches:
#         pack = {}
#         msms = m[0]
#         raw = m[1]
#         pack['intensities_raw'] = raw[3]
#         pack['masses_raw'] = raw[2]
#         pack['mass_analyzer'] = raw[1].split(" ")[0]
#         pack['modified_sequence'] = "".join(
#             list(peptide_parser(msms[7].strip("_")))[::-1])
#         pack['charge'] = int(msms[15])
#         re.append(pack)
#     return re


def generate_matched_ions_reverse(matches):
    re = []
    for m in matches:
        pack = {}
        msms = m[0]
        raw = m[1]
        pack['intensities_raw'] = raw[3]
        pack['masses_raw'] = raw[2]
        pack['mass_analyzer'] = raw[1].split(" ")[0]
        pack['modified_sequence'] = "".join(
            list(peptide_parser(msms[MSMS_NAME.index("Modified sequence")].strip("_")))[::-1])
        pack['charge'] = int(msms[MSMS_NAME.index("Charge")])
        pack['id'] = int(msms[MSMS_NAME.index("id")])
        re.append(pack)
    return re

def remove_intensities(matches):
    all_intensities = []
    for m in matches:
        raw = m[1]
        inten = raw[3]
        all_intensities.extend([float(i) for i in inten.split(" ")])
    lower = np.quantile(all_intensities, [0.1])[0]
    print("Low intensity", lower)
    remove = 0
    total = 0
    for m in matches:
        raw = m[1]
        inten = raw[3]
        inten = [float(i) for i in inten.split(" ")]
        total += len(inten)
        mass = raw[2]
        mass = [float(i) for i in mass.split(" ")]
        assert len(inten) == len(mass)
        new_inten = []
        new_mass = []
        for i in range(len(inten)):
            if inten[i] <= lower:
                continue
            new_inten.append(inten[i])
            new_mass.append(mass[i])
        remove += len(inten) - len(new_inten)
        raw[3] = " ".join([str(i) for i in new_inten])
        raw[2] = " ".join([str(i) for i in new_mass])
    print(remove, total)
    return matches


def to_tensor(frag_msms):
    frag_msms = [i.reshape(1, -1) for i in frag_msms]
    frag_msms = np.concatenate(frag_msms, axis=0)
    return torch.from_numpy(frag_msms)

def get_irt_all(run_model, data_cand):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    def perpare_data(seqs, nces, charges):
        seqs = torch.from_numpy(seqs)
        nces = torch.from_numpy(nces).unsqueeze(1)
        charges = torch.from_numpy(charges)
        data = {}
        data["sequence_integer"] = seqs.to(device)
        data['peptide_mask'] = ms.helper.create_mask(seqs).to(device)
        return data
    
    run_model = run_model.to(device)
    run_model = run_model.eval()
    with torch.no_grad():
        data = perpare_data(*data_cand)
        rts = []
        for b in range(0, len(data['sequence_integer']), 2048):
            d = {k: v[b:b+2048] for k, v in data.items()}
            pred = run_model(d)
            # check_empty = torch.any(d['peptide_mask'], dim=1)
            # sas[~check_empty] = 0.
            rts.append(pred)
        all_rt = torch.cat(rts, dim=0).cpu().numpy()
    all_rt = all_rt*np.sqrt(constants.iRT_rescaling_var) + constants.iRT_rescaling_mean
    return all_rt


def get_sa_all(run_model, data_nce_cand, frag_msms, gpu_index=0, **kwargs):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")

    def perpare_data(seqs, nces, charges):
        seqs = torch.from_numpy(seqs)
        nces = torch.from_numpy(nces).unsqueeze(1)
        charges = torch.from_numpy(charges)
        data = {}
        data["sequence_integer"] = seqs.to(device)
        data["collision_energy_aligned_normed"] = nces.to(device)
        data['precursor_charge_onehot'] = charges.to(device)
        data['peptide_mask'] = ms.helper.create_mask(seqs).to(device)
        return data

    run_model = run_model.to(device)
    run_model = run_model.eval()
    with torch.no_grad():
        data = perpare_data(*data_nce_cand)
        sass = []
        pred_tensor = []
        for b in range(0, len(frag_msms), 512):
            d = {k: v[b:b+512] for k, v in data.items()}
            pred = run_model(d)
            gt_frag = to_tensor(frag_msms[b:b+512]).to(device)
            # gt_frag = gt_frag/gt_frag.max()
            sas, pred = helper.predict_sa(gt_frag, pred, d)
            # check_empty = torch.any(d['peptide_mask'], dim=1)
            # sas[~check_empty] = 0.
            sass.append(sas.cpu())
            pred_tensor.append(pred.cpu())
        all_sa = torch.cat(sass, dim=0)
        all_pred = torch.cat(pred_tensor, dim=0)
    return all_sa, all_pred


def get_single_score_all(run_model, data_nce_cand, frag_msms):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    def perpare_data(seqs, nces, charges, rt):
        seqs = torch.from_numpy(seqs)
        nces = torch.from_numpy(nces).unsqueeze(1)
        charges = torch.from_numpy(charges)
        frag_msms_torch = torch.from_numpy(frag_msms)
        data = {}
        data["sequence_integer"] = seqs.to(device)
        data["collision_energy_aligned_normed"] = nces.to(device)
        data['precursor_charge_onehot'] = charges.to(device)
        data['intensities_raw'] = frag_msms_torch.to(device)
        data['peptide_mask'] = ms.helper.create_mask(seqs).to(device)
        data['irt'] = torch.from_numpy(rt).to(device)
        return data

    run_model = run_model.to(device)
    run_model = run_model.eval()
    with torch.no_grad():
        data = perpare_data(*data_nce_cand)
        sass = []
        for b in range(0, len(frag_msms), 2048):
            d = {k: v[b:b+2048] for k, v in data.items()}
            sas = run_model(d)
            # gt_frag = gt_frag/gt_frag.max()
            # check_empty = torch.any(d['peptide_mask'], dim=1)
            # sas[~check_empty] = 0.
            sass.append(sas)
        all_sa = torch.cat(sass, dim=0)
    return all_sa

def get_sa_all_scale(run_model, data_nce_cand, frag_msms, inten_scales):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    def perpare_data(seqs, nces, charges):
        seqs = torch.from_numpy(seqs)
        nces = torch.from_numpy(nces).unsqueeze(1)
        charges = torch.from_numpy(charges)
        data = {}
        data["sequence_integer"] = seqs.to(device)
        data["collision_energy_aligned_normed"] = nces.to(device)
        data['precursor_charge_onehot'] = charges.to(device)
        data['peptide_mask'] = ms.helper.create_mask(seqs).to(device)
        return data

    run_model = run_model.to(device)
    run_model = run_model.eval()
    with torch.no_grad():
        data = perpare_data(*data_nce_cand)
        sass = []
        pred_tensor = []
        for b in range(0, len(frag_msms), 2048):
            d = {k: v[b:b+2048] for k, v in data.items()}
            pred = run_model(d)
            gt_frag = to_tensor(frag_msms[b:b+2048]).to(device)
            scales = to_tensor(inten_scales[b:b+2048]).to(device)
            # gt_frag = gt_frag/gt_frag.max()
            sas, pred = helper.predict_sa_scale(gt_frag, pred, d, scales)
            # check_empty = torch.any(d['peptide_mask'], dim=1)
            # sas[~check_empty] = 0.
            sass.append(sas)
            pred_tensor.append(pred)
        all_sa = torch.cat(sass, dim=0)
        all_pred = torch.cat(pred_tensor, dim=0)
    return all_sa, all_pred

def get_sa_random_all(run_model, frag_msms):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    def perpare_data(seqs, nces, charges):
        seqs = torch.from_numpy(seqs)
        nces = torch.from_numpy(nces).unsqueeze(1)
        charges = torch.from_numpy(charges)
        data = {}
        data["sequence_integer"] = seqs.to(device)
        data["collision_energy_aligned_normed"] = nces.to(device)
        data['precursor_charge_onehot'] = charges.to(device)
        data['peptide_mask'] = ms.helper.create_mask(seqs).to(device)
        return data

    run_model = run_model.to(device)
    run_model = run_model.eval()
    with torch.no_grad():
        data = perpare_data(*data_nce_cand)
        sass = []
        pred_tensor = []
        for b in range(0, len(frag_msms), 2048):
            d = {k: v[b:b+2048] for k, v in data.items()}
            pred = run_model(d)
            gt_frag = to_tensor(frag_msms[b:b+2048]).to(device)
            gt_frag = gt_frag/gt_frag.max()
            sas = helper.predict_sa(gt_frag, pred, d)
            check_empty = torch.any(d['peptide_mask'], dim=1)
            sas[~check_empty] = 0.
            sass.append(sas)
            pred_tensor.append(pred)
        all_sa = torch.cat(sass, dim=0)
        all_pred = torch.cat(pred_tensor, dim=0)
    return all_sa, all_pred

def generate_from_msms(msms_data, name, nces=33):
    seqs = [i[name.index("Modified sequence")].strip("_") for i in msms_data]
    seqs = [bio_helper.peptide_to_inter(i) for i in seqs]
    seqs = np.concatenate(seqs)

    charges = [int(i[name.index("Charge")]) for i in msms_data]
    charges = [bio_helper.one_hot(i-1) for i in charges]
    charges = np.concatenate(charges)

    data_nce_cand = [
        seqs,
        np.ones((len(seqs), ), dtype=int)*nces/100.0,
        charges
    ]
    return data_nce_cand


def generate_from_msms_rt(msms_data, name, nces=33):
    seqs = [i[name.index("Modified sequence")].strip("_") for i in msms_data]
    seqs = [bio_helper.peptide_to_inter(i) for i in seqs]
    seqs = np.concatenate(seqs)

    charges = [int(i[name.index("Charge")]) for i in msms_data]
    charges = [bio_helper.one_hot(i-1) for i in charges]
    charges = np.concatenate(charges)

    rt = [float(m[name.index("Retention time")]) for m in msms_data]
    rt = np.array(rt)

    data_nce_cand = [
        seqs,
        np.ones((len(seqs), ), dtype=int)*nces/100.0,
        charges,
        rt.reshape(-1, 1)
    ]
    return data_nce_cand


def generate_from_msms_delta(msms_data, name, nces=33):
    seqs = [i[name.index("All modified sequences")].split(";")[
        1].strip("_") for i in msms_data]
    seqs = [bio_helper.peptide_to_inter(i) for i in seqs]
    seqs = np.concatenate(seqs)

    charges = [int(i[name.index("Charge")]) for i in msms_data]
    charges = [bio_helper.one_hot(i-1) for i in charges]
    charges = np.concatenate(charges)

    data_nce_cand = [
        seqs,
        np.ones((len(seqs), ), dtype=int)*nces/100.0,
        charges
    ]
    return data_nce_cand


def generate_from_msms_reverse(msms_data, name, nces=33):
    seqs = [i[name.index("Modified sequence")].strip("_") for i in msms_data]
    seqs = ["".join(list(peptide_parser(i))[::-1]) for i in seqs]
    seqs = [bio_helper.peptide_to_inter(i) for i in seqs]
    seqs = np.concatenate(seqs)

    charges = [int(i[name.index("Charge")]) for i in msms_data]
    charges = [bio_helper.one_hot(i-1) for i in charges]
    charges = np.concatenate(charges)

    data_nce_cand = [
        seqs,
        np.ones((len(seqs), ), dtype=int)*nces/100.0,
        charges
    ]
    return data_nce_cand


def generate_from_mi(m_i, name, nces=33):
    seqs = [i['modified_sequence'] for i in m_i]
    seqs = [bio_helper.peptide_to_inter(i) for i in seqs]
    seqs = np.concatenate(seqs)

    charges = [i['charge'] for i in m_i]
    charges = [bio_helper.one_hot(i-1) for i in charges]
    charges = np.concatenate(charges)

    data_nce_cand = [
        seqs,
        np.ones((len(seqs), ), dtype=int)*nces/100.0,
        charges
    ]
    return data_nce_cand

def generate_from_msms_random(msms_data, name, nces=33, mutal=.9):
    seqs = [i[name.index("Modified sequence")].strip("_") for i in msms_data]
    seqs = ["".join(random_pick_petides(peptide_parser(i), mutal=mutal))
            for i in seqs]
    seqs = [bio_helper.peptide_to_inter(i) for i in seqs]
    seqs = np.concatenate(seqs)

    charges = [int(i[name.index("Charge")]) for i in msms_data]
    charges = [bio_helper.one_hot(i-1) for i in charges]
    charges = np.concatenate(charges)

    data_nce_cand = [
        seqs,
        np.ones((len(seqs), ), dtype=int)*nces/100.0,
        charges
    ]
    return data_nce_cand

def random_pick_petides(seq, mutal=.9):
    new_seq = []
    for s in seq:
        if random.random() > mutal:
            new_s = random.choice(constants.AMINOS)
            new_seq.append(new_s)
        else:
            new_seq.append(s)
    return new_seq

def read_name(msms_file):
    with open(msms_file) as f:
        for line in f:
            name = line.strip().split("\t")
            break
    return name

def save_m_r_ions(msms_file, raw_dir, sample_size=None):
    name, msms_data = read_msms(
        msms_file)
    msms_data = filter_msms(name, msms_data)
    
    save2 = os.path.splitext(msms_file)[0]+"_ions.txt"
    if sample_size is not None:
        msms_data = sample(msms_data, sample_size)
        save2 = os.path.splitext(msms_file)[0]+f"_{sample_size}_ions.txt"
    msms_data.sort(key=lambda x: int(x[name.index("id")]))
    m_r = loc_msms_in_raw(msms_data, raw_dir)
    m_r = sorted(m_r, key=lambda x: int(x[0][name.index("id")]))
    m_r = filter_m_r(m_r)
    print(len(msms_data), len(m_r))
    matched_ions_pre = generate_matched_ions(m_r)
    matched_ions_pre_delta = generate_matched_ions_delta(m_r)

    m_i_delta = [bio_helper.match_all(i, 'yb')
                 for i in tqdm(matched_ions_pre_delta)]
    m_i = [bio_helper.match_all(i, 'yb') for i in tqdm(matched_ions_pre)]

    with open(save2, 'w') as f:
        for m1, ion_delta, ion in zip(m_r, m_i_delta, m_i):
            mr_line = str(m1)
            delta_line = str(ion_delta)
            ion_line = str(ion)
            f.write("\t".join([mr_line, delta_line, ion_line]) + "\n")
    return name


def save_m_r_ions_reverse(msms_file, raw_dir, sample_size=None):
    name, msms_data = read_msms(
        msms_file)
    msms_data = filter_msms(name, msms_data)
    print(len(msms_data))
    save2 = os.path.splitext(msms_file)[0]+"_ions_reverse.txt"
    if sample_size is not None:
        msms_data = choices(msms_data, k=sample_size)
        save2 = os.path.splitext(msms_file)[0]+f"_{sample_size}_ions_reverse.txt"
    msms_data.sort(key=lambda x: int(x[name.index("id")]))
    m_r = loc_msms_in_raw(msms_data, raw_dir)
    m_r = sorted(m_r, key=lambda x: int(x[0][name.index("id")]))
    print(len(m_r))
    matched_ions_pre = generate_matched_ions(m_r)
    matched_ions_pre_delta = generate_matched_ions_delta(m_r)
    matched_ions_pre_reverse = generate_matched_ions_reverse(m_r)

    m_i_delta = [bio_helper.match_all(i, 'yb')
                 for i in tqdm(matched_ions_pre_delta)]
    m_i = [bio_helper.match_all(i, 'yb') for i in tqdm(matched_ions_pre)]
    m_i_rever = [bio_helper.match_all(i, 'yb')
                 for i in tqdm(matched_ions_pre_reverse)]

    with open(save2, 'w') as f:
        for m1, ion_delta, ion, ion_rever in zip(m_r, m_i_delta, m_i, m_i_rever):
            mr_line = str(m1)
            delta_line = str(ion_delta)
            ion_line = str(ion)
            reverse_line = str(ion_rever)
            f.write(
                "\t".join([mr_line, delta_line, ion_line, reverse_line]) + "\n")
    return name

def read_m_r_ions(save2):
    m_r = []
    m_i_delta = []
    m_i = []
    with open(save2) as f:
        for line in f:
            m1, m2, m3 = line.strip().split("\t")
            m_r.append(eval(m1))
            m_i_delta.append(eval(m2))
            m_i.append(eval(m3))
    return m_r, m_i_delta, m_i


def read_m_r_ions_reverse(save2):
    m_r = []
    m_i_delta = []
    m_i = []
    m_i_rever = []
    with open(save2) as f:
        for line in f:
            m1, m2, m3, m4 = line.strip().split("\t")
            m_r.append(eval(m1))
            m_i_delta.append(eval(m2))
            m_i.append(eval(m3))
            m_i_rever.append(eval(m4))
    return m_r, m_i_delta, m_i, m_i_rever

def generate_peaks_info(m_ions_pre, cores=4):
    from multiprocessing import Pool
    with Pool(cores) as p:
        m_i = p.map(bio_helper.match_all_multi, m_ions_pre)
    m_i = sorted(m_i, key=lambda x: x[0])
    m_i = [m[1:] for m in m_i]
    return m_i

if __name__ == "__main__":
    from time import time
    msms_f = "/data/prosit/figs/figure6/IGC/maxquant/txt/msms.txt"
    raw_files = "/data/prosit/figs/figure6/all_raws"
    name, msms = read_msms(msms_f)
    print(len(msms))
    msms = filter_msms(name, msms)
    print(len(msms))
    m_r = loc_msms_in_raw(msms, raws_dir=raw_files)
    m_r = sorted(m_r, key=lambda x: int(x[0][name.index("id")]))
    print(len(m_r))
    matched_ions_pre = generate_matched_ions(m_r)
    start = time()
    m_i = [bio_helper.match_all(i, 'yb') for i in matched_ions_pre]
    print(time() - start)

    start = time()
    generate_peaks_info(matched_ions_pre)
    print(time() - start)

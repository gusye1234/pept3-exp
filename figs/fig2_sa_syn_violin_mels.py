import cProfile
import pandas as pd
import seaborn as sns
import numpy as np
import os
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import bio_helper
from tools import *
from fdr_test import fixed_features
import sys
sys.path.append("..")
from pept3 import model
plt.rcParams['svg.fonttype'] = 'none'


def index_psmid_table(table_file, psmid):
    table = pd.read_csv(table_file, sep='\t')
    return table[table['SpecId'].apply(lambda x: x in psmid)]


def shared_psmid(nf_psms, f_psms, threshold=0.01):
    nf = pd.read_csv(nf_psms, sep='\t')
    f = pd.read_csv(f_psms, sep='\t')
    nf_psmid = set(["-".join(i.split('-')[:-1])
                   for i in nf[nf['q-value'] <= threshold]['PSMId']])
    f_psmid = set(["-".join(i.split('-'))
                  for i in f[f['q-value'] <= threshold]['PSMId']])
    return nf_psmid - f_psmid, nf_psmid.intersection(f_psmid), f_psmid - nf_psmid


def shared_peptide(nf_psms, f_psms, threshold=0.01):
    nf = pd.read_csv(nf_psms, sep='\t')
    f = pd.read_csv(f_psms, sep='\t')
    nf_psmid = set(nf[nf['q-value'] <= threshold]
                   ['peptide'].apply(lambda x: x.strip("_").strip(".")))
    f_psmid = set(f[f['q-value'] <= threshold]
                  ['peptide'].apply(lambda x: x.strip("_").strip(".")))
    pep_psmid = {i: j for i, j in zip(f[f['q-value'] <= threshold]['peptide'].apply(
        lambda x: x.strip("_").strip(".")), f[f['q-value'] <= threshold]['PSMId'])}
    return nf_psmid - f_psmid, nf_psmid.intersection(f_psmid), f_psmid - nf_psmid, pep_psmid


want_field = ['collision_energy_aligned_normed', 'sequence_integer',
              'precursor_charge_onehot', 'intensities_raw', 'score']


def filter_HLA(file):
    data = h5py.File(file, 'r')
    index = []
    rawfiles = np.array(data['rawfile']).astype("str")
    for i in rawfiles:
        index.append("HLA" in i)
    index = np.array(index)
    re_data = {}
    for k in want_field:
        re_data[k] = np.array(data[k])[index]
    return re_data


temp_DATA = []
temp_DATA.append(filter_HLA(
    '/data/yejb/prosit/figs/boosting/train/prediction_hcd_train.hdf5'))
temp_DATA.append(filter_HLA(
    '/data/yejb/prosit/figs/boosting/train/prediction_hcd_val.hdf5'))
temp_DATA.append(filter_HLA(
    '/data/yejb/prosit/figs/boosting/train/prediction_hcd_ho.hdf5'))

DATA = {}
for k in want_field:
    DATA[k] = np.concatenate([temp[k] for temp in temp_DATA])
del temp_DATA


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1),)
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_xlim(0.25, len(labels) + 0.75)
    # ax.set_xlabel('Sample name')


def plot_half_violin(data_dict: dict, names=['No fine-tuned', 'Fine-tuned']):
    fig, ax = plt.subplots(figsize=(4, 6), dpi=100)
    labels = list(data_dict)

    no_finetuned = [v[0] for v in data_dict.values()]
    finetuned = [v[1] for v in data_dict.values()]

    mins = [min(np.min(i), np.min(j)) for i, j in zip(no_finetuned, finetuned)]
    maxs = [max(np.max(i), np.max(j)) for i, j in zip(no_finetuned, finetuned)]

    plot1 = ax.violinplot(no_finetuned, showmeans=False,
                          showextrema=False, showmedians=False)
    for b in plot1['bodies']:
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(
            b.get_paths()[0].vertices[:, 0], -np.inf, m)
        b.set_edgecolor('lightgray')
        # b.set_edgewidth(2)
    plot2 = ax.violinplot(finetuned, showmeans=False,
                          showextrema=False, showmedians=True)
    for b in plot2['bodies']:
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further left than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(
            b.get_paths()[0].vertices[:, 0], m, np.inf)
        # b.set_color('b')
        b.set_edgecolor('lightgray')
        # b.set_edgewidth(2)
    for i in range(len(labels)):
        ax.text(i + 1 - 0.2, 0.4,
                f"n={len(finetuned[i])}", fontsize=8, rotation=90, va='center')
    # ax.vlines([i+1 for i in range(len(labels))], mins, maxs, color='gray', linestyles='--', lw=1)

    x_axises = np.array([i + 1 for i in range(len(labels))])
    nf_sa_mean = [np.mean(i) for i in no_finetuned]
    f_sa_mean = [np.mean(i) for i in finetuned]

    ax.hlines(nf_sa_mean, x_axises - 0.2, x_axises,
              color='slateblue', linestyles='-', lw=1)
    ax.hlines(f_sa_mean, x_axises, x_axises + 0.2,
              color='orange', linestyles='-', lw=1)
    ax.legend([plot1['bodies'][0], plot2['bodies'][0]],
              names, loc='lower right', frameon=False)
    set_axis_style(ax, labels)
    ax.set_ylabel("Spectral Angle", fontsize=15)
    ax.set_ylim(0, 1)
    return fig, ax


def psmid2charge(psms):
    charges = [int(i.split("-")[-1]) for i in psms]
    return charges


def seq2int(S):
    len = S.shape[-1]
    base = 22
    base_p = np.array([
        [base**i if i < len // 2 else 0 for i in range(len)],
        [base**(i - len // 2) if i >= len // 2 else 0 for i in range(len)]], dtype='int64')
    re = S @ base_p.T
    return re


INT_S = seq2int(DATA['sequence_integer']).squeeze()
print(INT_S.shape)


def find_syn_spectrum(peptides, charges, Global_S):
    from time import time
    start = time()
    seqs = [bio_helper.peptide_to_inter(i) for i in peptides]
    seqs = np.concatenate(seqs)

    charges = np.array(charges).squeeze() - 1

    found = 0
    data_charges = np.argmax(DATA['precursor_charge_onehot'], axis=1)
    Global_S = Global_S.squeeze()
    pre_data = {}
    pre_data['peptides'] = []
    int_seqs = seq2int(seqs).squeeze()
    G1 = Global_S[:, 0]
    G2 = Global_S[:, 1]
    for count, (seq, c) in enumerate(zip(int_seqs, charges)):
        sys.stdout.write(f"Found {found}/{count+1}\r")
        sys.stdout.flush()

        _p1 = (G1 == seq[0])
        _p2 = (G2 == seq[1])
        p_index = np.logical_and(_p1, _p2)

        # _p_index = np.all(DATA['sequence_integer'] == seqs[count], axis=1)
        # assert np.all(p_index == _p_index)
        c_index = (data_charges == c)

        index = np.logical_and(p_index.reshape(-1), c_index.reshape(-1))

        if np.sum(index) == 0:
            continue
        else:
            found += 1
        arg_select = np.argmax(DATA['score'][index])
        pre_data['peptides'].append(peptides[count])
        for k in want_field:
            pre_item = DATA[k][index][arg_select].reshape(1, -1)
            if k in pre_data:
                pre_data[k] = np.concatenate([pre_data[k], pre_item])
            else:
                pre_data[k] = pre_item
    print()
    return pre_data


frag_model = "prosit_l1"
if frag_model == "prosit_cid":
    run_model = model.PrositFrag()
    run_model.load_state_dict(torch.load(
        "../checkpoints/frag_boosting/best_cid_frag_PrositFrag-512.pth", map_location="cpu"))
    run_model = run_model.eval()
elif frag_model == "prosit_hcd":
    run_model = model.PrositFrag()
    run_model.load_state_dict(torch.load(
        "../checkpoints/frag_boosting/best_hcd_frag_PrositFrag-512.pth", map_location="cpu"))
    run_model = run_model.eval()
elif frag_model == "prosit_l1":
    run_model = model.PrositFrag()
    run_model.load_state_dict(torch.load(
        "../checkpoints/frag_boosting/best_frag_l1_PrositFrag-1024.pth", map_location="cpu"))
    run_model = run_model.eval()


def pick_finetuned_model(path):
    run_model1 = model.PrositFrag()
    run_model2 = model.PrositFrag()
    weights = torch.load(os.path.join(
        path, f"{frag_model}.pth"), map_location="cpu")
    run_model1.load_state_dict(weights[0])
    run_model2.load_state_dict(weights[1])
    run_model1 = run_model1.eval()
    run_model2 = run_model2.eval()
    return run_model1, run_model2


def track_raw_spectrum(rawfile, scannum, intensities_raw, psmids):
    intens = []
    for psmid in psmids:
        packs = psmid.split("-")
        charge = int(packs[-1])
        pep = packs[-2]
        sn = int(packs[-3])
        rf = '-'.join(packs[:-3])

        index = np.logical_and(
            (rawfile == rf.encode()).reshape(-1), (scannum == sn).reshape(-1))
        intens.append(intensities_raw[index])
    return np.concatenate(intens)


nces = 32
hla_mel = pd.read_csv("./data/HLA_Mel.csv")
hla_mel = hla_mel[hla_mel['Experiment'].apply(
    lambda x: x.endswith("HLA-I"))]
Mels = hla_mel['Experiment'].unique()
set_threshold = 0.1
which_part_str = 'share'
if not os.path.exists("figs/fig2_sa_violin"):
    os.mkdir("figs/fig2_sa_violin")
RAW = False
if RAW:
    # for which in Mels:
    for which in ['Mel-15_HLA-I']:
        print("-------------------------------")
        f_model_path = os.path.join('../checkpoints/finetuned/HLA-I', which)
        print("load from", f_model_path)
        f_model1, f_model2 = pick_finetuned_model(f_model_path)
        f_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/{frag_model}/percolator_hdf5_Mels_{set_threshold}/{which}"
        nf_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/forPride/rescoring_for_paper_2/Mels/{which}/percolator"
        f_peps = os.path.join(f_tab, "prosit_target.peptides")
        nf_peps = os.path.join(nf_tab, "prosit_target.peptides")
        hdf5_file = os.path.join(nf_tab, "../data.hdf5")

        hdf5_data = h5py.File(hdf5_file, "r")
        hdf5_rawfile = np.array(hdf5_data['rawfile'])
        hdf5_scannum = np.array(hdf5_data['scan_number'])
        hdf5_intens = np.array(hdf5_data['intensities_raw'])

        L, S, G, pep2psmid = shared_peptide(nf_peps, f_peps)

        # print(list(pep2psmid.values())[:10])
        # break
        which_part = {
            'lost': list(L), 'share': list(S), 'gain': list(G)
        }[which_part_str]
        print("Search", len(which_part))

        part_charges = psmid2charge([pep2psmid[i] for i in which_part])
        pre_data = find_syn_spectrum(which_part, part_charges, INT_S)

        candi_peps = pre_data['peptides']
        raw_intens = track_raw_spectrum(hdf5_rawfile, hdf5_scannum, hdf5_intens, [
                                        pep2psmid[i] for i in candi_peps])
        # print(raw_intens.shape)
        # print({k:v.shape for k, v in pre_data.items() if isinstance(v, np.ndarray)})

        frag_msms = pre_data['intensities_raw']
        # print(pre_data['collision_energy_aligned_normed'].min(), pre_data['collision_energy_aligned_normed'].max())
        data_nce_cand = [
            pre_data['sequence_integer'].astype('int'),
            # pre_data['collision_energy_aligned_normed'],
            np.ones((len(pre_data['sequence_integer']), ),
                    dtype=int) * nces / 100.0,
            pre_data['precursor_charge_onehot'].astype("int")
        ]
        prosit_sa, prosit_inten = get_sa_all(
            run_model, data_nce_cand, frag_msms, pearson=(frag_model == 'pdeep2'))
        prosit_sa = prosit_sa.cpu().numpy()
        prosit_inten = prosit_inten.cpu().numpy()

        sa_data = {
            'sequence_integer': torch.from_numpy(data_nce_cand[0]),
            'precursor_charge_onehot': torch.from_numpy(data_nce_cand[2]),
        }
        raw_sa, _ = helper.predict_sa(torch.from_numpy(
            frag_msms), torch.from_numpy(raw_intens), sa_data)
        raw_sa = raw_sa.cpu().numpy()

        data_dict = {which: (prosit_sa, raw_sa)}
        fig, ax = plot_half_violin(
            data_dict, names=['Prosit 2019', 'Raw spectra'])
        fig.savefig(
            f"figs/fig2_sa_violin/{frag_model}-{which_part_str}-{which}-raw.svg", dpi=300)
        plt.close()
else:
    # for which in Mels:
    for which in ['Mel-15_HLA-I']:
        # for which in ['Mel-28_HLA-I']:
        print("-------------------------------")
        f_model_path = os.path.join('../checkpoints/finetuned/HLA-I', which)
        print("load from", f_model_path)
        f_model1, f_model2 = pick_finetuned_model(f_model_path)
        f_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/{frag_model}/percolator_hdf5_Mels_{set_threshold}/{which}"
        nf_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/forPride/rescoring_for_paper_2/Mels/{which}/percolator"
        f_peps = os.path.join(f_tab, "prosit_target.peptides")
        nf_peps = os.path.join(nf_tab, "prosit_target.peptides")
        hdf5_file = os.path.join(nf_tab, "../data.hdf5")

        hdf5_data = h5py.File(hdf5_file, "r")
        hdf5_rawfile = np.array(hdf5_data['rawfile'])
        hdf5_scannum = np.array(hdf5_data['scan_number'])
        hdf5_intens = np.array(hdf5_data['intensities_raw'])

        L, S, G, pep2psmid = shared_peptide(nf_peps, f_peps)

        # print(list(pep2psmid.values())[:10])
        # break
        which_part = {
            'lost': list(L), 'share': list(S), 'gain': list(G)
        }[which_part_str]
        print("Search", len(which_part))

        part_charges = psmid2charge([pep2psmid[i] for i in which_part])
        pre_data = find_syn_spectrum(which_part, part_charges, INT_S)

        candi_peps = pre_data['peptides']
        # print(raw_intens.shape)
        # print({k:v.shape for k, v in pre_data.items() if isinstance(v, np.ndarray)})

        frag_msms = pre_data['intensities_raw']
        # print(pre_data['collision_energy_aligned_normed'].min(), pre_data['collision_energy_aligned_normed'].max())
        data_nce_cand = [
            pre_data['sequence_integer'].astype('int'),
            # pre_data['collision_energy_aligned_normed'],
            np.ones((len(pre_data['sequence_integer']), ),
                    dtype=int) * nces / 100.0,
            pre_data['precursor_charge_onehot'].astype("int")
        ]
        prosit_sa, prosit_inten = get_sa_all(
            run_model, data_nce_cand, frag_msms, pearson=(frag_model == 'pdeep2'))
        prosit_sa = prosit_sa.cpu().numpy()
        prosit_inten = prosit_inten.cpu().numpy()

        f_sas = []
        f_spectra = []
        for ft_model in [f_model1, f_model2]:
            finetune_prosit_sa, finetune_prosit_inten = get_sa_all(
                ft_model, data_nce_cand, frag_msms, pearson=(frag_model == 'pdeep2'))
            finetune_prosit_sa = finetune_prosit_sa.cpu().numpy()
            finetune_prosit_inten = finetune_prosit_inten.cpu().numpy()
            f_sas.append(finetune_prosit_sa)
            f_spectra.append(finetune_prosit_inten)

        finetune_prosit_sa = (f_sas[0] + f_sas[1]) / 2
        # finetune_prosit_sa = f_sas[0]
        finetune_prosit_inten = (f_spectra[0] + f_spectra[1]) / 2

        data_dict = {which: (prosit_sa, finetune_prosit_sa)}
        fig, ax = plot_half_violin(
            data_dict, names=['Prosit 2019', 'Fine-tuned Prosit'])
        fig.savefig(
            f"figs/fig2_sa_violin/{frag_model}-{which_part_str}-{which}-ft.svg", dpi=300)
        plt.close()

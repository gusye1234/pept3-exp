import sys
sys.path.append("./figs")
from contextlib import redirect_stdout
import pandas as pd
import torch
import os
from time import time
from ms import helper
from ms import model
from ms import finetune
import h5py
from tqdm import tqdm
import numpy as np
from tools import get_sa_from_array
from fdr_test import fdr_test, fixed_features, fdr_test_twofold
from copy import deepcopy

# from figs.fdr_test import fdr_test_reverse


def overlap_analysis(tab1, tab2, testfdr=0.01, compare=["sa", "sa"]):
    baseline = "sa"
    table1 = pd.read_csv(os.path.join(
        tab1, f"{compare[0]}_target.psms"), sep='\t')
    table2 = pd.read_csv(os.path.join(
        tab2, f"{compare[1]}_target.psms"), sep='\t')

    id1 = set(table1[table1['q-value'] < testfdr]['PSMId'])
    id2 = set(table2[table2['q-value'] < testfdr]['PSMId'])
    overlap = id1.intersection(id2)
    union = id1.union(id2)
    print(f"{compare}-{testfdr}:", (len(id1) - len(overlap)) / len(union),
          len(overlap) / len(union), (len(id2) - len(overlap)) / len(union))
    return len(id1) - len(overlap), len(overlap), len(id2) - len(overlap)


def eval_fdr(run_model1, run_model2, table_file, feature_csv, origin_prosit_tab, save_tab, fdr_threshold=0.01, show_fdr=[0.1, 0.01, 0.001, 0.0001], sample_size=None, need_all=False, irt_model=None, id2remove=None, pearson=False, gpu_index=0):
    run_model1 = run_model1.eval()
    run_model2 = run_model2.eval()

    record = {}
    record['fdrs'] = [100 * i for i in show_fdr]
    totest = ["prosit"]

    with torch.no_grad():
        print("Starting scoring")
        data = h5py.File(table_file, 'r')
        sample_size = len(data['sequence_integer']
                          ) if sample_size is None else sample_size
        seq_data = np.array(data['sequence_integer'][:sample_size])
        charges = np.array(data['precursor_charge_onehot'][:sample_size])
        nces = np.array(data['collision_energy_aligned_normed'][:sample_size])
        frag_msms = np.array(data['intensities_raw'][:sample_size])
        label = np.array(data['reverse'][:sample_size]).astype("int").squeeze()
        scan_number = np.array(data['scan_number'][:sample_size]).squeeze()
        rawfiles = np.array(data['rawfile'][:sample_size]).squeeze()
        del data

        label[label == 1] = -1
        label[label == 0] = 1
        total_index = np.arange(len(label))
        target_index = total_index[label == 1]
        decoy1_index = np.array(id2remove)
        decoy1_index = decoy1_index[decoy1_index < sample_size]
        decoy2_index = np.array([i for i in total_index if (
            label[i] == -1 and i not in decoy1_index)])
        print(
            f"Comb: {len(target_index)}, {len(decoy1_index)}, {len(decoy2_index)}")
        # --------------------------------------------
        t_sa_1, t_sa_t_1 = get_sa_from_array(
            run_model1, seq_data[target_index], nces[target_index],
            charges[target_index], frag_msms[target_index], gpu_index=gpu_index)
        t_sa_2, t_sa_t_2 = get_sa_from_array(
            run_model2, seq_data[target_index], nces[target_index],
            charges[target_index], frag_msms[target_index], gpu_index=gpu_index)
        t_sa = ((t_sa_1 + t_sa_2) / 2).cpu().numpy()
        t_sa_t = ((t_sa_t_1 + t_sa_t_2) / 2).cpu().numpy()
        # --------------------------------------------
        d2_sa, d2_sa_t = get_sa_from_array(
            run_model1, seq_data[decoy2_index], nces[decoy2_index],
            charges[decoy2_index], frag_msms[decoy2_index], gpu_index=gpu_index)
        d1_sa, d1_sa_t = get_sa_from_array(
            run_model2, seq_data[decoy1_index], nces[decoy1_index],
            charges[decoy1_index], frag_msms[decoy1_index], gpu_index=gpu_index)
        d2_sa = d2_sa.cpu().numpy()
        d2_sa_t = d2_sa_t.cpu().numpy()
        d1_sa = d1_sa.cpu().numpy()
        d1_sa_t = d1_sa_t.cpu().numpy()
        # --------------------------------------------
        sas = np.concatenate([t_sa, d1_sa, d2_sa], axis=0)
        sas_tensors = np.concatenate([t_sa_t, d1_sa_t, d2_sa_t], axis=0)
        orders = np.concatenate(
            [target_index, decoy1_index, decoy2_index], axis=0)
        orders = np.argsort(orders)
        sas = sas[orders]
        sas_tensors = sas_tensors[orders]

        feature_table = pd.read_csv(feature_csv, nrows=sample_size)
        assert (len(scan_number) == len(feature_table))
        assert all(
            [True if p[0].decode() == p[1] else False for p in zip(rawfiles, np.array(feature_table['raw_file']))])
        assert (scan_number == np.array(feature_table['scan_number'])).all()
        Rawfile = feature_table['raw_file'].to_list()
        Charges = np.array(feature_table['precursor_charge'])
        Features = {}
        Features['spectral_angle'] = sas
        Features["KR"] = feature_table['KR']
        Features["andromeda"] = feature_table['score']
        Features["iRT"] = feature_table['iRT']
        Features['collision_energy_aligned_normed'] = feature_table['collision_energy_aligned_normed']
        Features["Protein"] = feature_table['sequence']
        Features["Peptide"] = feature_table['sequence'].apply(
            lambda x: "_." + x + "._")

        pack = [(None, None, sa, st, None, frag)
                for sa, st, frag in zip(sas, sas_tensors, frag_msms)]

        def add_pred(pack):

            def b(tensor):
                return tensor.reshape(29, 2, 3)[:, 1, :]

            def y(tensor):
                return tensor.reshape(29, 2, 3)[:, 0, :]

            Features['not_pred_seen'] = [
                np.sum(m[3][m[5] > 0] == 0) for m in pack]
            Features['not_pred_seen_b'] = [
                np.sum(b(m[3])[b(m[5]) > 0] == 0) for m in pack]
            Features['not_pred_seen_y'] = [
                np.sum(y(m[3])[y(m[5]) > 0] == 0) for m in pack]
            Features['pred_nonZero_fragments'] = [
                np.sum(m[3] > 0) for m in pack]
            Features['pred_nonZero_b'] = [np.sum(b(m[3]) > 0) for m in pack]
            Features['pred_nonZero_y'] = [np.sum(y(m[3]) > 0) for m in pack]
            Features['pred_not_seen'] = [
                np.sum(m[5][m[3] > 0] == 0) for m in pack]
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
            Features['raw_nonZero_fragments'] = [
                np.sum(m[5] > 0) for m in pack]
            Features['raw_nonZero_b'] = [np.sum(b(m[5]) > 0) for m in pack]
            Features['raw_nonZero_y'] = [np.sum(y(m[5]) > 0) for m in pack]

            theoretically = Features['sequence_length'] * 2 * Charges + 1e-9
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

        def retrieve_from_prosit(origin_prosit_tab):
            prosit = pd.read_csv(
                origin_prosit_tab, sep='\t')
            prosit["SpecId"] = prosit["SpecId"].apply(
                lambda x: "-".join(x.split('-')[:-1]))
            assert prosit['SpecId'].is_unique
            prosit = prosit.set_index("SpecId")

            our_name = [f"{r}-{s}-{p}-{c}" for r, s, p, c in zip(
                Rawfile, scan_number, Features['Protein'], Charges)]
            wanted_features = ['Label', "ScanNr", 'ExpMass', 'Mass', 'deltaM_ppm', 'deltaM_da',
                               'absDeltaM_ppm', 'absDeltaM_da', 'missedCleavages', 'sequence_length',
                               'delta_sa', 'Charge1', 'Charge2', 'Charge3', 'Charge4', 'Charge5',
                               'Charge6']
            for feat in tqdm(wanted_features):
                Features[feat] = np.array(prosit.loc[our_name][feat])
            Features["SpecId"] = our_name

        retrieve_from_prosit(origin_prosit_tab)
        add_pred(pack)
        prosit_features = ["SpecId", 'Label', "ScanNr", 'ExpMass', 'Mass', 'deltaM_ppm', 'deltaM_da',
                           'absDeltaM_ppm', 'absDeltaM_da', 'missedCleavages', 'sequence_length',
                           "spectral_angle", "delta_sa", "andromeda", "iRT", "KR", "collision_energy_aligned_normed"] + \
            "raw_nonZero_fragments  raw_nonZero_y   raw_nonZero_b   pred_nonZero_fragments  pred_nonZero_y  pred_nonZero_b  pred_not_seen  pred_not_seen_y pred_not_seen_b pred_seen_zero  pred_seen_zero_y        pred_seen_zero_b      pred_seen_nonzero        pred_seen_nonzero_y     pred_seen_nonzero_b     not_pred_seen   not_pred_seen_y not_pred_seen_b rel_pred_nonZero_y      rel_pred_nonZero_b      rel_pred_not_seen       rel_pred_not_seen_y    rel_pred_not_seen_b     rel_pred_seen_zero      rel_pred_seen_zero_y    rel_pred_seen_zero_b  rel_pred_seen_nonzero    rel_pred_seen_nonzero_y rel_pred_seen_nonzero_b rel_not_pred_seen       rel_not_pred_seen_y    rel_not_pred_seen_b     relpred_pred_not_seen2pred_nonZero_fragments    relpred_pred_not_seen_y2pred_nonZero_y relpred_pred_not_seen_b2pred_nonZero_b  relpred_pred_seen_zero2pred_nonZero_fragments  relpred_pred_seen_zero_y2pred_nonZero_y relpred_pred_seen_zero_b2pred_nonZero_b relpred_pred_seen_nonzero2pred_nonZero_fragments       relpred_pred_seen_nonzero_y2pred_nonZero_y      relpred_pred_seen_nonzero_b2pred_nonZero_b     relpred_not_pred_seen2pred_nonZero_fragments    relpred_not_pred_seen_y2pred_nonZero_y relpred_not_pred_seen_b2pred_nonZero_b  rel_raw_nonZero_b       rel_raw_nonZero_y     rel_raw_nonZero_fragments".split() + \
            ['Charge1', 'Charge2', 'Charge3', 'Charge4',
                'Charge5', 'Charge6', "Peptide", "Protein"]

        prosit_tab = pd.DataFrame(Features)
        prosit_tab = prosit_tab[prosit_features]
        prosit_tab.to_csv(f"{save_tab}/prosit.tab", sep='\t', index=False)
    print(" start percolator... ")
    for name in totest:
        start = time()
        os.system(f"percolator -v 0 --weights {save_tab}/{name}_weights.csv \
                 -Y --testFDR 0.01 --trainFDR 0.01 \
                --results-psms {save_tab}/{name}_target.psms \
                --results-peptides {save_tab}/{name}_target.peptides \
                --decoy-results-psms {save_tab}/{name}_decoy.psms \
                --decoy-results-peptides {save_tab}/{name}_decoy.peptides \
                {save_tab}/{name}.tab")
        target_tab = pd.read_csv(os.path.join(
            save_tab, f"{name}_target.psms"), sep='\t')
        record[name] = []
        for fdr in show_fdr:
            record[name].append((target_tab['q-value'] < fdr).sum())
        print(f"{name}:{time()-start:.1f}", end='-')
    print()
    return pd.DataFrame(record)


if __name__ == "__main__":
    run_model = model.PrositIRT()
    run_model.load_state_dict(torch.load(
        f"./checkpoints/irt/best_valid_irt_{run_model.comment()}-1024.pth", map_location="cpu"))
    prosit_irt = run_model.eval()

    frag_model = "prosit_hcd"
    if frag_model == "prosit_cid":
        run_model = model.PrositFrag()
        run_model.load_state_dict(torch.load(
            "./checkpoints/frag_boosting/best_cid_frag_PrositFrag-512.pth", map_location="cpu"))
        run_model = run_model.eval()
    elif frag_model == "prosit_hcd":
        run_model = model.PrositFrag()
        run_model.load_state_dict(torch.load(
            "./checkpoints/frag_boosting/best_hcd_frag_PrositFrag-512.pth", map_location="cpu"))
        run_model = run_model.eval()
    elif frag_model == "prosit_l1":
        run_model = model.PrositFrag()
        run_model.load_state_dict(torch.load(
            "./checkpoints/frag_boosting/best_frag_l1_PrositFrag-1024.pth", map_location="cpu"))
        run_model = run_model.eval()

    sample_size = None
    gpu_index = 5
    set_threshold = 0.1
    max_epochs = 20
    print("Running twofold", frag_model)
    if_pearson = (frag_model in ['pdeep2'])
    hla_mel = pd.read_csv("./figs/data/HLA_Mel.csv")
    hla_mel = hla_mel[hla_mel['Experiment'].apply(
        lambda x: x.endswith("HLA-I"))]
    Mels = hla_mel['Experiment'].unique()
    for which in Mels:
        print("-------------------------------")
        print("boosting figure3", which)
        save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/{frag_model}"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/percolator_hdf5_Mels_{set_threshold}/"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/percolator_hdf5_Mels_{set_threshold}/{which}"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        feature_csv = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/forPride/rescoring_for_paper_2/Mels/{which}/percolator/features.csv"
        origin_prosit_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/forPride/rescoring_for_paper_2/Mels/{which}/percolator/prosit.tab"
        tabels_file = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/forPride/rescoring_for_paper_2/Mels/{which}/data.hdf5"
        finetune_model1, finetune_model2, id2remove = finetune.semisupervised_finetune_twofold(
            run_model, tabels_file, max_epochs=max_epochs, pearson=if_pearson, gpu_index=gpu_index, only_id2remove=False, q_threshold=set_threshold)
        print(eval_fdr(finetune_model1, finetune_model2, tabels_file, feature_csv, origin_prosit_tab, save_tab,
                       irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove, pearson=if_pearson, gpu_index=gpu_index).to_string())

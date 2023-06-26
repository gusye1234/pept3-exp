import sys
from contextlib import redirect_stdout
import pandas as pd
import torch
import os
from time import time
import h5py
from tqdm import tqdm
import numpy as np
from .tools import get_sa_from_array


def eval_fdr_hdf5(models, table_file, feature_csv, origin_prosit_tab, save_tab, fdr_threshold=0.01,
                  show_fdr=[0.1, 0.01, 0.001, 0.0001], sample_size=None, need_all=False, irt_model=None,
                  id2selects=None, pearson=False, gpu_index=0):
    """
    Retrieve relevant features from HDF5 files, and generate DL features for filling back.
    """
    models = [m.eval() for m in models]

    record = {}
    record['fdrs'] = [100 * i for i in show_fdr]
    totest = ["prosit"]

    with torch.no_grad():
        data = h5py.File(table_file, 'r')
        sample_size = len(data['sequence_integer']
                          ) if sample_size is None else sample_size
        seq_data = np.array(data['sequence_integer'][:sample_size])
        charges = np.array(data['precursor_charge_onehot'][:sample_size])
        nces = np.array(data['collision_energy_aligned_normed'][:sample_size])
        frag_msms = np.array(data['intensities_raw'][:sample_size])
        label = np.array(data['reverse'][:sample_size]).astype("int").squeeze()
        scan_number = np.array(data['scan_number'][:sample_size]).squeeze()
        del data

        label[label == 1] = -1
        label[label == 0] = 1
        total_index = np.arange(len(label))
        
        sas_list = []
        sas_tensor_list = [] 
        orders = []
        for now_i, (model, ids) in enumerate(zip(models, id2selects)):
            ids = ids[ids < sample_size] 
            print(f">> generating fold-{now_i} [{len(ids)}]...")
            sa, sa_tensor = get_sa_from_array(model,
                                              seq_data[ids],
                                              nces[ids],
                                              charges[ids], 
                                              frag_msms[ids], 
                                              gpu_index=gpu_index)
            sa = sa.cpu().numpy()
            sa_tensor = sa_tensor.cpu().numpy()
            
            sas_list.append(sa)
            sas_tensor_list.append(sa_tensor)
            orders.append(ids)
        # --------------------------------------------
        sas = np.concatenate(sas_list, axis=0)
        sas_tensors = np.concatenate(sas_tensor_list, axis=0)
        orders = np.concatenate(orders, axis=0)
        orders = np.argsort(orders)
        sas = sas[orders]
        sas_tensors = sas_tensors[orders]

        feature_table = pd.read_csv(feature_csv, nrows=sample_size)
        assert (len(scan_number) == len(feature_table))
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
        print(f"{name}:{time()-start:.1f}", end='...')
    print()
    return pd.DataFrame(record)


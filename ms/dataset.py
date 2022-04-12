import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .msms import *


class IrtDataset:
    def __init__(self, file):
        self.tr, self.va, self.te = self.init(file)

    def init(self, file):
        data = h5py.File(file, 'r')

        # hfd5 file error
        x_tr = torch.from_numpy(np.array(data['Y_train']))
        y_tr = torch.from_numpy(np.array(data['X_train'])).float()

        x_va = torch.from_numpy(np.array(data['X_val']))
        x_te = torch.from_numpy(np.array(data['X_holdout']))
        y_va = torch.from_numpy(np.array(data['Y_val'])).float()
        y_te = torch.from_numpy(np.array(data['Y_holdout'])).float()
        return (x_tr, y_tr), (x_va, y_va), (x_te, y_te)

    def train(self):
        print(f"Load {len(self.tr[0])} for train")
        return MatchDataset(self.tr[0], self.tr[1])

    def valid(self):
        print(f"Load {len(self.va[0])} for valid")
        return MatchDataset(self.va[0], self.va[1])

    def test(self):
        print(f"Load {len(self.te[0])} for test")
        return MatchDataset(self.te[0], self.te[1])


class MatchDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y)
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {'sequence_integer': self.x[index].long(), "irt": self.y[index]}


class FragDataset:
    def __init__(self, file,
                 val_file=None,
                 test_file=None,
                 xlabel=["sequence_integer",
                         "precursor_charge_onehot",
                         "collision_energy_aligned_normed"],
                 ylabel="intensities_raw",
                 ratio=0.8,
                 transform=None):
        self.x, self.y = self.init(file, xlabel, ylabel)
        self.names = xlabel + [ylabel]
        if test_file:
            self.test_x, self.test_y = self.init(test_file, xlabel, ylabel)
        else:
            self.test_x, self.test_y = self.vx, self.vy
        if val_file is None:
            self.tx, self.ty, self.vx, self.vy = self.split(ratio)
        else:
            self.tx, self.ty = self.x, self.y
            self.vx, self.vy = self.init(val_file, xlabel, ylabel)

    def init(self, file, xlabel, ylabel):
        data = h5py.File(file, 'r')
        x_data = [np.array(data.get(i)) for i in xlabel]
        y_data = np.array(data.get(ylabel))

        x_data = [torch.from_numpy(i) for i in x_data]
        y_data = torch.from_numpy(y_data)
        return x_data, y_data

    def split(self, ratio):
        cutoff = int(len(self.y)*ratio)
        return [i[:cutoff] for i in self.x], self.y[:cutoff], [i[cutoff:] for i in self.x], self.y[cutoff:]

    def train(self):
        print(f"Load Train: {len(self.ty)}")
        return TableDataset(self.names, self.tx, self.ty)

    def valid(self):
        print(f"Load Valid: {len(self.vy)}")
        return TableDataset(self.names, self.vx, self.vy)

    def test(self):
        print(f"Load Test: {len(self.test_y)}")
        return TableDataset(self.names, self.test_x, self.test_y)


class TableDataset(Dataset):
    def __init__(self, names, xs, y):
        self.x = xs
        self.y = y
        self.names = names

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        re = [i[index] for i in self.x]
        re.append(self.y[index])
        return {self.names[i]: re[i] for i in range(len(re))}


class SemiDataset:
    def __init__(self, table_input, score_init="andromeda", pi=0.9):
        self._file_input = table_input
        self._data = pd.read_csv(table_input, sep='\t').sample(
            frac=1, random_state=2022).reset_index(drop=True)
        self._pi = pi
        self._frag_msms = self.backbone_spectrums()

        self._d, self._df, self._test_d, self._test_df = self.split_dataset()
        self.assign_train_score(self._d[score_init])
        self.assign_test_score(self._test_d[score_init])

    def shuffle(self):
        # total_ids = np.array(self._data['SpecId'])
        # random_order = np.random.shuffle(np.arange(len(total_ids)))
        # self._random_id_matches = {
        #     i: j for i, j in zip(total_ids, total_ids[random_order])
        # }
        pass

    def reverse(self):
        print(f"[Dataset reverse]: {len(self._d)} -> {len(self._test_d)}")
        self._d, self._test_d = self._test_d, self._d
        self._df, self._test_df = self._test_df, self._df
        return self

    def backbone_spectrums(self):
        sp = self._data.apply(
            lambda x: reverse_annotation(
                x['peak_ions'], x['peak_inten'], x['Charge'], x['sequence_length']).reshape(1, -1),
            axis=1
        )
        return np.concatenate(sp, axis=0)

    def num_of_scalars_feat(self):
        pass

    def assign_train_score(self, scores):
        assert len(scores) == len(self._d)
        self._scores = np.array(scores)

    def assign_test_score(self, scores):
        assert len(scores) == (len(self._test_d))
        self._test_scores = np.array(scores)

    def split_dataset(self):
        targets = self._data[self._data['Label'] == 1]
        targets_frag = self._frag_msms[self._data['Label'] == 1]
        decoys = self._data[self._data['Label'] == -1]
        decoys_frag = self._frag_msms[self._data['Label'] == -1]
        # print(f"Total {len(targets)} targets, {len(decoys)} decoys from {self._file_input}")

        train_data = targets.append(decoys[:len(decoys)//2])
        train_frag = np.concatenate(
            (targets_frag, decoys_frag[:len(decoys)//2]), axis=0)
        test_data = targets.append(decoys[len(decoys)//2:])
        test_decoy_frag = np.concatenate(
            (targets_frag, decoys_frag[len(decoys)//2:]), axis=0)
        # print(f"    {len(train_data)} PSMs will be used for training")
        return train_data, train_frag, test_data, test_decoy_frag

    def id2remove(self):
        return np.array(self._d[self._d['Label'] == -1]['SpecId'])

    def q_compute(self, scores, table, pi):
        ratio = (table['Label'] == 1).sum()/(table['Label'] == -1).sum()
        ratio = pi*ratio
        indexs = np.arange(len(scores))
        labels = np.array(table['Label'])
        orders = np.argsort(scores)

        indexs = indexs[orders]
        labels = labels[orders]

        target_sum = np.flip(np.cumsum(np.flip(labels == 1)))
        decoy_sum = np.flip(np.cumsum(np.flip(labels == -1)))

        target_sum[:-1] = target_sum[1:]
        decoy_sum[:-1] = decoy_sum[1:]

        fdrs = ratio*decoy_sum/(target_sum + 1e-9)
        fdrs[-1] = 0
        q_values = np.zeros_like(fdrs)
        min_fdrs = np.inf
        for i, fdr in enumerate(fdrs):
            min_fdrs = min(min_fdrs, fdr)
            q_values[i] = min_fdrs

        remap = np.argsort(indexs)
        q_values = q_values[remap]
        return q_values

    def Q_values(self):
        q_values = self.q_compute(self._scores, self._d, self._pi)
        return q_values

    def Q_values_test(self):
        q_values = self.q_compute(self._test_scores, self._test_d, self._pi)
        return q_values

    def prepare_sa_data(self, table, frag_msms):
        xlabel = ["sequence_integer",
                  "precursor_charge_onehot",
                  "collision_energy_aligned_normed"]
        ylabel = "intensities_raw"
        names = xlabel + [ylabel, "label"]

        y_data = torch.from_numpy(frag_msms)
        seq_data = list(table.apply(
            lambda x: peptide_to_inter(x['Peptide']), axis=1))
        seq_data = torch.from_numpy(np.concatenate(seq_data))

        charges = list(table.apply(lambda x: one_hot(x['Charge'] - 1), axis=1))
        charges = torch.from_numpy(np.concatenate(charges))

        nces = np.array(table['collision_energy_aligned_normed'])
        nces = torch.from_numpy(nces).unsqueeze(1)

        labels = np.array(table['Label'])
        labels = torch.from_numpy(labels)

        data_sa = [seq_data, charges, nces, y_data, labels]
        return names, data_sa

    def prepare_rt_data(self, table):
        names = ['sequence_integer', "irt", "label"]

        seq_data = list(table.apply(
            lambda x: peptide_to_inter(x['Peptide']), axis=1))
        seq_data = torch.from_numpy(np.concatenate(seq_data))

        rt = np.array(table['retention_time'])
        self._rt_mean, self._rt_std = np.mean(rt), np.std(rt)
        rt = (rt - self._rt_mean)/self._rt_std
        rt = torch.from_numpy(rt)

        labels = np.array(table['Label'])
        labels = torch.from_numpy(labels)

        data_rt = [seq_data, rt, labels]
        return names, data_rt

    def prepare_data(self, table, frag_msms):
        xlabel = ["sequence_integer",
                  "precursor_charge_onehot",
                  "collision_energy_aligned_normed"]
        names = xlabel+["intensities_raw", "irt", "label"]

        y_data = torch.from_numpy(frag_msms)
        seq_data = list(table.apply(
            lambda x: peptide_to_inter(x['Peptide']), axis=1))
        seq_data = torch.from_numpy(np.concatenate(seq_data))

        charges = list(table.apply(lambda x: one_hot(x['Charge'] - 1), axis=1))
        charges = torch.from_numpy(np.concatenate(charges))

        nces = np.array(table['collision_energy_aligned_normed'])
        nces = torch.from_numpy(nces).unsqueeze(1)

        rt = np.array(table['retention_time'])
        self._rt_mean, self._rt_std = np.mean(rt), np.std(rt)
        rt = (rt - self._rt_mean)/self._rt_std
        rt = torch.from_numpy(rt).unsqueeze(1)

        labels = np.array(table['Label'])
        labels = torch.from_numpy(labels)
        data = [seq_data, charges, nces, y_data, rt, labels]
        return names, data

    def semisupervised_sa_finetune(self, threshold=0.1):
        q_values = self.q_compute(self._scores, self._d, self._pi)
        sat_d = self._d[q_values <= threshold]
        sat_f = self._df[q_values <= threshold]
        names, data_sa = self.prepare_sa_data(sat_d, sat_f)
        return FinetuneTableDataset(names, data_sa)

    def semisupervised_sa_finetune_noneg(self, threshold=0.1):
        q_values = self.q_compute(self._scores, self._d, self._pi)
        target_q_values = q_values[self._d['Label'] == 1]
        sat_d = self._d[self._d['Label'] == 1][target_q_values <= threshold]
        sat_f = self._df[self._d['Label'] == 1][target_q_values <= threshold]
        print(len(sat_d))
        names, data_sa = self.prepare_sa_data(sat_d, sat_f)
        return FinetuneTableDataset(names, data_sa)

    def semisupervised_sa_finetune_double_stanard(self, target_thres=0.01, decoy_thres=0.1):
        q_values = self.q_compute(self._scores, self._d, self._pi)
        target_q_values = q_values[self._d['Label'] == 1]
        decoy_q_values = q_values[self._d['Label'] == -1]
        sat_d_target = self._d[self._d['Label']
                               == 1][target_q_values <= target_thres]
        sat_d_decoy = self._d[self._d['Label']
                              == -1][decoy_q_values <= decoy_thres]
        sat_f_target = self._df[self._d['Label']
                                == 1][target_q_values <= target_thres]
        sat_f_decoy = self._df[self._d['Label']
                               == -1][decoy_q_values <= decoy_thres]

        sat_d = pd.concat([sat_d_target, sat_d_decoy], ignore_index=True)
        sat_f = np.concatenate([sat_f_target, sat_f_decoy], axis=0)
        print(
            f"Target {len(sat_d[sat_d['Label'] == 1])}, Decoy {len(sat_d[sat_d['Label'] == -1])}")
        names, data_sa = self.prepare_sa_data(sat_d, sat_f)
        return FinetuneTableDataset(names, data_sa)

    def semisupervised_sa_finetune_allDecoy(self, threshold=0.1):
        q_values = self.q_compute(self._scores, self._d, self._pi)
        target_q_values = q_values[self._d['Label'] == 1]
        sat_d_target = self._d[self._d['Label']
                               == 1][target_q_values <= threshold]
        sat_d_target = sat_d_target.reset_index(drop=True)
        sat_d_decoy = self._d[self._d['Label'] == -1]
        sat_d_decoy = sat_d_decoy.reset_index(drop=True)

        sat_f_target = self._df[self._d['Label']
                                == 1][target_q_values <= threshold]
        sat_f_decoy = self._df[self._d['Label'] == -1]

        top_decoy_order = np.argsort(
            q_values[self._d['Label'] == -1])[:len(sat_f_target)]
        sat_d_decoy = sat_d_decoy.loc[list(top_decoy_order)]
        sat_f_decoy = sat_f_decoy[top_decoy_order]

        sat_d = pd.concat([sat_d_target, sat_d_decoy], ignore_index=True)
        sat_f = np.concatenate([sat_f_target, sat_f_decoy], axis=0)
        print(f"Target {len(sat_d_target)}, Decoy {len(sat_d_decoy)}")
        names, data_sa = self.prepare_sa_data(sat_d, sat_f)
        return FinetuneTableDataset(names, data_sa)

    def semisupervised_RT_finetune(self, threshold=0.1):
        q_values = self.q_compute(self._scores, self._d, self._pi)
        sat_d = self._d[q_values <= threshold]
        names, data_sa = self.prepare_rt_data(sat_d)
        return FinetuneTableDataset(names, data_sa)

    def semisupervised_finetune(self, threshold=0.1):
        q_values = self.q_compute(self._scores, self._d, self._pi)
        sat_d = self._d[q_values <= threshold]
        sat_f = self._df[q_values <= threshold]
        names, data = self.prepare_data(sat_d, sat_f)
        return FinetuneTableDataset(names, data)

    def semisupervised_pair_sa_finetune(self, threshold=0.1):
        q_values = self.q_compute(self._scores, self._d, self._pi)
        sat_d = self._d[q_values <= threshold]
        sat_f = self._df[q_values <= threshold]

        pos_d = sat_d[sat_d['Label'] == 1]
        pos_df = sat_f[sat_d['Label'] == 1]

        neg_d = self._d[self._d['Label'] == -1]
        neg_df = self._df[self._d['Label'] == -1]

        names, pos_data_sa = self.prepare_sa_data(pos_d, pos_df)
        names, neg_data_sa = self.prepare_sa_data(neg_d, neg_df)

        return PairFinetuneTableDataset(names, pos_data_sa, neg_data_sa)

    def supervised_sa_finetune(self):
        names, data_sa = self.prepare_data(self._d, self._df)
        return FinetuneTableDataset(names, data_sa)

    def train_all_data(self):
        return self.supervised_sa_finetune()

    def test_all_data(self):
        names, data_sa = self.prepare_data(self._test_d, self._test_df)
        return FinetuneTableDataset(names, data_sa)


class FinetuneTableDataset(Dataset):
    def __init__(self, names, xs):
        self.x = xs
        self.names = names

    def __len__(self):
        return len(self.x[0])

    def __getitem__(self, index):
        re = [i[index] for i in self.x]
        return {self.names[i]: re[i] for i in range(len(re))}


class PairFinetuneTableDataset(Dataset):
    def __init__(self, names, xs, nxs):
        self.x = xs
        self.nx = nxs
        self.names = names

    def __len__(self):
        return len(self.x[0])

    def __getitem__(self, index):
        re = [i[index] for i in self.x]
        neg_index = torch.randint(len(self.nx[0]), (1, ))
        neg_re = [i[neg_index] for i in self.nx]
        pos_data = {self.names[i]: re[i] for i in range(len(re))}
        neg_data = {
            self.names[i]+"_neg": neg_re[i].squeeze(0) for i in range(len(neg_re))}
        pos_data.update(neg_data)
        return pos_data


if __name__ == "__main__":
    data_file = "/data/prosit/figs/figure6/sprot_human/percolator/try/prosit_l1/fixed_features.tab"
    data = SemiDataset(data_file)
    qs = data.q_compute(data._scores, data._d, data._pi)
    iter_data = data.semisupervised_pair_sa_finetune()
    print({k: v.shape for k, v in iter_data[0].items()})

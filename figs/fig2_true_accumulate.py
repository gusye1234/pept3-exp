import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


def plot_cum_fdr_two(targets, decoys, targets_f, decoys_f, names=['Target', "Decoy"], which='', frag_model='', title=''):
    import matplotlib as mpl

    plt.style.use(['ieee', "high-vis", 'no-latex'])
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams['svg.fonttype'] = 'none'

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    target_c = ['blue', 'green']
    for i in range(2):
        if i == 0:
            t_value = np.abs(targets)
            d_value = np.abs(decoys)
        else:
            t_value = np.abs(targets_f)
            d_value = np.abs(decoys_f)

        t_v, t_base = np.histogram(t_value, bins=100)
        d_v, d_base = np.histogram(d_value, bins=100)

        # t_cum_sum = np.cumsum(t_v[::-1])[::-1] / len(t_value)
        t_cum_sum = t_v
        # d_cum_sum = np.cumsum(d_v[::-1])[::-1] / len(d_value)
        d_cum_sum = d_v

        surfix = "" if i == 0 else " with PepT3"
        ax.stairs(t_v, t_base, label=names[0] + surfix, linestyle='-')
        ax.stairs(d_v, d_base, label=names[1] + surfix, linestyle="--")
        # ax.plot(t_base[:-1], t_cum_sum, label=prefix +
        #         names[0], linestyle='-')
        # ax.plot(d_base[:-1], d_cum_sum, label=prefix +
        #         names[1], linestyle="--")
    ax.set_xlabel("Spectral Angle", fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper center", frameon=False)
    ax.set_xlim((0, 1))
    ax.invert_xaxis()
    fig.savefig(
        f"fig/fig2-cumulative-curve-{frag_model}-{which}_true.svg", dpi=300, bbox_inches="tight")
    mpl.rcParams.update(mpl.rcParamsDefault)


import sys
sys.path.append("..")
from tools import *
import tools
from fdr_test import fixed_features
from pept3 import model, finetune
import pept3
from pept3.dataset import SemiDataset
from torch.utils.data import DataLoader
frag_model = "prosit_l1"

shownames = ["Trypsin", "Chymo", "Lys-C", 'Glu-C']
for which, show in zip(["trypsin", 'chymo', 'lysc', 'gluc'], shownames):
    print(which)
    result_dir = f"/data2/yejb/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}"
    no_finetuned_dir = os.path.join(result_dir, "no_finetuned_3fold")
    sa_feat = pd.read_csv(os.path.join(
        no_finetuned_dir, "sa.tab"), sep='\t')
    finetuned_dir = os.path.join(result_dir, "finetuned_3fold_0.1")
    finetune_sa_feat = pd.read_csv(os.path.join(
        finetuned_dir, "sa.tab"), sep='\t')

    plot_cum_fdr_two(sa_feat[sa_feat['Label'] == 1]['spectral_angle'],
                     sa_feat[sa_feat['Label'] == -1]['spectral_angle'],
                     finetune_sa_feat[finetune_sa_feat['Label']
                                      == 1]['spectral_angle'],
                     finetune_sa_feat[finetune_sa_feat['Label']
                                      == -1]['spectral_angle'],
                     which=which, frag_model=frag_model, title=show)

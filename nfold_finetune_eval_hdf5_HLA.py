import sys
from contextlib import redirect_stdout
import pandas as pd
import torch
import os
from time import time
from pept3 import helper
from pept3 import model
from pept3 import finetune
import h5py
from tqdm import tqdm
import numpy as np
from pept3.tools import get_sa_from_array
from pept3.fdr_eval import eval_fdr_hdf5 as eval_fdr

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

run_model = model.PrositIRT()
run_model.load_state_dict(torch.load(
    f"./checkpoints/irt/best_valid_irt_{run_model.comment()}-1024.pth", map_location="cpu"))
prosit_irt = run_model.eval()


frag_model = "prosit_l1"
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

from datetime import datetime
now = datetime.now().strftime("%y%m%d%H")
with open(f'logs/2020-hla-{frag_model}-{now}.log', 'w') as sys.stdout:
# if __name__ == "__main__":

    sample_size = None
    gpu_index = 0
    set_threshold = 0.1
    max_epochs = 20
    print("Running twofold", frag_model)
    if_pearson = (frag_model in ['pdeep2'])

    # ----------------------------------------------------
    hla_mel = pd.read_csv("./figs/data/HLA_Mel.csv")
    hla_mel = hla_mel[hla_mel['Experiment'].apply(
        lambda x: x.endswith("HLA-I"))]
    Mels = hla_mel['Experiment'].unique()
    for which in Mels:
    # for which in ['Mel-15_HLA-I']:
        print("-------------------------------")
        print("boosting figure3", which)
        save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/{frag_model}"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/{frag_model}/3fold_Mels_{set_threshold}/"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/{frag_model}/3fold_Mels_{set_threshold}/{which}"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        save_tab2 = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/{frag_model}/3fold_Mels_{set_threshold}/{which}_ori"
        if not os.path.exists(save_tab2):
            os.mkdir(save_tab2)
        feature_csv = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/forPride/rescoring_for_paper_2/Mels/{which}/percolator/features.csv"
        origin_prosit_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/forPride/rescoring_for_paper_2/Mels/{which}/percolator/prosit.tab"
        tabels_file = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/forPride/rescoring_for_paper_2/Mels/{which}/data.hdf5"
        models, id2selects = finetune.semisupervised_finetune_nfold(
            run_model, tabels_file, max_epochs=max_epochs, pearson=if_pearson, gpu_index=gpu_index, only_id2select=False, q_threshold=set_threshold)
        
        ori_models = [run_model for _ in models]
        print(eval_fdr(ori_models, tabels_file, feature_csv, origin_prosit_tab, save_tab2,
                       irt_model=prosit_irt, sample_size=sample_size, id2selects=id2selects, pearson=if_pearson, gpu_index=gpu_index).to_string())
        print(eval_fdr(models, tabels_file, feature_csv, origin_prosit_tab, save_tab,
                       irt_model=prosit_irt, sample_size=sample_size, id2selects=id2selects, pearson=if_pearson, gpu_index=gpu_index).to_string())

    # # ------------------------------------------
    # set_threshold = 0.1
    # hla_mel = pd.read_csv("./figs/data/HLA_Mel.csv")
    # hla_mel = hla_mel[hla_mel['Experiment'].apply(
    #     lambda x: x.endswith("HLA-II"))]
    # Mels = hla_mel['Experiment'].unique()
    # for which in Mels:
    #     print("-------------------------------")
    #     print("boosting HLA II", which)
    #     save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_2/{frag_model}"
    #     if not os.path.exists(save_tab):
    #         os.mkdir(save_tab)
    #     save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_2/{frag_model}/percolator_hdf5_Mels_{set_threshold}/"
    #     if not os.path.exists(save_tab):
    #         os.mkdir(save_tab)
    #     save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_2/{frag_model}/percolator_hdf5_Mels_{set_threshold}/{which}"
    #     if not os.path.exists(save_tab):
    #         os.mkdir(save_tab)
    #     feature_csv = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_2/forPride/rescoring_for_paper_2/Mels/{which}/percolator/features.csv"
    #     origin_prosit_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_2/forPride/rescoring_for_paper_2/Mels/{which}/percolator/prosit.tab"
    #     tabels_file = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_2/forPride/rescoring_for_paper_2/Mels/{which}/data.hdf5"
    #     finetune_model1, finetune_model2, id2remove = finetune.semisupervised_finetune_twofold(
    #         run_model, tabels_file, max_epochs=max_epochs, pearson=if_pearson, gpu_index=gpu_index, only_id2remove=False, q_threshold=set_threshold)
    #     print(eval_fdr(finetune_model1, finetune_model2, tabels_file, feature_csv, origin_prosit_tab, save_tab,
    #                    irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove, pearson=if_pearson, gpu_index=gpu_index).to_string())

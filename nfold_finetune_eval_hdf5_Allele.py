import sys
sys.path.append("./figs")
from contextlib import redirect_stdout
import pandas as pd
import torch
import os
from time import time
from pept3 import helper
from pept3 import model
from pept3 import finetune
from pept3.fdr_eval import eval_fdr_hdf5 as eval_fdr
import h5py
from tqdm import tqdm
import numpy as np
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
    
set_threshold = 0.1
from datetime import datetime
now = datetime.now().strftime("%y%m%d%H")
with open(f'logs/2020-allele-{frag_model}-{set_threshold}-{now}.log', 'w') as sys.stdout:
# if __name__ == "__main__":
    sample_size = None
    gpu_index = 0
    max_epochs = 20
    print("Running twofold", frag_model)
    if_pearson = (frag_model in ['pdeep2'])
    alleles_rawfile = {}
    with open("figs/data/allele_raw.txt") as f:
        for l in f:
            pack = l.strip().split("\t")
            alleles_rawfile[pack[0]] = set(pack[1:])
    Alleles = sorted(alleles_rawfile.keys())
    for which in Alleles:
        print("-------------------------------")
        print("boosting figure3", which)
        save_tab = f"/data1/yejb/prosit/figure3/{frag_model}/"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        save_tab = f"/data1/yejb/prosit/figure3/{frag_model}/3fold_hdf5_allele_{set_threshold}/"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        save_tab = f"/data1/yejb/prosit/figure3/{frag_model}/3fold_hdf5_allele_{set_threshold}/{which}"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        save_tab2 = f"/data1/yejb/prosit/figure3/{frag_model}/3fold_hdf5_allele_{set_threshold}/{which}_ori"
        if not os.path.exists(save_tab2):
            os.mkdir(save_tab2)
        feature_csv = f"/data2/yejb/prosit/figs/alleles/forPRIDE/Alleles/{which}/percolator/features.csv"
        origin_prosit_tab = f"/data2/yejb/prosit/figs/alleles/forPRIDE/Alleles/{which}/percolator/prosit.tab"
        tabels_file = f"/data2/yejb/prosit/figs/alleles/forPRIDE/Alleles/{which}/data.hdf5"
        models, id2selects = finetune.semisupervised_finetune_nfold(
            run_model, tabels_file, max_epochs=max_epochs, pearson=if_pearson, gpu_index=gpu_index, only_id2select=False, q_threshold=set_threshold)
        # torch.save([finetune_model1.state_dict(), finetune_model2.state_dict()],
        #            os.path.join(model_saving, f"{frag_model}.pth"))
        ori_models = [run_model for _ in models]
        print(eval_fdr(ori_models, tabels_file, feature_csv, origin_prosit_tab, save_tab2,
                       irt_model=prosit_irt, sample_size=sample_size, id2selects=id2selects, pearson=if_pearson, gpu_index=gpu_index).to_string())
        print(eval_fdr(models, tabels_file, feature_csv, origin_prosit_tab, save_tab,
                       irt_model=prosit_irt, sample_size=sample_size, id2selects=id2selects, pearson=if_pearson, gpu_index=gpu_index).to_string())

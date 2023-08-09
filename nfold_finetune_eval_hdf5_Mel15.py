from copy import deepcopy
from pept3.helper import fixed_features
from pept3.tools import get_sa_from_array
import numpy as np
from tqdm import tqdm
import h5py
from pept3 import finetune
from pept3 import model
from pept3 import helper
from pept3.fdr_eval import eval_fdr_hdf5 as eval_fdr
from time import time
import os
import torch
import pandas as pd
from contextlib import redirect_stdout
import sys



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
set_threshold = 0.01
    
from datetime import datetime
now = datetime.now().strftime("%y%m%d%H")
with open(f'logs/2020-mel15-{set_threshold}-{now}.log', 'w') as sys.stdout:
    
    sample_size = None
    gpu_index = 0
    print("Running twofold", frag_model)
    if_pearson = (frag_model in ['pdeep2'])
    for which in ['Mel15']:
        print("-------------------------------")
        print("boosting figure3", which)
        # save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/percolator_hdf5/"
        save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/3fold_hdf5_{set_threshold}/"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        feature_csv = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/forPride/rescoring_for_paper_2/percolator/features.csv"
        origin_prosit_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/forPride/rescoring_for_paper_2/percolator/prosit.tab"
        tabels_file = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/forPride/rescoring_for_paper_2/data.hdf5"
        model_saving_path = f"./checkpoints/finetuned/{which}"
        if not os.path.exists(model_saving_path):
            os.mkdir(model_saving_path)
        models, id2select = finetune.semisupervised_finetune_nfold(
            run_model, tabels_file, pearson=if_pearson, gpu_index=gpu_index, only_id2select=False, q_threshold=set_threshold)

        for i, m in enumerate(models):
            torch.save(m.state_dict(),
                       os.path.join(model_saving_path, f"{frag_model}_part_{i}.pth"))
        # for i, m  in enumerate(models):
        #     m.load_state_dict(torch.load(os.path.join(model_saving_path, f"{frag_model}_part_{i}.pth"), map_location="cpu"))
        
        print(eval_fdr(models, tabels_file, feature_csv, origin_prosit_tab, save_tab,
                       irt_model=prosit_irt, sample_size=sample_size, id2selects=id2select, pearson=if_pearson, gpu_index=gpu_index).to_string())

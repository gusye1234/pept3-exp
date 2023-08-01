
from datetime import datetime
import sys

import pandas as pd
import torch
import os
from time import time
from pept3 import helper
from pept3 import model
from pept3 import finetune
from pept3.helper import fixed_features, fdr_test_nfold
from round1_utils import merge_decoy

def overlap_analysis(tab1, tab2, testfdr=0.01, compare=["prosit_combined", "prosit_best"]):
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


def overlap_analysis_peptides(tab1, tab2, testfdr=0.01, compare=["prosit_combined", "prosit_best"]):
    baseline = "sa"
    table1 = pd.read_csv(os.path.join(
        tab1, f"{compare[0]}_target.psms"), sep='\t')
    table2 = pd.read_csv(os.path.join(
        tab2, f"{compare[1]}_target.psms"), sep='\t')

    id1 = set(table1[table1['q-value'] < testfdr]
              ['peptide'].apply(lambda x: x.strip('_').strip(".")))
    id2 = set(table2[table2['q-value'] < testfdr]
              ['peptide'].apply(lambda x: x.strip('_').strip(".")))
    overlap = id1.intersection(id2)
    union = id1.union(id2)
    print(f"Peptides {compare}-{testfdr}:", (len(id1) - len(overlap)) / len(union),
          len(overlap) / len(union), (len(id2) - len(overlap)) / len(union))
    return len(id1) - len(overlap), len(overlap), len(id2) - len(overlap)


def eval_fdr(models, msms_file, raw_dir, save_tab, 
             fdr_threshold=0.1, show_fdr=[0.1, 0.01, 0.001, 0.0001], 
             sample_size=None, need_all=False, irt_model=None, 
             id2selects=None, pearson=False):

    record = {}
    record['fdrs'] = [100 * i for i in show_fdr]
    totest = ["andromeda", "sa", "prosit_combined"]
    if irt_model is not None:
        totest.append("prosit_best")

    with torch.no_grad():
        models = [m.eval() for m in models]
        fdr_test_nfold(models, msms_file, raw_dir, save_tab, id2selects,
                       sample_size=sample_size, need_all=need_all, irt_model=irt_model, totest=totest, pearson=pearson)
        print("Saving feature tab to", save_tab)

    print(" start percolator... ")
    for name in totest:
        start = time()
        os.system(f"percolator -v 0 --weights {save_tab}/{name}_weights.csv \
                --post-processing-tdc --only-psms --testFDR {fdr_threshold} \
                --results-psms {save_tab}/{name}_target.psms \
                --decoy-results-psms {save_tab}/{name}_decoy.psms \
                {save_tab}/{name}.tab")
        target_tab = pd.read_csv(os.path.join(
            save_tab, f"{name}_target.psms"), sep='\t')
        record[name] = []
        for fdr in show_fdr:
            record[name].append((target_tab['q-value'] < fdr).sum())
        print(f"{name}:{time()-start:.1f}", end='-')
    print()
    return pd.DataFrame(record)


ori_which = "trypsin"
from_which = "chymo"
merge_which = "trypsin_chymo"
merge_tab = f"/data2/yejb/prosit/figs/fig235/round1/{merge_which}/"
if not os.path.exists(merge_tab):
    os.mkdir(merge_tab)
# merge_decoy(
#     f"/data2/yejb/prosit/figs/fig235/{ori_which}/maxquant/combined/txt/",
#     f"/data2/yejb/prosit/figs/fig235/{ori_which}/percolator_up/try/prosit_l1",
#     f"/data2/yejb/prosit/figs/fig235/{from_which}/maxquant/combined/txt/",
#     f"/data2/yejb/prosit/figs/fig235/{from_which}/percolator_up/try/prosit_l1",
#     merge_tab,
#     sample_num=5000
# )


run_model = model.PrositIRT()
run_model.load_state_dict(torch.load(
    f"./checkpoints/irt/best_valid_irt_{run_model.comment()}-1024.pth", map_location="cpu"))
prosit_irt = run_model.eval()

frag_model = "prosit_l1"
# frag_model = "pdeep2"
if frag_model == "prosit_l1":
    run_model = model.PrositFrag()
    run_model.load_state_dict(torch.load(
        "./checkpoints/best/best_frag_l1_PrositFrag-1024.pth", map_location="cpu"))
    run_model = run_model.eval()
elif frag_model == "pdeep2":
    run_model = model.pDeep2_nomod()
    run_model.load_state_dict(torch.load(
        "./checkpoints/best/best_frag_l1_pDeep2-1024.pth", map_location="cpu"))
    run_model = run_model.eval()

q_threshold = 0.1

now = datetime.now().strftime("%y%m%d%H")
# with open(f'logs/2019-round1-merge-{q_threshold}-{frag_model}-{now}.log', 'w') as sys.stdout:

if __name__ == "__main__":
    sample_size = None
    print("Running twofold", frag_model)
    if_pearson = (frag_model in ['pdeep2'])
    analysis_dict = {}
    
    print("-------------------------------")
    print(merge_which)
    save_tab = f"/data2/yejb/prosit/figs/fig235/round1/{merge_which}/percolator/"
    if not os.path.exists(save_tab):
        os.mkdir(save_tab)
    msms_file = f"/data2/yejb/prosit/figs/fig235/round1/{merge_which}/msms.txt"
    raw_dir = f"/data2/yejb/prosit/figs/fig235/{ori_which}/raw"

    tabels_file = fixed_features(msms_file, raw_dir,
                                    f"/data2/yejb/prosit/figs/fig235/round1/{merge_which}")
    save_tab1 = f"/data2/yejb/prosit/figs/fig235/round1/{merge_which}/percolator/before"
    save_tab2 = f"/data2/yejb/prosit/figs/fig235/round1/{merge_which}/percolator/after"
    if not os.path.exists(save_tab1):
        os.mkdir(save_tab1)
    if not os.path.exists(save_tab2):
        os.mkdir(save_tab2)
    models, id2selects = finetune.semisupervised_finetune_nfold(
        run_model, tabels_file, pearson=if_pearson, only_id2select=False, q_threshold=q_threshold, gpu_index=7)
    ori_models = [run_model for _ in models]
    print(eval_fdr(ori_models, msms_file, raw_dir, save_tab1,
          irt_model=prosit_irt, sample_size=sample_size, id2selects=id2selects, pearson=if_pearson).to_string())
    print(eval_fdr(models, msms_file, raw_dir, save_tab2,
            irt_model=prosit_irt, sample_size=sample_size, id2selects=id2selects, pearson=if_pearson).to_string())
        
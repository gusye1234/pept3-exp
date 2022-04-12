import sys

# from figs.fdr_test import fdr_test_reverse
sys.path.append("./figs")

from ms import finetune
from ms import model
from ms import helper
from time import time
from fdr_test import fdr_test, fixed_features
import os
import torch
import pandas as pd
from contextlib import redirect_stdout


def overlap_analysis(tab1, tab2, testfdr=[0.01], compare=["prosit_combined", "prosit_best"]):
    baseline = "sa"
    table1 = pd.read_csv(os.path.join(tab1, f"{compare[0]}_target.psms"), sep='\t')
    table2 = pd.read_csv(os.path.join(
        tab2, f"{compare[1]}_target.psms"), sep='\t')
    
    for fdr in testfdr:
        id1 = set(table1[table1['q-value'] < fdr]['PSMId'])
        id2 = set(table2[table2['q-value'] < fdr]['PSMId'])
        overlap = id1.intersection(id2)
        union = id1.union(id2)
        print(f"{fdr}:", (len(id1) - len(overlap))/len(union), len(overlap)/len(union), (len(id2)-len(overlap))/len(union))

def eval_fdr(run_model, msms_file, raw_dir, save_tab, fdr_threshold=0.1, show_fdr=[0.1, 0.01, 0.001, 0.0001], sample_size=None, need_all=False, irt_model=None, id2remove=None, gpu_index=0):
    run_model = run_model.eval()
    record = {}
    record['fdrs'] = [100*i for i in show_fdr]
    totest = ["andromeda", "sa", "prosit_combined"]
    if irt_model is not None:
        totest.append("prosit_best")

    with torch.no_grad():
        fdr_test(run_model, msms_file, raw_dir, save_tab,
                 sample_size=sample_size, need_all=need_all, irt_model=irt_model, id2remove=id2remove, totest=totest, gpu_index=gpu_index)

    # totest = ["andromeda", "sa", "combined",
    #           "prosit", "prosit_combined", "prosit_ratio"]
    print(" start percolator... ", end='')
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


if __name__ == "__main__":
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

    sample_size = None
    gpu_index = 4
    print("Running", frag_model)
    for which in ["Mel1", "HLA_2", "HLA_1"]:
        print("-------------------------------")
        print(which)
        save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/percolator/{frag_model}_finetune"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        msms_file = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/forPride/txt/msms.txt"
        raw_dir =   f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/forPride/rescoring_for_paper_2/raw"
        tabels_file = fixed_features(
            msms_file, raw_dir, f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/fix_features/")
        finetune_model, id2remove = finetune.semisupervised_finetune(run_model, tabels_file, gpu_index=gpu_index)

        save_tab1 = os.path.join(save_tab, "no_finetuned")
        if not os.path.exists(save_tab1):
            os.mkdir(save_tab1)
        print(eval_fdr(run_model, msms_file, raw_dir, save_tab1, irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove, gpu_index=gpu_index).to_string())
        save_tab2 = os.path.join(save_tab, "finetuned")
        if not os.path.exists(save_tab2):
            os.mkdir(save_tab2)
        print(eval_fdr(finetune_model, msms_file, raw_dir, save_tab2, irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove, gpu_index=gpu_index).to_string())
        overlap_analysis(save_tab1, save_tab2)
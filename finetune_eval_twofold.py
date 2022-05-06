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
from fdr_test import fdr_test, fixed_features, fdr_test_twofold


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


def eval_fdr(run_model1, run_model2, msms_file, raw_dir, save_tab, fdr_threshold=0.01, show_fdr=[0.1, 0.01, 0.001, 0.0001], sample_size=None, need_all=False, irt_model=None, id2remove=None, pearson=False, gpu_index=0):
    run_model1 = run_model1.eval()
    run_model2 = run_model2.eval()

    record = {}
    record['fdrs'] = [100 * i for i in show_fdr]
    totest = ["andromeda", "sa", "prosit_combined"]
    if irt_model is not None:
        totest.append("prosit_best")

    with torch.no_grad():
        fdr_test_twofold(run_model1, run_model2, msms_file, raw_dir, save_tab,
                         sample_size=sample_size, need_all=need_all, irt_model=irt_model, id2remove=id2remove, totest=totest, pearson=pearson, gpu_index=gpu_index)

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


def combined_eval_fdr(no_finetuned_dir, finetuned_dir, fdr_threshold=0.1, show_fdr=[0.1, 0.01, 0.001, 0.0001]):
    father_dir = os.path.dirname(finetuned_dir)
    combined_dir = os.path.join(father_dir, "combined")

    if not os.path.exists(combined_dir):
        os.mkdir(combined_dir)
    totest = ['prosit_combined', "prosit_best"]
    record = {}
    record['fdrs'] = [100 * i for i in show_fdr]
    print("Re-evaluate no finetuned")
    for name in totest:
        os.system(f"percolator -v 0 --weights {combined_dir}/{name}_weights_nofinetuned.csv \
                --post-processing-tdc --only-psms --testFDR {fdr_threshold} \
                --results-psms {combined_dir}/{name}_target_nofinetuned.psms \
                --decoy-results-psms {combined_dir}/{name}_decoy_nofinetuned.psms \
                {no_finetuned_dir}/{name}.tab")
        target_tab = pd.read_csv(os.path.join(
            combined_dir, f"{name}_target_nofinetuned.psms"), sep='\t')
        record[name + "_no_finetuned"] = []
        for fdr in show_fdr:
            record[name +
                   "_no_finetuned"].append((target_tab['q-value'] < fdr).sum())
    print("Re-evaluate finetuned")
    for name in totest:
        os.system(f"percolator -v 0 --weights {combined_dir}/{name}_weights_finetuned.csv \
                --post-processing-tdc --only-psms --testFDR {fdr_threshold} \
                --results-psms {combined_dir}/{name}_target_finetuned.psms \
                --decoy-results-psms {combined_dir}/{name}_decoy_finetuned.psms \
                {finetuned_dir}/{name}.tab")
        target_tab = pd.read_csv(os.path.join(
            combined_dir, f"{name}_target_finetuned.psms"), sep='\t')
        record[name + "_finetuned"] = []
        for fdr in show_fdr:
            record[name +
                   "_finetuned"].append((target_tab['q-value'] < fdr).sum())
    for name in totest:
        no_tab = pd.read_csv(f"{no_finetuned_dir}/{name}.tab", sep='\t')
        fi_tab = pd.read_csv(f"{finetuned_dir}/{name}.tab", sep='\t')
        no_tab['spectral_angle'] = fi_tab['spectral_angle']
        no_tab['delta_sa'] = fi_tab['delta_sa']
        no_tab.to_csv(f"{combined_dir}/combined_{name}.tab",
                      sep='\t', index=False)
    for name in totest:
        os.system(f"percolator -v 0 --weights {combined_dir}/{name}_weights_combined.csv \
                --post-processing-tdc --only-psms --testFDR {fdr_threshold} \
                --results-psms {combined_dir}/{name}_target_combined.psms \
                --decoy-results-psms {combined_dir}/{name}_decoy_combined.psms \
                {combined_dir}/combined_{name}.tab")
        target_tab = pd.read_csv(os.path.join(
            combined_dir, f"{name}_target_combined.psms"), sep='\t')
        record[name + "_combined"] = []
        for fdr in show_fdr:
            record[name +
                   "_combined"].append((target_tab['q-value'] < fdr).sum())
    print(pd.DataFrame(record).to_string())
    return combined_dir


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
    gpu_index = 2
    print("Running twofold", frag_model)
    if_pearson = (frag_model in ['pdeep2'])
    analysis_dict = {}
    for which in ["HLA_2", "Mel1", "HLA_1"]:
        print("-------------------------------")
        print(which)
        save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/percolator/{frag_model}_finetune/finetuned_twofold"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        msms_file = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/forPride/txt/msms.txt"
        raw_dir = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/forPride/rescoring_for_paper_2/raw"

        tabels_file = fixed_features(msms_file, raw_dir,
                                     f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/fix_features/")
        finetune_model1, finetune_model2, id2remove = finetune.semisupervised_finetune_twofold(
            run_model, tabels_file, pearson=if_pearson, gpu_index=gpu_index, only_id2remove=False)
        print(eval_fdr(finetune_model1, finetune_model2, msms_file, raw_dir, save_tab,
              irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove, pearson=if_pearson, gpu_index=gpu_index).to_string())

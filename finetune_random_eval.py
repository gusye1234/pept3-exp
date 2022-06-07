import sys
from ms import finetune
from ms import model
from ms import helper
from time import time
import os
import torch
import pandas as pd
from contextlib import redirect_stdout
sys.path.append("./figs")
from fdr_test import fdr_test, fixed_features, fixed_features_random

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
    print(f"{compare}-{testfdr}:", (len(id1) - len(overlap))/len(union),
          len(overlap)/len(union), (len(id2)-len(overlap))/len(union))
    return len(id1) - len(overlap), len(overlap), len(id2)-len(overlap)


def eval_fdr(run_model, msms_file, raw_dir, save_tab, fdr_threshold=0.1, show_fdr=[0.1, 0.01, 0.001, 0.0001], sample_size=None, need_all=False, irt_model=None, id2remove=None, pearson=False):
    run_model = run_model.eval()
    record = {}
    record['fdrs'] = [100*i for i in show_fdr]
    totest = ["andromeda", "sa", "prosit_combined"]
    if irt_model is not None:
        totest.append("prosit_best")

    with torch.no_grad():
        fdr_test(run_model, msms_file, raw_dir, save_tab,
                 sample_size=sample_size, need_all=need_all, irt_model=irt_model, id2remove=id2remove, totest=totest, pearson=pearson)

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
    record['fdrs'] = [100*i for i in show_fdr]
    print("Re-evaluate no finetuned")
    for name in totest:
        os.system(f"percolator -v 0 --weights {combined_dir}/{name}_weights_nofinetuned.csv \
                --post-processing-tdc --only-psms --testFDR {fdr_threshold} \
                --results-psms {combined_dir}/{name}_target_nofinetuned.psms \
                --decoy-results-psms {combined_dir}/{name}_decoy_nofinetuned.psms \
                {no_finetuned_dir}/{name}.tab")
        target_tab = pd.read_csv(os.path.join(
            combined_dir, f"{name}_target_nofinetuned.psms"), sep='\t')
        record[name+"_no_finetuned"] = []
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
        record[name+"_finetuned"] = []
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
        record[name+"_combined"] = []
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
    if frag_model == "trans":
        run_model = model.TransProBest()
        run_model.load_state_dict(torch.load(
            "./checkpoints/best/best_valid_frag_TransPro-6-3-128-0.1-256-1048.pth", map_location="cpu"))
        run_model = run_model.eval()
    elif frag_model == "trans_l1":
        run_model = model.TransProBest()
        run_model.load_state_dict(torch.load(
            "/home/gus/Desktop/ms_pred/checkpoints/frag/best_frag_l1_TransProBest-6-3-128-0.1-256-1024.pth", map_location="cpu"))
        run_model = run_model.eval()
    elif frag_model == "prosit":
        run_model = model.PrositFrag()
        run_model.load_state_dict(torch.load(
            "./checkpoints/best/best_valid_irt_PrositFrag-1024.pth", map_location="cpu"))
        run_model = run_model.eval()
    elif frag_model == "prosit_l1":
        run_model = model.PrositFrag()
        run_model.load_state_dict(torch.load(
            "/home/gus/Desktop/ms_pred/checkpoints/best/best_frag_l1_PrositFrag-1024.pth", map_location="cpu"))
        run_model = run_model.eval()
    elif frag_model == "pdeep2":
        run_model = model.pDeep2_nomod()
        run_model.load_state_dict(torch.load(
            "/home/gus/Desktop/ms_pred/checkpoints/best/best_frag_l1_pDeep2-1024.pth", map_location="cpu"))
        run_model = run_model.eval()

    sample_size = None
    print("Running", frag_model)
    if_pearson = (frag_model in ['pdeep2'])
    analysis_dict = {}
    for which in ["sprot_human", "IGC", "sprot_all", "sprot_bacteria_human"]:
        print("-------------------------------")
        print(which)
        save_tab = f"/data/prosit/figs/figure6/{which}/percolator/try/{frag_model}"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        msms_file = f"/data/prosit/figs/figure6/{which}/maxquant/txt/msms.txt"
        raw_dir = f"/data/prosit/figs/figure6/all_raws"

        save_tab1 = f"/data/prosit/figs/figure6/{which}/percolator/try/{frag_model}/no_finetuned"
        if not os.path.exists(save_tab1):
            os.mkdir(save_tab1)
        save_tab2 = f"/data/prosit/figs/figure6/{which}/percolator/try/{frag_model}/finetuned"
        if not os.path.exists(save_tab2):
            os.mkdir(save_tab2)

        # combined_tab = combined_eval_fdr(save_tab1, save_tab2)
        # overlap_analysis(save_tab1, combined_tab, )

        tabels_file, random_matches = fixed_features_random(
            msms_file, raw_dir, f"/data/prosit/figs/figure6/{which}/percolator/try/prosit_l1")
        finetune_model, id2remove = finetune.semisupervised_finetune(
            run_model, tabels_file, pearson=if_pearson, enable_test=True)

        # print(eval_fdr(run_model, msms_file, raw_dir, save_tab1,
        #       irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove, pearson=if_pearson).to_string())

        # print(eval_fdr(finetune_model, msms_file, raw_dir, save_tab2,
        #       irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove, pearson=if_pearson).to_string())
        # analysis_dict[which] = overlap_analysis(save_tab1, save_tab2)

    for which in ["trypsin", 'chymo', "lysc", "gluc"]:
        print("-------------------------------")
        print(which)
        save_tab = f"/data/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        msms_file = f"/data/prosit/figs/fig235/{which}/maxquant/combined/txt/msms.txt"
        raw_dir = f"/data/prosit/figs/fig235/{which}/raw"

        save_tab1 = f"/data/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/no_finetuned"
        if not os.path.exists(save_tab1):
            os.mkdir(save_tab1)
        save_tab2 = f"/data/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/finetuned"
        if not os.path.exists(save_tab2):
            os.mkdir(save_tab2)

        tabels_file, random_matches = fixed_features_random(msms_file, raw_dir,
                                     f"/data/prosit/figs/fig235/{which}/percolator_up/try/prosit_l1")
        finetune_model, id2remove = finetune.semisupervised_finetune(
            run_model, tabels_file, pearson=if_pearson, enable_test=True)
        # print(eval_fdr(run_model, msms_file, raw_dir, save_tab1,
        #       irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove, pearson=if_pearson).to_string())

        # print(eval_fdr(finetune_model, msms_file, raw_dir, save_tab2,
        #       irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove, pearson=if_pearson).to_string())
        # analysis_dict[which] = overlap_analysis(save_tab1, save_tab2)

    print("-------------------------------")
    print("Davis")
    save_tab = f"/data/prosit/figs/figure5/percolator/try/{frag_model}"
    if not os.path.exists(save_tab):
        os.mkdir(save_tab)
    msms_file = f"/data/prosit/figs/figure5/maxquant/combined/txt/msms.txt"
    raw_dir = f"/data/prosit/figs/figure5/raw"

    save_tab1 = f"/data/prosit/figs/figure5/percolator/try/{frag_model}/no_finetuned"
    if not os.path.exists(save_tab1):
        os.mkdir(save_tab1)
    save_tab2 = f"/data/prosit/figs/figure5/percolator/try/{frag_model}/finetuned"
    if not os.path.exists(save_tab2):
        os.mkdir(save_tab2)

    tabels_file, random_matches = fixed_features_random(msms_file, raw_dir,
                                 f"/data/prosit/figs/figure5/percolator/try/prosit_l1")
    # finetune_model, id2remove = finetune.semisupervised_finetune(
    #     run_model, tabels_file)
    # print(eval_fdr(run_model, msms_file, raw_dir, save_tab1,
    #         irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove).to_string())

    # print(eval_fdr(finetune_model, msms_file, raw_dir, save_tab2,
    #         irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove).to_string())
    # analysis_dict["davis"] = overlap_analysis(save_tab1, save_tab2)
    # print(analysis_dict)

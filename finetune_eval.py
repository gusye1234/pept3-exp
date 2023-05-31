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


def overlap_analysis_peptides(tab1, tab2, testfdr=0.01, compare=["sa", "sa"]):
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


def eval_fdr(run_model1, run_model2, msms_file, raw_dir, save_tab, fdr_threshold=0.1, show_fdr=[0.1, 0.01, 0.001, 0.0001], sample_size=None, need_all=False, irt_model=None, id2remove=None, pearson=False):

    record = {}
    record['fdrs'] = [100 * i for i in show_fdr]
    totest = ["andromeda", "sa", "prosit_combined"]
    if irt_model is not None:
        totest.append("prosit_best")

    with torch.no_grad():
        run_model1 = run_model1.eval()
        run_model2 = run_model2.eval()
        fdr_test_twofold(run_model1, run_model2, msms_file, raw_dir, save_tab,
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


if __name__ == "__main__":
    run_model = model.PrositIRT()
    run_model.load_state_dict(torch.load(
        f"./checkpoints/irt/best_valid_irt_{run_model.comment()}-1024.pth", map_location="cpu"))
    prosit_irt = run_model.eval()

    frag_model = "pdeep2"
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
    print("Running twofold", frag_model)
    if_pearson = (frag_model in ['pdeep2'])
    analysis_dict = {}
    # ,
    for which in ["sprot_human", "IGC", "sprot_all", "sprot_bacteria_human"]:
        print("-------------------------------")
        print(which)
        save_tab = f"/data2/yejb/prosit/figs/figure6/{which}/percolator/try/{frag_model}"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        msms_file = f"/data2/yejb/prosit/figs/figure6/{which}/maxquant/txt/msms.txt"
        raw_dir = f"/data2/yejb/prosit/figs/figure6/all_raws"

        save_tab1 = f"/data2/yejb/prosit/figs/figure6/{which}/percolator/try/{frag_model}/no_finetuned_twofold"
        save_tab2 = f"/data2/yejb/prosit/figs/figure6/{which}/percolator/try/{frag_model}/finetuned_twofold"
        if not os.path.exists(save_tab2):
            os.mkdir(save_tab2)

        tabels_file = fixed_features(
            msms_file, raw_dir, f"/data2/yejb/prosit/figs/figure6/{which}/percolator/try/prosit_l1")
        finetune_model1, finetune_model2, id2remove = finetune.semisupervised_finetune_twofold(
            run_model, tabels_file, pearson=if_pearson, only_id2remove=False, onlypos=True, max_epochs=20)
        # print(eval_fdr(finetune_model1, finetune_model2, msms_file, raw_dir, save_tab2,
        #       irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove, pearson=if_pearson).to_string())
        print(eval_fdr(run_model, run_model, msms_file, raw_dir, save_tab1,
              irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove, pearson=if_pearson).to_string())
        exit()
        # analysis_dict[which] = overlap_analysis_peptides(save_tab1, save_tab2)
    # exit()
    for which in ["trypsin", 'chymo', "lysc", "gluc"]:
        print("-------------------------------")
        print(which)
        save_tab = f"/data2/yejb/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        msms_file = f"/data2/yejb/prosit/figs/fig235/{which}/maxquant/combined/txt/msms.txt"
        raw_dir = f"/data2/yejb/prosit/figs/fig235/{which}/raw"

        tabels_file = fixed_features(msms_file, raw_dir,
                                     f"/data2/yejb/prosit/figs/fig235/{which}/percolator_up/try/prosit_l1")
        save_tab1 = f"/data2/yejb/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/no_finetuned_twofold"
        save_tab2 = f"/data2/yejb/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/finetuned_twofold"
        if not os.path.exists(save_tab2):
            os.mkdir(save_tab2)
        finetune_model1, finetune_model2, id2remove = finetune.semisupervised_finetune_twofold(
            run_model, tabels_file, pearson=if_pearson)
        print(eval_fdr(finetune_model1, finetune_model2, msms_file, raw_dir, save_tab2,
              irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove, pearson=if_pearson).to_string())
        # analysis_dict[which] = overlap_analysis_peptides(save_tab1, save_tab2)

    print("-------------------------------")
    print("Davis")
    save_tab = f"/data2/yejb/prosit/figs/figure5/percolator/try/{frag_model}"
    if not os.path.exists(save_tab):
        os.mkdir(save_tab)
    msms_file = f"/data2/yejb/prosit/figs/figure5/maxquant/combined/txt/msms.txt"
    raw_dir = f"/data2/yejb/prosit/figs/figure5/raw"

    tabels_file = fixed_features(msms_file, raw_dir,
                                 f"/data2/yejb/prosit/figs/figure5/percolator/try/prosit_l1")
    save_tab1 = f"/data2/yejb/prosit/figs/figure5/percolator/try/{frag_model}/no_finetuned_twofold"
    save_tab2 = f"/data2/yejb/prosit/figs/figure5/percolator/try/{frag_model}/finetuned_twofold"
    if not os.path.exists(save_tab2):
        os.mkdir(save_tab2)

    finetune_model1, finietune_model2, id2remove = finetune.semisupervised_finetune_twofold(
        run_model, tabels_file)

    print(eval_fdr(finetune_model1, finetune_model2, msms_file, raw_dir, save_tab2,
                   irt_model=prosit_irt, sample_size=sample_size, id2remove=id2remove, pearson=if_pearson).to_string())
    # analysis_dict['davis'] = overlap_analysis_peptides(save_tab1, save_tab2)
# print(analysis_dict)

import pandas as pd
from contextlib import redirect_stdout
import sys

# from figs.fdr_test import fdr_test_reverse
sys.path.append("./figs")
import torch
import os
from fdr_test import fdr_test, fixed_features
from ms import model
from ms import finetune


def eval_fdr(run_model, msms_file, raw_dir, save_tab, fdr_threshold=0.01, show_fdr=[0.1, 0.01, 0.001, 0.0001], sample_size=None, need_all=False, irt_model=None, pearson=False):
    run_model = run_model.eval()
    record = {}
    record['fdrs'] = [100 * i for i in show_fdr]
    with torch.no_grad():
        fdr_test(run_model, msms_file, raw_dir, save_tab, sample_size=sample_size,
                 need_all=need_all, irt_model=irt_model, pearson=pearson)

    totest = ["andromeda", "sa", "prosit_combined"]
    if irt_model is not None:
        totest.append("prosit_best")
    print(" start percolator... ")
    for name in totest:
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

    return pd.DataFrame(record)


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
    if_pearson = (frag_model in ['pdeep2'])
    print("Running", frag_model)
    for which in ["sprot_human", "IGC", "sprot_all", "sprot_bacteria_human"]:
        print("-------------------------------")
        print(which)
        save_tab = f"/data/prosit/figs/figure6/{which}/percolator/try/{frag_model}/no_finetuned_twofold"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        msms_file = f"/data/prosit/figs/figure6/{which}/maxquant/txt/msms.txt"
        raw_dir = f"/data/prosit/figs/figure6/all_raws"
        print(eval_fdr(run_model, msms_file, raw_dir, save_tab, irt_model=prosit_irt,
              sample_size=sample_size, pearson=if_pearson).to_string())

    # from pprint import pprint
    for which in ["trypsin", 'chymo', "lysc", "gluc"]:
        print("-------------------------------")
        print(which)
        save_tab = f"/data/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/no_finetuned_twofold"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        msms_file = f"/data/prosit/figs/fig235/{which}/maxquant/combined/txt/msms.txt"
        raw_dir = f"/data/prosit/figs/fig235/{which}/raw"
        print(eval_fdr(run_model, msms_file, raw_dir, save_tab, irt_model=prosit_irt,
              sample_size=sample_size, pearson=if_pearson).to_string())
    print("-------------------------------")
    print("Davis")
    save_tab = f"/data/prosit/figs/figure5/percolator/try/{frag_model}/no_finetuned_twofold"
    if not os.path.exists(save_tab):
        os.mkdir(save_tab)
    msms_file = f"/data/prosit/figs/figure5/maxquant/combined/txt/msms.txt"
    raw_dir = f"/data/prosit/figs/figure5/raw"
    print(eval_fdr(run_model, msms_file, raw_dir, save_tab, irt_model=prosit_irt,
          sample_size=sample_size, pearson=if_pearson).to_string())

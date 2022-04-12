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

def eval_fdr(run_model, msms_file, raw_dir, save_tab, fdr_threshold=0.1, show_fdr=[0.1, 0.01, 0.001, 0.0001],sample_size=None, irt_model=None, gpu_index=0):
    run_model = run_model.eval()
    record = {}
    record['fdrs'] = [100*i for i in show_fdr]
    with torch.no_grad():
        fdr_test(run_model, msms_file, raw_dir, save_tab, sample_size=sample_size, irt_model=irt_model, gpu_index=gpu_index)
    
    totest = ["andromeda", "sa", "combined", "prosit_combined"]
    if irt_model is not None:
        totest.append("prosit_best")
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
    gpu_index = 1
    print("Running", frag_model)
    # "HLA_2", "Mel1",
    for which in ["HLA_1"]:
        print("-------------------------------")
        print(which)
        save_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/percolator/{frag_model}_finetune/no_finetuned_all"
        if not os.path.exists(save_tab):
            os.mkdir(save_tab)
        msms_file = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/forPride/txt/msms.txt"
        raw_dir = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_{which}/forPride/rescoring_for_paper_2/raw"

        print(eval_fdr(run_model, msms_file, raw_dir, save_tab, irt_model=prosit_irt,
              sample_size=sample_size, gpu_index=gpu_index).to_string())

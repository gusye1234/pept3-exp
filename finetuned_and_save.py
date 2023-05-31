import torch
import sys
sys.path.append("figs/")
from pept3 import finetune
from fdr_test import fixed_features


which = "chymo"
print(which)
msms_file = f"/data/prosit/figs/fig235/{which}/maxquant/combined/txt/msms.txt"
raw_dir = f"/data/prosit/figs/fig235/{which}/raw"
fixed_features_dir = f"/data/prosit/figs/fig235/{which}/percolator_up/try/prosit_l1"
tabels_file = fixed_features(
    msms_file, raw_dir, fixed_features_dir)

from pept3 import model
import pept3

frag_model = "prosit_l1"
model_list = {
    "prosit_l1": model.PrositFrag,
    "pdeep2": model.pDeep2_nomod
}

checkpoints_list = {
    "prosit_l1": "/home/gus/Desktop/ms_pred/checkpoints/best/best_frag_l1_PrositFrag-1024.pth",
    "pdeep2": "/home/gus/Desktop/ms_pred/checkpoints/best/best_frag_l1_pDeep2-1024.pth"
}
run_model = model_list[frag_model]()
run_model.load_state_dict(torch.load(
    checkpoints_list[frag_model], map_location="cpu"))
prosit = run_model.eval()
if True:
    print(frag_model)
    finetuned_prosit1, finetuned_prosit2, id2remove = finetune.semisupervised_finetune_twofold(
        prosit, tabels_file, pearson=(frag_model == 'pdeep2'), enable_test=False)

    torch.save(finetuned_prosit1.state_dict(),
               f"/home/gus/Desktop/ms_pred/checkpoints/finetuned/{frag_model}_{which}-1.pth")
    torch.save(finetuned_prosit2.state_dict(),
               f"/home/gus/Desktop/ms_pred/checkpoints/finetuned/{frag_model}_{which}-2.pth")

import pandas as pd


def counting(msmsfile, percolator_tab_file):
    with open(msmsfile) as f:
        psms_counting = sum(1 for _ in f)
    per_tab = pd.read_csv(percolator_tab_file, sep='\t')
    filtered_psms = len(per_tab)
    filter_target = (per_tab['Label'] == 1).sum()
    filter_decoy = (per_tab['Label'] == -1).sum()
    del per_tab
    # psms_result = pd.read_csv(percolator_psms, sep='\t')
    # prosit_1 = (psms_result['q-value'] < 0.01).sum()
    # psms_result = pd.read_csv(finetuned_psms, sep='\t')
    # ft_1 = (psms_result['q-value'] < 0.01).sum()
    return psms_counting, filtered_psms, filter_target, filter_decoy


core_cols = ['Dataset', "Sub-type", "#Maxquant 100% FDR PSMs", "#Filtered 100% PSMs",
             "#Targets in Filtered 100% PSMs", "#Decoys in Filtered 100% PSMs"]
core_data = []

# --------------

dataset = "Bekker-Jenses"
frag_model = 'prosit_l1'
for which in ["trypsin", 'chymo', "lysc", "gluc"]:
    msms_file = f"/data/prosit/figs/fig235/{which}/maxquant/combined/txt/msms.txt"
    save_tab1 = f"/data/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/no_finetuned_twofold/sa.tab"
    save_tab2 = f"/data/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/finetuned_twofold/sa.tab"
    counts = counting(msms_file, save_tab1, save_tab2)
    core_data.append((
        dataset, which, *counts
    ))

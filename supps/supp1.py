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
regenerated_those_three = False
if regenerated_those_three:
    dataset = "Bekker-Jensen"
    frag_model = 'prosit_l1'
    for which in ["trypsin", 'chymo', "lysc", "gluc"]:
        msms_file = f"/data/prosit/figs/fig235/{which}/maxquant/combined/txt/msms.txt"
        save_tab1 = f"/data/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/no_finetuned_twofold/sa.tab"
        # save_tab2 = f"/data/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/finetuned_twofold/sa.tab"
        counts = counting(msms_file, save_tab1)
        core_data.append((
            dataset, which, *counts
        ))

    # ------------------
    dataset = "Davis"
    frag_model = 'prosit_l1'
    msms_file = f"/data/prosit/figs/figure5/maxquant/combined/txt/msms.txt"
    save_tab1 = save_tab1 = f"/data/prosit/figs/figure5/percolator/try/{frag_model}/no_finetuned_twofold/sa.tab"
    # save_tab2 = f"/data/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/finetuned_twofold/sa.tab"
    counts = counting(msms_file, save_tab1)
    core_data.append((
        dataset, "", *counts
    ))
    # ---------------------
    dataset = 'Metaproteomics'
    frag_model = 'prosit_l1'
    for which in ["sprot_human", "IGC", "sprot_all", "sprot_bacteria_human"]:
        msms_file = f"/data/prosit/figs/figure6/{which}/maxquant/txt/msms.txt"
        save_tab1 = f"/data/prosit/figs/figure6/{which}/percolator/try/{frag_model}/no_finetuned_twofold/sa.tab"
        counts = counting(msms_file, save_tab1)
        core_data.append((
            dataset, which, *counts
        ))
else:
    core_data.extend([
        ("Bekker-Jensen", 'trypsin', 761563, 663031, 502736, 160295),
        ("Bekker-Jensen", 'chymo', 990431, 921917, 596067, 325850),
        ("Bekker-Jensen", 'lysc', 834749, 714865, 500689, 214176),
        ("Bekker-Jensen", 'gluc', 898603, 774544, 514496, 260048),
        ('Davis', '', 1148358, 141702, 105864, 35838),
        ('Metaproteomics', 'sprot_human', 355909, 337862, 175446, 162416),
        ('Metaproteomics', 'IGC', 358491, 342049, 228140, 113909),
        ('Metaproteomics', 'sprot_all', 351263, 334216, 187417, 146799),
        ('Metaproteomics', 'sprot_bacteria_human', 346649, 329605, 183636, 145969)
    ])


core_pd = pd.DataFrame(columns=core_cols, data=core_data)
print(core_pd.head())
core_pd.to_csv("data/supp1.csv", index=False)

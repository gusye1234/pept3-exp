import pandas as pd
from tqdm import tqdm


def counting(msmsfile, percolator_tab_file):
    with open(msmsfile) as f:
        psms_counting = sum(1 for _ in f)
    per_tab = pd.read_csv(percolator_tab_file, sep='\t')
    filtered_psms = len(per_tab)
    filter_target = (per_tab['Label'] == 1).sum()
    filter_decoy = (per_tab['Label'] == -1).sum()
    del per_tab
    return psms_counting, filtered_psms, filter_target, filter_decoy


def quick_1(psms):
    psms_result = pd.read_csv(psms, sep='\t')
    fdr_1 = (psms_result['q-value'] < 0.01).sum()
    return fdr_1


# --------------
core_cols = ['Dataset', "Sub-type", "#Maxquant 100% FDR PSMs", "#Filtered 100% PSMs",
             "#Targets in Filtered 100% PSMs", "#Decoys in Filtered 100% PSMs", "#rescored pDeep2 SA-feature 1% PSMs", "#finetuned pDeep2 SA-feature 1% PSMs", "#rescored prosit SA-feature 1% PSMs", "#finetuned prosit SA-feature 1% PSMs", "#rescored pDeep2 Prosit-feature 1% PSMs", "#finetuned pDeep2 Prosit-feature 1% PSMs", "#rescored prosit Prosit-feature 1% PSMs", "#finetuned prosit Prosit-feature 1% PSMs"]
core_data = []

regenerated_those_three = False
if regenerated_those_three:
    dataset = "Bekker-Jensen"
    frag_model = 'prosit_l1'
    for which in ["trypsin", 'chymo', "lysc", "gluc"]:
        msms_file = f"/data2/yejb/prosit/figs/fig235/{which}/maxquant/combined/txt/msms.txt"
        save_tab1 = f"/data2/yejb/prosit/figs/fig235/{which}/percolator_up/try/prosit_l1/no_finetuned_3fold/sa.tab"
        counts = counting(msms_file, save_tab1)
        fdr_1s = []
        for feat in ['sa', 'prosit_best']:
            for frag_model in ['pdeep2', 'prosit_l1']:
                save_tab = f"/data2/yejb/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/no_finetuned_3fold/{feat}_target.psms"
                ft_save_tab = f"/data2/yejb/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/finetuned_3fold_0.1/{feat}_target.psms"
                fdr_1s.append(quick_1(save_tab))
                fdr_1s.append(quick_1(ft_save_tab))
        core_data.append((
            dataset, which, *counts, *fdr_1s
        ))

    # ------------------
    dataset = "Davis"
    frag_model = 'prosit_l1'
    msms_file = f"/data2/yejb/prosit/figs/figure5/maxquant/combined/txt/msms.txt"
    save_tab1 = save_tab1 = f"/data2/yejb/prosit/figs/figure5/percolator/try/{frag_model}/no_finetuned_3fold/sa.tab"
    # save_tab2 = f"/data2/yejb/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/finetuned_3fold_0.1/sa.tab"
    counts = counting(msms_file, save_tab1)
    fdr_1s = []
    for feat in ['sa', 'prosit_best']:
        for frag_model in ['pdeep2', 'prosit_l1']:
            save_tab = f"/data2/yejb/prosit/figs/figure5/percolator/try/{frag_model}/no_finetuned_3fold/{feat}_target.psms"
            ft_save_tab = f"/data2/yejb/prosit/figs/figure5/percolator/try/{frag_model}/finetuned_3fold_0.1/{feat}_target.psms"
            fdr_1s.append(quick_1(save_tab))
            fdr_1s.append(quick_1(ft_save_tab))
    core_data.append((
        dataset, "", *counts, *fdr_1s
    ))
    # ---------------------
    dataset = 'Metaproteomics'
    frag_model = 'prosit_l1'
    for which in ["sprot_human", "IGC", "sprot_all", "sprot_bacteria_human"]:
        msms_file = f"/data2/yejb/prosit/figs/figure6/{which}/maxquant/txt/msms.txt"
        save_tab1 = f"/data2/yejb/prosit/figs/figure6/{which}/percolator/try/{frag_model}/no_finetuned_3fold/sa.tab"
        counts = counting(msms_file, save_tab1)
        fdr_1s = []
        for feat in ['sa', 'prosit_best']:
            for frag_model in ['pdeep2', 'prosit_l1']:
                save_tab = f"/data2/yejb/prosit/figs/figure6/{which}/percolator/try/{frag_model}/no_finetuned_3fold/{feat}_target.psms"
                ft_save_tab = f"/data2/yejb/prosit/figs/figure6/{which}/percolator/try/{frag_model}/finetuned_3fold_0.1/{feat}_target.psms"
                fdr_1s.append(quick_1(save_tab))
                fdr_1s.append(quick_1(ft_save_tab))
        core_data.append((
            dataset, which, *counts, *fdr_1s
        ))
else:
    core_data.extend([
        ('Bekker-Jensen', 'trypsin', 761563, 663031, 502736, 160295,
         336262, 344939, 336109, 343046, 343341, 348017, 344015, 346872),
        ('Bekker-Jensen', 'chymo', 990431, 921917, 596067, 325850,
         155740, 234868, 150314, 239636, 254858, 265466, 259423, 267979),
        ('Bekker-Jensen', 'lysc', 834749, 714865, 500689, 214176,
         261380, 278672, 261846, 280695, 281830, 288102, 282866, 287623),
        ('Bekker-Jensen', 'gluc', 898603, 774544, 514496, 260048,
         120788, 209760, 124314, 207521, 238502, 247307, 241551, 247290),
        ('Davis', '', 1148358, 141702, 105864, 35838, 58588,
         69201, 62117, 69775, 70209, 71125, 70380, 71271),
        ('Metaproteomics', 'sprot_human', 355909, 337862, 175446,
         162416, 9688, 16470, 11624, 15676, 16763, 18212, 17502, 18358),
        ('Metaproteomics', 'IGC', 358491, 342049, 228140, 113909,
         82524, 101779, 82615, 101129, 101635, 109092, 105573, 109101),
        ('Metaproteomics', 'sprot_all', 351263, 334216, 187417, 146799,
         20986, 30491, 22801, 33817, 34513, 38135, 36908, 39150),
        ('Metaproteomics', 'sprot_bacteria_human', 346649, 329605, 183636, 145969, 21490, 28438, 22962, 32010, 32744, 35303, 35144, 37111)])

df_1 = pd.DataFrame(columns=core_cols, data=core_data)

# ---------------------------
# HLA class 1
core_cols = ["Mels", "#Maxquant 100% FDR PSMs", "#Filtered 100% PSMs",
             "#Targets in Filtered 100% PSMs", "#Decoys in Filtered 100% PSMs", "#rescored prosit Prosit-feature 1% PSMs", "#finetuned prosit Prosit-feature 1% PSMs"]

core_data = []
dataset = "melanoma patient HLA class I"
hla_mel = pd.read_csv("../figs/data/HLA_Mel.csv")
hla_mel = hla_mel[hla_mel['Experiment'].apply(lambda x: x.endswith("HLA-I"))]
Mels = hla_mel['Experiment'].unique()
for which in tqdm(Mels):
    origin_prosit_tab = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/forPride/rescoring_for_paper_2/Mels/{which}/percolator/prosit.tab"
    msms_file = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/forPride/rescoring_for_paper_2/Mels/{which}/msms.txt"
    counts = counting(msms_file, origin_prosit_tab)
    no_ft_fdr1 = quick_1(
        f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/forPride/rescoring_for_paper_2/Mels/{which}/percolator/prosit_target.psms")
    ft_fdr1 = quick_1(
        f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/prosit_hcd/3fold_Mels_0.1/{which}/prosit_target.psms")
    core_data.append((
        which, *counts, no_ft_fdr1, ft_fdr1
    ))
df_2 = pd.DataFrame(columns=core_cols, data=core_data)
# ----------------------------------------
# IAA noIAA
core_cols = ["Alleles", "#Maxquant 100% FDR PSMs", "#Filtered 100% PSMs",
             "#Targets in Filtered 100% PSMs", "#Decoys in Filtered 100% PSMs", "#rescored prosit Prosit-feature 1% PSMs", "#finetuned prosit Prosit-feature 1% PSMs"]
core_data = []
dataset = 'monoallelic HLA Class I'
alleles_rawfile = {}
with open("../figs/data/allele_raw.txt") as f:
    for l in f:
        pack = l.strip().split("\t")
        alleles_rawfile[pack[0]] = set(pack[1:])
Alleles = sorted(alleles_rawfile.keys())
for which in tqdm(Alleles):
    origin_prosit_tab = f"/data2/yejb/prosit/figs/alleles/forPRIDE/Alleles/{which}/percolator/prosit.tab"
    msms_file = f"/data2/yejb/prosit/figs/alleles/forPRIDE/Alleles/{which}/msms.txt"
    counts = counting(msms_file, origin_prosit_tab)
    no_ft_fdr1 = quick_1(
        f"/data2/yejb/prosit/figs/alleles/forPRIDE/Alleles/{which}/percolator/prosit_target.psms")
    ft_fdr1 = quick_1(
        f"/data1/yejb/prosit/figure3/prosit_hcd/3fold_hdf5_allele_0.1/{which}/prosit_target.psms")
    core_data.append((
        which, *counts, no_ft_fdr1, ft_fdr1
    ))
df_3 = pd.DataFrame(columns=core_cols, data=core_data)

writer = pd.ExcelWriter("data/supp1.xlsx", engine='openpyxl')
df_1.to_excel(writer, sheet_name="Bekker,Davis,Metaproteomics", index=False)
df_2.to_excel(writer, sheet_name="melanoma patient HLA class I", index=False)
df_3.to_excel(writer, sheet_name='monoallelic HLA Class I', index=False)
writer.save()

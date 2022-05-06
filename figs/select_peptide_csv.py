import os
import numpy as np
import pandas as pd
from collections import defaultdict
from pprint import pprint
from zipfile import ZipFile

which = "IAA"


naming_map = {
    # for noIAA
    'B54_': "B5401",
    "B_2705": "B2705",
    "C_1601": "C1601",
    "C_1601": "C1601",
    "C_1502": "C1502",
    "B_2705": "B2705",
    "C_0602": "C0602",
    "A101": "A0101",
    'A31': "A3101",
    "A68": "A6801",
    "B51": "B5101",
    'B54': "B5401",
    "B57": "B5701",

}
naming_map = {v: k for k, v in naming_map.items()}
normal_names = ['A0101', 'A0201', 'A0204', 'A0207', 'A0301', 'A2402',
                'A3301', 'B4402', 'B4403', 'C0302', 'C0304', 'C1402']
for n in normal_names:
    naming_map[n] = n
noiaa_naming_map = naming_map

# -------------------------------
zip_data = ZipFile("/data1/yejb/Fig3_Maxquant.zip")
all_names = [f for f in zip_data.namelist() if f.startswith(
    f"forPRIDE/IAA/annotation") and f.endswith("csv")]
all_names = [os.path.basename(i) for i in all_names]

names = [i.split("_")[3]
         for i in all_names if i.split("_")[3][0] in "ABCG"]

names += [i.split("_")[4]
          for i in all_names if i.split("_")[3] == "HLA"]
iaa_names = sorted(list(set(names)))
# -------------------------------

no_tab_dir = f"/data1/yejb/prosit/figure3/supply_origin"
tab_dir_IAA = "/data1/yejb/prosit/figure3/percolator/prosit_hcd_finetune_IAA_0.01"
tab_dir_noIAA = "/data1/yejb/prosit/figure3/percolator/prosit_hcd_finetune_noIAA_0.01"
# tab_dir_IAA = f"/data1/yejb/prosit/figure3/forPRIDE/IAA/percolator"
# tab_dir_noIAA = f"/data1/yejb/prosit/figure3/forPRIDE/noIAA/percolator"

prosit_tab_IAA = pd.read_csv(os.path.join(
    tab_dir_IAA, 'prosit_target.psms'), sep='\t')
prosit_tab_noIAA = pd.read_csv(os.path.join(
    tab_dir_noIAA, 'prosit_target.psms'), sep='\t')
# prosit_tab = pd.concat([prosit_tab_IAA, prosit_tab_noIAA], ignore_index=True)

no_prosit_tab = pd.read_csv(os.path.join(
    no_tab_dir, 'supp3.csv'))


c_iaa = 0
c_noiaa = 0

total_table = None
iaa_noiaa_names = sorted(list(set(iaa_names + list(noiaa_naming_map.keys()))))

all_names = sorted(no_prosit_tab['Allele'].dropna().unique())
for contain_name in all_names:
    if contain_name[0] not in "ABCG":
        continue
    if contain_name not in iaa_noiaa_names:
        print("Wrong shot:", contain_name)
        continue
    print(contain_name)
    to_dir = f"{no_tab_dir}/Alleles"
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)

    threshold = 0.01
    index_tab = prosit_tab_IAA.set_index("PSMId")
    target_tab_iaa = index_tab.filter(like=contain_name, axis=0)
    target_tab_iaa.reset_index(inplace=True)
    c_iaa += len(target_tab_iaa)
    if contain_name in noiaa_naming_map:
        index_tab = prosit_tab_noIAA.set_index("PSMId")
        target_tab_noiaa = index_tab.filter(
            like=noiaa_naming_map[contain_name] + "_", axis=0)
        target_tab_noiaa.reset_index(inplace=True)
        target_tab = pd.concat(
            [target_tab_iaa, target_tab_noiaa], ignore_index=True)
        # print(" Both iaa and no iaa", len(
        #     target_tab_iaa), len(target_tab_noiaa))
        c_noiaa += len(target_tab_noiaa)
    else:
        target_tab = target_tab_iaa
    f_p_petides = target_tab[target_tab['q-value'] < threshold]['peptide'].apply(
        lambda x: x.strip("_").strip(".")).unique()

    nof_p_peptides = no_prosit_tab[no_prosit_tab['Allele'] == contain_name].dropna(
        subset=['Maximal rescored MaxQuant score (Prosit rescoring)'])['Sequence'].unique()
    sm_v2_peptides = no_prosit_tab[no_prosit_tab['Allele'] == contain_name].dropna(
        subset=['Maximal SpectrumMill HLA v2 score (original study)'])['Sequence'].unique()
    maxquant_peptides = no_prosit_tab[no_prosit_tab['Allele'] == contain_name].dropna(
        subset=['Maximal MaxQuant/Andromeda score'])['Sequence'].unique()

    maxquant_peptides = set(maxquant_peptides)
    f_p_petides = set(f_p_petides)
    nof_p_peptides = set(nof_p_peptides)
    sm_v2_peptides = set(sm_v2_peptides)

    print(
        f"  Maxquant {len(maxquant_peptides)}, SM {len(sm_v2_peptides)}, prosit {len(nof_p_peptides)}, finetuned {len(f_p_petides)}")

    for var_add, pep_add in zip(['prosit_add', 'finetuned_add', "SM_v2"], [nof_p_peptides, f_p_petides, sm_v2_peptides]):
        to_dir = f"{no_tab_dir}/{var_add}"
        if not os.path.exists(to_dir):
            os.mkdir(to_dir)
        if var_add != "SM_v2":
            add_peptides = pep_add - sm_v2_peptides
        else:
            add_peptides = sm_v2_peptides
        len_peptide = defaultdict(list)
        for p in add_peptides:
            len_peptide[len(p)].append(p)
        save_file = os.path.join(to_dir, f"add-{contain_name}.txt")
        with open(save_file, "w") as f:
            all_len = sorted(len_peptide.keys())
            for l in all_len:
                line = " ".join([str(l)] + len_peptide[l])
                f.write(line + "\n")

    name = 'prosit'
    all_peptides = list(f_p_petides.union(
        nof_p_peptides).union(sm_v2_peptides).union(maxquant_peptides))
    note = {
        "SM_v2": [],
        "Maxquant": [],
        "Prosit": [],
        "Fine-tuned Prosit": []
    }
    for p in all_peptides:
        note['SM_v2'].append(1 if p in sm_v2_peptides else 0)
        note['Prosit'].append(1 if p in nof_p_peptides else 0)
        note['Fine-tuned Prosit'].append(1 if p in f_p_petides else 0)
        note['Maxquant'].append(1 if p in maxquant_peptides else 0)
    note['Sequence'] = all_peptides
    note['Allele'] = contain_name

    allele_table = pd.DataFrame(
        note)[['Sequence', "Allele", "SM_v2", 'Maxquant', "Prosit", "Fine-tuned Prosit"]]
    if total_table is None:
        total_table = allele_table
    else:
        total_table = pd.concat([total_table, allele_table], ignore_index=True)

total_table.to_csv(os.path.join(
    no_tab_dir, f"q=0.01-fine-tuned-IAA-noIAA-{threshold}.csv"), index=False)
print(f"IAA Total len {len(prosit_tab_IAA)} - {c_iaa}")
print(f"noIAA Total len {len(prosit_tab_noIAA)} - {c_noiaa}")

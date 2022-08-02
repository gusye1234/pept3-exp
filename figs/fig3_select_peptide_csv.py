import os
import numpy as np
import pandas as pd
from collections import defaultdict
from pprint import pprint
from zipfile import ZipFile

# -------------------------------
no_tab_dir = f"/data1/yejb/prosit/figure3/supply_origin"
tab_dir_IAA_noIAA = "/data1/yejb/prosit/figure3/percolator/IAA_noIAA/"
prosit_tab_dir_IAA_noIAA = "/data1/yejb/prosit/figure3/forPRIDE/IAA_noIAA"

# prosit_tab_IAA_noIAA = pd.read_csv(os.path.join(
#     tab_dir_IAA_noIAA, 'prosit_target.peptides'), sep='\t')
# nof_tab_IAA_noIAA = pd.read_csv(os.path.join(
#     prosit_tab_dir_IAA_noIAA, 'prosit_target.peptides'), sep='\t')
# print((prosit_tab_IAA_noIAA['q-value'] < 0.01).sum(),
#       (nof_tab_IAA_noIAA['q-value'] < 0.01).sum())

no_prosit_tab = pd.read_csv(os.path.join(
    no_tab_dir, 'supp3.csv'))
all_alleles_names = sorted(no_prosit_tab['Allele'].dropna().unique())
# -------------------------------
# alleles_rawfile = {}
# with open("allele_raw.txt", 'w') as f:
#     for allele in all_alleles_names:
#         print(" ", allele)
#         df = pd.read_excel("./mono_peptides.xlsx", allele, engine="openpyxl")
#         need_raws = df['filename'].apply(lambda x: x.split(".")[0])
#         alleles_rawfile[allele] = set(need_raws)
#         f.write("\t".join([allele] + list(alleles_rawfile[allele])) + "\n")
alleles_rawfile = {}
with open("data/allele_raw.txt") as f:
    for l in f:
        pack = l.strip().split("\t")
        alleles_rawfile[pack[0]] = set(pack[1:])

total_table = None
# f_index_tab = prosit_tab_IAA_noIAA.set_index("PSMId")
# nf_index_tab = nof_tab_IAA_noIAA.set_index("PSMId")
allele_len_tab = 0
nf_allele_len_tab = 0

prosit_DIR = "/data1/yejb/prosit/figure3/forPRIDE/Alleles"
ft_DIR = "/data1/yejb/prosit/figure3/percolator_hdf5_allele_0.1"
for contain_name in all_alleles_names:
    if contain_name[0] not in "ABCG":
        continue
    if contain_name not in alleles_rawfile:
        print("Wrong shot:", contain_name)
        continue
    print(contain_name)
    to_dir = f"{no_tab_dir}/Alleles"
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)

    threshold = 0.01

    target_tab = pd.read_csv(os.path.join(
        ft_DIR, f"{contain_name}/prosit_target.peptides"), sep='\t')
    allele_len_tab += len(target_tab)
    f_p_petides = target_tab[target_tab['q-value'] < threshold]['peptide'].apply(
        lambda x: x.strip("_").strip(".")).unique()
    # all_rawfiles = list(target_tab['PSMId'].apply(
    #     lambda x: x.split('-')[0]).unique())

    target_tab = pd.read_csv(os.path.join(
        prosit_DIR, f"{contain_name}/percolator/prosit_target.peptides"), sep='\t')
    nf_allele_len_tab += len(target_tab)
    nof_p_peptides = target_tab[target_tab['q-value'] < threshold]['peptide'].apply(
        lambda x: x.strip("_").strip(".")).unique()

    csv_nof_p_peptides = no_prosit_tab[no_prosit_tab['Allele'] == contain_name].dropna(
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
        f"  Maxquant {len(maxquant_peptides)}, SM {len(sm_v2_peptides)}, prosit(origin) {len(csv_nof_p_peptides)}, prosit {len(nof_p_peptides)}, finetuned {len(f_p_petides)}")

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
        save_file = os.path.join(to_dir, f"len-add-{contain_name}.txt")
        with open(save_file, "w") as f:
            all_len = sorted(len_peptide.keys())
            for l in all_len:
                line = " ".join([str(l)] + len_peptide[l])
                f.write(line + "\n")
        save_file = os.path.join(to_dir, f"{contain_name}.txt")
        with open(save_file, "w") as f:
            all_pep_add = sorted(list(pep_add))
            f.write("\n".join(all_pep_add))

        if var_add != "SM_v2":
            save_file = os.path.join(to_dir, f"sub-{contain_name}.txt")
            pep_sub = (sm_v2_peptides - pep_add)
            with open(save_file, "w") as f:
                pep_sub = [p for p in pep_sub if (
                    len(p) >= 8 and len(p) <= 11)]
                all_pep_add = sorted(list(pep_sub))
                f.write("\n".join(all_pep_add))

            save_file = os.path.join(to_dir, f"add-{contain_name}.txt")
            pep_more = (pep_add - sm_v2_peptides)
            with open(save_file, "w") as f:
                pep_more = [p for p in pep_more if (
                    len(p) >= 8 and len(p) <= 11)]
                all_pep_add = sorted(list(pep_more))
                f.write("\n".join(all_pep_add))

            save_file = os.path.join(to_dir, f"share-{contain_name}.txt")
            pep_share = pep_add.intersection(sm_v2_peptides)
            with open(save_file, "w") as f:
                pep_share = [p for p in pep_share if (
                    len(p) >= 8 and len(p) <= 11)]
                all_pep_add = sorted(list(pep_share))
                f.write("\n".join(all_pep_add))
        else:
            to_dir = f"{no_tab_dir}/ft_nonft"
            if not os.path.exists(to_dir):
                os.mkdir(to_dir)
            
            pep_more = (f_p_petides - nof_p_peptides)
            with open(save_file, "w") as f:
                pep_more = [p for p in pep_more if (
                    len(p) >= 8 and len(p) <= 11)]
                all_pep_add = sorted(list(pep_more))
                f.write("\n".join(all_pep_add))
            len_peptide = defaultdict(list)
            for p in pep_more:
                len_peptide[len(p)].append(p)
            save_file = os.path.join(to_dir, f"len-add-{contain_name}.txt")
            with open(save_file, "w") as f:
                all_len = sorted(len_peptide.keys())
                for l in all_len:
                    line = " ".join([str(l)] + len_peptide[l])
                    f.write(line + "\n")
            save_file = os.path.join(to_dir, f"{contain_name}.txt")
            with open(save_file, "w") as f:
                all_pep_add = sorted(list(pep_add))
                f.write("\n".join(all_pep_add))
                
            save_file = os.path.join(to_dir, f"sub-{contain_name}.txt")
            pep_sub = (nof_p_peptides - f_p_petides)
            with open(save_file, "w") as f:
                pep_sub = [p for p in pep_sub if (
                    len(p) >= 8 and len(p) <= 11)]
                all_pep_add = sorted(list(pep_sub))
                f.write("\n".join(all_pep_add))

            save_file = os.path.join(to_dir, f"add-{contain_name}.txt")
            pep_more = (f_p_petides - nof_p_peptides)
            with open(save_file, "w") as f:
                pep_more = [p for p in pep_more if (
                    len(p) >= 8 and len(p) <= 11)]
                all_pep_add = sorted(list(pep_more))
                f.write("\n".join(all_pep_add))

            save_file = os.path.join(to_dir, f"share-{contain_name}.txt")
            pep_share = f_p_petides.intersection(nof_p_peptides)
            with open(save_file, "w") as f:
                pep_share = [p for p in pep_share if (
                    len(p) >= 8 and len(p) <= 11)]
                all_pep_add = sorted(list(pep_share))
                f.write("\n".join(all_pep_add))

    name = 'prosit'
    all_peptides = list(f_p_petides.union(
        nof_p_peptides).union(sm_v2_peptides).union(maxquant_peptides).union(csv_nof_p_peptides))
    note = {
        "SM_v2": [],
        "Maxquant": [],
        "Prosit": [],
        "Prosit(original)": [],
        "Fine-tuned Prosit": []
    }
    for p in all_peptides:
        note['SM_v2'].append(1 if p in sm_v2_peptides else 0)
        note['Prosit'].append(1 if p in nof_p_peptides else 0)
        note['Fine-tuned Prosit'].append(1 if p in f_p_petides else 0)
        note['Maxquant'].append(1 if p in maxquant_peptides else 0)
        note['Prosit(original)'].append(1 if p in csv_nof_p_peptides else 0)
    note['Sequence'] = all_peptides
    note['Allele'] = contain_name

    allele_table = pd.DataFrame(
        note)[['Sequence', "Allele", "SM_v2", 'Maxquant', "Prosit(original)", "Prosit", "Fine-tuned Prosit"]]
    if total_table is None:
        total_table = allele_table
    else:
        total_table = pd.concat([total_table, allele_table], ignore_index=True)


# total_rawfile = set(prosit_tab_IAA_noIAA['PSMId'].apply(
#     lambda x: "-".join(x.split('-')[:-3])).unique())
# known_rawfile = set(alleles_rawfile_log['Rawfile'])
# unknown_rawfile = total_rawfile - known_rawfile
# alleles_rawfile_log['Allele'] = ['Unknown'] * \
#     len(unknown_rawfile) + alleles_rawfile_log['Allele']
# alleles_rawfile_log['Rawfile'] = list(
#     unknown_rawfile) + alleles_rawfile_log['Rawfile']

# df = pd.DataFrame(identified_log_dict)
# df.to_csv("./identified.csv", sep='\t')
# df = pd.DataFrame(alleles_rawfile_log)
# df.to_csv("./allele_rawfile.csv", index=False, sep='\t')
total_table.to_csv(os.path.join(
    no_tab_dir, f"combined-fine-tuned-IAA-noIAA-{threshold}.csv"), index=False)
# print(f"IAA Total len {len(prosit_tab_IAA_noIAA)} - {allele_len_tab}")
# print(f"noIAA Total len {len(nof_tab_IAA_noIAA)} - {nf_allele_len_tab}")

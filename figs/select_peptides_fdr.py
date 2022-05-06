import os
import numpy as np
import pandas as pd
from collections import defaultdict
from pprint import pprint
from zipfile import ZipFile

which = "IAA"

no_tab_dir = f"/data1/yejb/prosit/figure3/forPRIDE/{which}/percolator"
tab_dir = f"/data1/yejb/prosit/figure3/percolator/prosit_hcd_finetune_{which}"

prosit_tab = pd.read_csv(os.path.join(
    tab_dir, 'prosit_target.psms'), sep='\t')
no_prosit_tab = pd.read_csv(os.path.join(
    no_tab_dir, 'prosit_target.psms'), sep='\t')
print(prosit_tab[prosit_tab['q-value'] < 0.01]['peptide'].unique().shape,
      no_prosit_tab[no_prosit_tab['q-value'] < 0.01]['peptide'].unique().shape)
print(len(prosit_tab), len(no_prosit_tab))

zip_data = ZipFile("/data1/yejb/Fig3_Maxquant.zip")
all_names = [f for f in zip_data.namelist() if f.startswith(
    f"forPRIDE/{which}/annotation") and f.endswith("csv")]
all_names = [os.path.basename(i) for i in all_names]

names = [i.split("_")[3]
         for i in all_names if i.split("_")[3][0] in "ABCG"]

names += [i.split("_")[4]
          for i in all_names if i.split("_")[3] == "HLA"]
names = set(names)
c = 0

add_hit = 0
de_hit = 0

# names = ['C1203', "B3802", "C0403"]
for contain_name in names:
    if contain_name[0] not in "ABCG":
        continue
    print(contain_name)
    to_dir = f"{tab_dir}/Alleles"
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)

    to_no_dir = f"{no_tab_dir}/Alleles"
    if not os.path.exists(to_no_dir):
        os.mkdir(to_no_dir)
    # print(f"Total len {len(prosit_tab)}")
    index_tab = prosit_tab.set_index("PSMId")
    target_tab = index_tab.filter(like="_" + contain_name, axis=0)
    target_tab.reset_index(inplace=True)
    print(f"Total len {len(target_tab)}")

    index_tab = no_prosit_tab.set_index("PSMId")
    target_no_tab = index_tab.filter(like="_" + contain_name, axis=0)
    target_no_tab.reset_index(inplace=True)
    print(f"Total len {len(target_no_tab)}")
    c += len(target_tab)

    # n1 = set(target_tab['PSMId'])
    # n2 = set(target_no_tab['PSMId'])
    # print(len(n1), len(n1.intersection(n2)), len(n2))
    # pprint(sorted(list(target_tab['PSMId']))[:10])
    # pprint(sorted(list(target_no_tab['PSMId']))[:10])

    name = 'prosit'
    show_fdr = [0.1, 0.01, 0.001]

    for fdr in show_fdr:
        print(
            f"{fdr:5}: {(target_tab['q-value'] < fdr).sum()}, {(target_no_tab['q-value'] < fdr).sum()}")

    threshold = 0.01
    for dir_name, tab in zip([to_dir, to_no_dir], [target_tab, target_no_tab]):
        save2 = tab[tab['q-value'] < threshold]
        all_peptides = save2['peptide'].apply(
            lambda x: x.strip("_").strip(".")).unique()
        len_peptide = defaultdict(list)
        for p in all_peptides:
            len_peptide[len(p)].append(p)

        save_file = os.path.join(dir_name, f"{contain_name}-{threshold}.txt")
        with open(save_file, "w") as f:
            all_len = sorted(len_peptide.keys())
            for l in all_len:
                line = " ".join([str(l)] + len_peptide[l])
                f.write(line + "\n")
    all_peptides = target_tab[target_tab['q-value'] <= threshold]
    all_peptides = all_peptides['peptide'].apply(
        lambda x: x.strip("_").strip(".")).unique()

    all_no_peptides = target_no_tab[target_no_tab['q-value'] <= threshold]
    all_no_peptides = all_no_peptides['peptide'].apply(
        lambda x: x.strip("_").strip(".")).unique()

    all_peptides = set(all_peptides)
    all_no_peptides = set(all_no_peptides)
    overlap = all_peptides.intersection(all_no_peptides)
    # with open("See.txt", 'w') as f:
    #     for l in sorted(list(all_no_peptides)):
    #         f.write(l + "\n")
    print(len(all_no_peptides), len(all_peptides))
    print(len(all_no_peptides) - len(overlap),
          len(overlap), len(all_peptides) - len(overlap))
    if (len(all_peptides) - len(overlap)) > (len(all_no_peptides) - len(overlap)):
        add_hit += 1
    else:
        de_hit += 1
    len_peptide = defaultdict(list)
    for p in (all_peptides - all_no_peptides):
        len_peptide[len(p)].append(p)
    save_file = os.path.join(to_dir, f"add-{contain_name}-{threshold}.txt")
    with open(save_file, "w") as f:
        all_len = sorted(len_peptide.keys())
        for l in all_len:
            line = " ".join([str(l)] + len_peptide[l])
            f.write(line + "\n")

print(f"Total len {len(prosit_tab)} - {c}")
print(add_hit, de_hit)

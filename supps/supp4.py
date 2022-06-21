import os
import pandas as pd


def read2pd(filename):
    sep = '-'
    if not os.path.exists(filename):
        print(f"{filename} not exist!")
        return None
    with open(filename) as f:
        while True:
            l = f.readline()
            if l == '':
                return None
            l = l.strip()
            if l.startswith(sep):
                head = f.readline().strip().split()
                f.readline()
                break
        content = []
        for _ in range(len(head)):
            content.append([])
        while True:
            l = f.readline().strip()
            if l.startswith(sep):
                break
            l = l.split()
            if len(l) == 15:
                lastterm = l.pop()
                l[-1] = l[-1] + lastterm
            elif len(l) == 13:
                l.append('')
            assert len(l) == len(head)
            for i in range(len(l)):
                content[i].append(l[i])
    pddict = {k: v for k, v in zip(head, content)}
    return pd.DataFrame(pddict)


def allele2hla(allele):
    return f"HLA-{allele[:3]}:{allele[3:]}"


def allele2hla_star(allele):
    return f"HLA-{allele[0]}*{allele[1:3]}:{allele[3:]}"


def netmhcpan_score_dict(dirs, allele):
    hla = allele2hla(allele)
    df_gain = read2pd(os.path.join(dirs, f"add-{hla}.out"))
    df_share = read2pd(os.path.join(dirs, f"share-{hla}.out"))
    df = pd.concat([df_gain, df_share], ignore_index=True)
    return {k: (v1, v2, v3) for k, v1, v2, v3 in zip(df['Peptide'], df['Score_EL'].astype("float"), df['%Rank_EL'], df['BindLevel'])}


def transphla_score_dict(dirs, allele):
    hla = allele2hla_star(allele)
    try:
        df_gain = pd.read_csv(os.path.join(dirs, f"add-{hla}.csv"))
        df_share = pd.read_csv(os.path.join(dirs, f"share-{hla}.csv"))
        df = pd.concat([df_gain, df_share], ignore_index=True)
        return {k: v for k, v in zip(df['peptide'], df['y_prob'].astype("float"))}
    except FileNotFoundError:
        return {}


no_tab_dir = f"/data1/yejb/prosit/figure3/supply_origin"
prosit_supp_data_tab = pd.read_csv(os.path.join(
    no_tab_dir, 'supp3.csv'))
all_alleles_names = sorted(prosit_supp_data_tab['Allele'].dropna().unique())
alleles_rawfile = {}

with open("../figs/data/allele_raw.txt") as f:
    for l in f:
        pack = l.strip().split("\t")
        alleles_rawfile[pack[0]] = set(pack[1:])

prosit_DIR = "/data1/yejb/prosit/figure3/forPRIDE/Alleles"
ft_DIR = "/data1/yejb/prosit/figure3/percolator_hdf5_allele_0.1"

netmhcpan_dir = "/data1/yejb/prosit/figure3/supply_origin/ft_nonft/netMHCpan_out"
transphla_dir = "/data1/yejb/prosit/figure3/supply_origin/ft_nonft/transphla/scores"
threshold = 0.01


def get_peptide_dict(df_file, threshold):
    target_tab = pd.read_csv(df_file, sep='\t')
    peptides = target_tab[target_tab['q-value'] < threshold]['peptide'].apply(
        lambda x: x.strip("_").strip(".")).unique()
    peptides_dict = {k.strip("_").strip("."): v for k, s, v in zip(
        target_tab['peptide'], target_tab['q-value'], target_tab['score']) if s <= threshold}
    return peptides, peptides_dict


def df_get_peptide_dict(df, allele_name, name):
    core = df[df['Allele'] == allele_name].dropna(
        subset=[name])
    peptides = core['Sequence'].unique()
    peptides_dict = {k: v for k, v in zip(core['Sequence'], core[name])}
    return peptides, peptides_dict


total_table = None
for contain_name in all_alleles_names:
    if contain_name[0] not in "ABCG":
        continue
    if contain_name not in alleles_rawfile:
        print("Wrong shot:", contain_name)
        continue

    ft_peps, ft_peps_dict = get_peptide_dict(
        os.path.join(
            ft_DIR, f"{contain_name}/prosit_target.peptides"), threshold
    )
    prosit_peps, prosit_peps_dict = get_peptide_dict(
        os.path.join(
            prosit_DIR, f"{contain_name}/percolator/prosit_target.peptides"), threshold
    )

    sm_v2_peps, sm_v2_peps_dict = df_get_peptide_dict(prosit_supp_data_tab,
                                                      contain_name,
                                                      'Maximal SpectrumMill HLA v2 score (original study)')
    prosit_ori_peps, prosit_ori_peps_dict = df_get_peptide_dict(prosit_supp_data_tab,
                                                                contain_name,
                                                                'Maximal rescored MaxQuant score (Prosit rescoring)')
    mq_peps, mq_peps_dict = df_get_peptide_dict(prosit_supp_data_tab,
                                                contain_name,
                                                'Maximal MaxQuant/Andromeda score')
    total_allele_peps = set().union(ft_peps, prosit_peps,
                                    sm_v2_peps, prosit_ori_peps, mq_peps)
    total_allele_peps = list(total_allele_peps)

    netmhcpan_scores = netmhcpan_score_dict(netmhcpan_dir, contain_name)
    transphla_scores = transphla_score_dict(transphla_dir, contain_name)
    core_cols = ['Sequence', "Allele",
                 "Maximal SpectrumMill HLA v2 score (original study)",
                 "Maximal MaxQuant/Andromeda score (original study)",
                 "Maximal rescored MaxQuant score (Prosit rescoring)(original study)",
                 "Prosit Percolator SVM score",
                 "Fine-tuned Prosit Percolator SVM score",
                 "NetMHCpan Score",
                 "NetMHCpan Rank",
                 "NetMHCpan Binding Level",
                 "Transphla Score"]
    core_data = []
    print(*[len(p) for p in [sm_v2_peps, mq_peps,
          prosit_ori_peps, prosit_peps, ft_peps]])

    for p in total_allele_peps:
        core_data.append([
            p,
            contain_name,
            sm_v2_peps_dict.get(p, float('NaN')),
            mq_peps_dict.get(p, float('NaN')),
            prosit_ori_peps_dict.get(p, float('NaN')),
            prosit_peps_dict.get(p, float('NaN')),
            ft_peps_dict.get(p, float('NaN')),
            *netmhcpan_scores.get(p, [float('NaN'), '', '']),
            transphla_scores.get(p, float('NaN')),
        ])

    allele_table = pd.DataFrame(columns=core_cols, data=core_data)
    total_table = pd.concat([total_table, allele_table], ignore_index=True)

writer = pd.ExcelWriter("data/supp4.xlsx", engine='openpyxl')
total_table.to_excel(writer, sheet_name="Finetuning over Alleles", index=False)
writer.save()

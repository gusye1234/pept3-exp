from xml.dom import NotFoundErr
import pandas as pd
import os
import re
from collections import defaultdict

hla_mel = pd.read_csv("../figs/data/HLA_Mel.csv")
hla_mel = hla_mel[hla_mel['Experiment'].apply(lambda x: x.endswith("HLA-I"))]

naming_map = {
    hla_mel.iloc[row]['Raw file']: hla_mel.iloc[row]['Experiment']
    for row in range(len(hla_mel))
}

MELS = set(hla_mel['Experiment'])


def read_mel(pep_dirs, melname, threshold=0.01, add_dir=None):
    from collections import defaultdict
    all_mels = defaultdict(list)
    if not add_dir:
        pep_path = os.path.join(
            pep_dirs, f"{melname}/prosit_target.peptides")
    else:
        pep_path = os.path.join(
            pep_dirs, f"{melname}/percolator/prosit_target.peptides")
    df = pd.read_csv(pep_path, sep='\t')
    f_p_peptides = df[df['q-value'] < threshold]['peptide'].apply(
        lambda x: x.strip("_").strip(".")).unique()
    f_p_dict = {
        k.strip("_").strip("."): v for k, v in zip(df['peptide'], df['q-value'])
        if v < threshold
    }
    return f_p_peptides, f_p_dict


ft_mels_table = None

prosit_mels_peptides = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/forPride/rescoring_for_paper_2/Mels/"
ft_mels_peptides = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/prosit_hcd/percolator_hdf5_Mels_0.1"

for melname in MELS:
    print(melname)
    prosit_peps, prosit_peps_dict = read_mel(
        prosit_mels_peptides, melname, add_dir=True)
    ft_peps, ft_peps_dict = read_mel(ft_mels_peptides, melname)
    core_cols = [
        'Experiments',
        'Modified Sequence',
        'Prosit Q-value',
        'Finetuned Prosit Q-values'
    ]
    total_peps = set().union(prosit_peps, ft_peps)
    core_data = []
    for p in total_peps:
        core_data.append((
            melname,
            p,
            prosit_peps_dict.get(p, float("NaN")),
            ft_peps_dict.get(p, float("NaN"))
        ))
    df = pd.DataFrame(columns=core_cols, data=core_data)
    ft_mels_table = pd.concat([ft_mels_table, df], ignore_index=True)


def extract_loc(data):
    all_locs = []
    new_data = []
    can_not_parse = 0
    out_bound = 0
    for pack in data:
        mut_flag = pack[0]
        missense = mut_flag.split("|")

        try:
            mis_loc = int(missense[11])
        except:
            can_not_parse += 1
            raise
        try:
            assert mis_loc <= len(pack[1])
            assert mis_loc > 0
        except:
            out_bound += 1
            raise
            # raise
        new_data.append(pack)
        all_locs.append(mis_loc)
    return all_locs, new_data


def if_frameshift(data):
    return "frameshift_variant" in data


def if_missense(data):
    return "missense" in data


def csv2peptide(csv_file):
    table = pd.read_csv(csv_file)
    peps_dict = defaultdict(list)
    for mut_type, mut_core, peps in zip(table['Mut Type'], table["Mut Protein"], table["Matched Peptides"]):
        for pep in peps.split(';'):
            peps_dict[pep].append((mut_type, mut_core))
    return peps_dict


def peptides2dict(tabfile):
    df = pd.read_csv(tabfile, sep='\t')
    f_p_peptides = df['peptide'].apply(
        lambda x: x.strip("_").strip(".")).unique()
    f_p_dict = {
        k.strip("_").strip(".").upper(): v for k, v in zip(df['peptide'], df['q-value'])
    }
    return f_p_peptides, f_p_dict


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


def mark_mut_pep(pep, pep_dict):
    assert len(pep_dict[pep]) > 0
    loc = extract_loc(pep_dict[pep])[0][0]
    core_type = pep_dict[pep][0][0]
    core_pep = pep_dict[pep][0][1]
    start = re.search(pep.lower(), core_pep.lower()).start()
    # print(core_type.split("|")[5], "|".join(core_type.split("|")[9:12]))
    # print(core_pep[loc - 1])
    local_start = loc - start - 1
    if local_start >= 0:
        if if_frameshift(core_type):
            re_pep = f"{pep[:local_start]}**{pep[local_start:]}"
        else:
            re_pep = f"{pep[:local_start]}*{pep[local_start]}*{pep[local_start + 1:]}"
    elif if_frameshift(core_type):
        re_pep = f"**{pep}"
    else:
        raise NotFoundErr("wrong matching")
    loc_range = (start + 1, start + 1 + len(pep))
    return loc, core_pep, re_pep, core_type, loc_range


def extract_info(mut_type):
    packs = mut_type.split("|")
    gene_name = packs[4]
    transcript = packs[7]
    mut = packs[5]
    # alter = packs[9].split(".")[1]
    alter = packs[9]
    return gene_name, transcript, mut, alter


prosit_peptide_file = "/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/forPride/rescoring_for_paper_2/percolator/prosit_target.peptides"
ft_peptide_file = "/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/percolator_hdf5_0.1/prosit_target.peptides"

_, prosit_peps_score_dict = peptides2dict(prosit_peptide_file)
_, ft_peps_score_dict = peptides2dict(ft_peptide_file)

prosit_mut_dict = csv2peptide("../figs/data/fig3_mut_prosit.csv")
ft_mut_dict = csv2peptide("../figs/data/fig3_mut_ft.csv")


alleles2see = ['HLA-A03:01',
               'HLA-A68:01',
               'HLA-B27:05',
               'HLA-B35:03']
netmhcpan_dfs = defaultdict(dict)
for allele in alleles2see:
    for name in ['loss', 'share', 'gain']:
        df = read2pd(
            f"../figs/data/netmhcpan/fig3_mut_ft-{name}-{allele}-netmhcpan.out")
        for k, v1, v2, v3 in zip(df['Peptide'], df['Score_EL'].astype("float"), df['%Rank_EL'], df['BindLevel']):
            if len(v3.strip()) != 0:
                netmhcpan_dfs[allele][k] = (v1, v2, v3)

prosit_mutation_table = {}
for p in prosit_mut_dict.keys():
    pro_loc, pro, neo_pep, mut_type, loc_range = mark_mut_pep(
        p, prosit_mut_dict)
    gene, trans, mut_t, alter = extract_info(mut_type)
    prosit_mutation_table[p] = [
        pro, f"({loc_range[0]}-{loc_range[1]})", neo_pep, gene, trans, mut_t, alter, prosit_peps_score_dict[p]]
    for allele in alleles2see:
        prosit_mutation_table[p].append(
            netmhcpan_dfs[allele].get(p, None))

ft_mutation_table = {}
for p in ft_mut_dict.keys():
    pro_loc, pro, neo_pep, mut_type, loc_range = mark_mut_pep(p, ft_mut_dict)
    gene, trans, mut_t, alter = extract_info(mut_type)
    ft_mutation_table[p] = [
        pro, f"({loc_range[0]}-{loc_range[1]})", neo_pep, gene, trans, mut_t, alter, prosit_peps_score_dict[p], ft_peps_score_dict[p]]
    for allele in alleles2see:
        ft_mutation_table[p].append(
            netmhcpan_dfs[allele].get(p, None))


def form_mut_df(mutation_table, ft=False):
    if not ft:
        core_cols = ["Experiments", "Sequence", "Protein", "Position",
                     "Neo-Epitope", "Gene name", "Transcript ID", "Mutation Type", "Amino-acid change", "Prosit peptide Q-value"] + \
            [f"NetMHCpan prediction \n{allele}\n(score;%rank;binding_level)" for allele in alleles2see]
    else:
        core_cols = ["Experiments", "Sequence", "Protein", "Position",
                     "Neo-Epitope", "Gene name", "Transcript ID", "Mutation Type", "Amino-acid change", "Prosit peptide Q-value", "Finetuned Prosit peptide Q-value"] + \
            [f"NetMHCpan prediction \n{allele}\n(score;%rank;binding_level)" for allele in alleles2see]
    core_data = []
    for k, v in mutation_table.items():
        core_data.append([
            "Mel-15_HLA-I",
            k,
            *v[:-4],
            *[f"{p[0]};{p[1]};{p[2]}" if p is not None else '' for p in v[-4:]],
        ])
    return pd.DataFrame(columns=core_cols, data=core_data)


def only_binders_form_mut_df(mutation_table, ft=False):
    if not ft:
        core_cols = ["Experiments", "Position",
                     "Neo-Epitope", "Gene name", "Transcript ID", "Mutation Type", "Amino-acid change", "Prosit peptide Q-value"] + \
            [f"NetMHCpan prediction \n{allele}\n(score;%rank;binding_level)" for allele in alleles2see]
    else:
        core_cols = ["Experiments", "Position",
                     "Neo-Epitope", "Gene name", "Transcript ID", "Mutation Type", "Amino-acid change", "Prosit peptide Q-value", "Finetuned Prosit peptide Q-value"] + \
            [f"NetMHCpan prediction \n{allele}\n(score;%rank;binding_level)" for allele in alleles2see]
    core_data = []
    for k, v in mutation_table.items():
        if all([p is None for p in v[-4:]]):
            continue
        core_data.append([
            "Mel-15_HLA-I",
            *v[1:-4],
            *[f"{p[0]};{p[1]};{p[2]}" if p is not None else '' for p in v[-4:]],
        ])
    return pd.DataFrame(columns=core_cols, data=core_data)


prosit_mut_df = form_mut_df(prosit_mutation_table)
ft_mut_df = form_mut_df(ft_mutation_table, ft=True)

prosit_only_binders = only_binders_form_mut_df(prosit_mutation_table)
prosit_only_binders.to_csv("data/mel15_prosit.csv", index=False)
ft_only_binders = only_binders_form_mut_df(ft_mutation_table, ft=True)
ft_only_binders.to_csv("data/mel15_ft.csv", index=False)

writer = pd.ExcelWriter("data/supp5.xlsx", engine='openpyxl')
# dfs = {
#     "Mel15 Prosit Neo-Epitopes": prosit_mut_df,
#     "Mel15 FT Prosit Neo-Epitopes": ft_mut_df
# }
ft_mels_table.to_excel(writer, sheet_name='HLA-I Rescoring', index=False)
prosit_mut_df.to_excel(
    writer, sheet_name="Mel15 Prosit Neo-Epitopes", index=False)
ft_mut_df.to_excel(
    writer, sheet_name="Mel15 FT Prosit Neo-Epitopes", index=False)

# for sheetname, df in dfs.items():  # loop through `dict` of dataframes
#     df.to_excel(writer, sheet_name=sheetname)  # send df to writer
#     worksheet = writer.sheets[sheetname]  # pull worksheet object
#     for idx, col in enumerate(df):  # loop through all columns
#         series = df[col]
#         max_len = max((
#             series.astype(str).map(len).max(),  # len of largest item
#             len(str(series.name))  # len of column name/header
#         )) + 1  # adding a little extra space
#         worksheet.set_column(idx, idx, max_len)  # set column width
writer.save()

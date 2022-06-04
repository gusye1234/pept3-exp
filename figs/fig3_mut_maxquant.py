from sys import prefix
import pandas as pd
from rich.progress import track


def read_peptide(check_dir, threshold=0.01):
    from os.path import join
    df = pd.read_csv(join(check_dir, "prosit_target.peptides"), sep='\t')
    f_p_peptides = df[df['q-value'] < threshold]['peptide'].apply(
        lambda x: x.strip("_").strip(".")).unique()
    return set(f_p_peptides)


def if_mutation(d):
    return "|" in d


def LSG(s1, s2):
    return s1 - s2, s1.intersection(s2), s2 - s1


protein_groups = pd.read_csv(
    '/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/forPride/txt/peptides.txt', sep='\t')
protein_groups = protein_groups[protein_groups['Reverse'] != '+']
# protein_groups = protein_groups[protein_groups['Proteins'].apply(
#     lambda x: True if len(x.split(";")) == 1 else False)]
protein_groups = protein_groups[protein_groups['Proteins'].apply(
    lambda x: all([if_mutation(p) for p in x.split(";")]))]

print(len(protein_groups))
assert protein_groups['Sequence'].is_unique

pep2allproteins = dict(
    zip(protein_groups['Sequence'], protein_groups['Proteins']))

pep2bestproteins = dict(
    zip(protein_groups['Sequence'], protein_groups['Leading razor protein']))

print(len(pep2allproteins), len(pep2bestproteins))
pep2allproteins = {k.lower(): v for k, v in pep2allproteins.items()}
pep2bestproteins = {k.lower(): v for k, v in pep2bestproteins.items()}


def find_best_mut(peptides, prefixs):
    mutations = []

    def check(prot):
        if prot is None:
            return False
        for pre in prefixs:
            if pre in prot:
                return True
        return False
    for pep in peptides:
        prot = pep2bestproteins.get(pep.lower(), None)
        if check(prot):
            mutations.append((pep, prot))
    return mutations


def find_all_mut(peptides, prefixs):
    mutations = []

    def check(prot):
        if prot is None:
            return False
        for pre in prefixs:
            if pre in prot:
                return True
        return False
    for pep in peptides:
        prot = pep2allproteins.get(pep.lower(), None)
        if check(prot):
            mutations.append((pep, prot))
    return mutations


prosit_peps = read_peptide(
    "/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/forPride/rescoring_for_paper_2/percolator", threshold=0.03)
ft_peps = read_peptide(
    "/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/percolator_hdf5_0.1", threshold=0.03)


# prosit_mut_best = find_best_mut(prosit_peps, [
#     "missense", "frameshift_variant", "non_coding_transcript_exon_variant"])
# ft_mut_best = find_best_mut(
#     ft_peps, ["missense", "frameshift_variant", "non_coding_transcript_exon_variant"])
# print(len(prosit_mut_best), len(ft_mut_best))

prosit_mut_all = find_all_mut(prosit_peps, [
    "missense", "frameshift_variant", "non_coding_transcript_exon_variant"])
ft_mut_all = find_all_mut(
    ft_peps, ["missense", "frameshift_variant", "non_coding_transcript_exon_variant"])
print(len(prosit_mut_all), len(ft_mut_all))
# --------------------------------
prosit_supp_data = pd.read_csv("./fig3_prosit_mel15_mut.csv")
all_pep_supp = set(prosit_supp_data['Sequence'])

# prosit_mut_best_pep = set(p[0] for p in prosit_mut_best)
# ft_mut_best_pep = set(p[0] for p in ft_mut_best)
# print([len(p) for p in LSG(all_pep_supp, prosit_mut_best_pep)])
# print([len(p) for p in LSG(all_pep_supp, ft_mut_best_pep)])

prosit_mut_all_pep = set(p[0] for p in prosit_mut_all)
ft_mut_all_pep = set(p[0] for p in ft_mut_all)
print([len(p) for p in LSG(all_pep_supp, prosit_mut_all_pep)])
print([len(p) for p in LSG(all_pep_supp, ft_mut_all_pep)])

# ---------------------
gain_df = None
Gain = LSG(all_pep_supp, prosit_mut_all_pep)[2]
print(len(Gain))
all_mut_peps = set(protein_groups['Sequence'])
for pep in Gain:
    before_len = len(gain_df) if gain_df is not None else 0
    part_df = protein_groups[protein_groups['Sequence'] == pep.upper()]
    gain_df = pd.concat([gain_df, part_df], ignore_index=True)
    if before_len == len(gain_df):
        print(pep, pep in all_mut_peps)
        print('---------------------------------')
gain_df.to_csv("fig3_mut_maxquant_prosit_peptides_gain.csv")

# ---------------------------------------
Gain = LSG(all_pep_supp, ft_mut_all_pep)[2]
Lost = LSG(all_pep_supp, ft_mut_all_pep)[0]
for name, p_set in zip(["loss", "gain"], [Lost, Gain]):
    print(len(p_set))
    gain_df = None
    all_mut_peps = set(protein_groups['Sequence'])
    for pep in p_set:
        before_len = len(gain_df) if gain_df is not None else 0
        part_df = protein_groups[protein_groups['Sequence'] == pep.upper()]
        gain_df = pd.concat([gain_df, part_df], ignore_index=True)
        if before_len == len(gain_df):
            print(pep, pep in all_mut_peps)
            print('---------------------------------')
    gain_df.to_csv(f"fig3_mut_maxquant_ft_peptides_{name}.csv")

import os
import re
from os.path import join
import pandas as pd
from collections import defaultdict
from typing import Dict, List

neo_file_prosit = "./data/fig3_mutation_our_prosit.txt"
# neo_file_ft_prosit = "./data/fig3_mutation_our_ft_twofold.txt"
neo_file_ft_prosit = "./data/fig3_mutation_our_ft_0.1.txt"
prosit_percolator_dir = "/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/forPride/rescoring_for_paper_2/percolator"
ft_prosit_percolator_dir = "/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/3fold_hdf5_0.1"
# ft_prosit_percolator_dir = "/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/percolator_hdf5_0.1"

def read_peptide(check_dir, threshold=0.01):
    df = pd.read_csv(join(check_dir, "prosit_target.peptides"), sep='\t')
    f_p_peptides = df[df['q-value'] < threshold]['peptide'].apply(
        lambda x: x.strip("_").strip(".")).unique()
    f_p_peptides = [p.upper() for p in f_p_peptides]
    return set(f_p_peptides), df


def read_fasta(mut_fasta):
    mut_pair = []
    with open(mut_fasta) as f:
        for l in f:
            l = l.strip()
            if l.startswith(">"):
                mut_pair.append([l])
            else:
                mut_pair[-1].append(l)
    return mut_pair


def read_mut(mut_txt):
    mut_pair = []
    with open(mut_txt) as f:
        curr_cur = []
        for l in f:
            l = l.strip()
            if len(l) == 0:
                continue
            if l.startswith("-----"):
                mut_pair.append(curr_cur)
                curr_cur = []
            else:
                curr_cur.append(l)
    mut_pep = defaultdict(list)
    for p in mut_pair:
        mut_pep["##".join(p[1:])].append(p[0])
    return mut_pep, mut_pair


def mut_dict2df(mut_dict, to_file):
    df_d = {}
    df_d['Mut Type'] = [k.split("##")[0] for k in mut_dict.keys()]
    df_d['Mut Protein'] = [k.split("##")[1] for k in mut_dict.keys()]
    df_d['Matched Peptides'] = [";".join(k) for k in mut_dict.values()]

    df = pd.DataFrame(df_d)
    df.to_csv(to_file)


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
            continue
        try:
            assert mis_loc <= len(pack[1])
            assert mis_loc > 0
        except:
            out_bound += 1
            continue
            # raise
        new_data.append(pack)
        all_locs.append(mis_loc)
    return all_locs, new_data


def extract_missense_loc(data):
    all_locs = []
    loc_pat = re.compile(r"p\.[a-zA-Z]*(\d+)[a-zA-Z]*")
    for pack in data:
        mut_flag = pack[0]
        missense = [p for p in mut_flag.split("|") if p.startswith("p.")]
        assert len(missense) == 1
        mis_loc = int(re.match(loc_pat, missense[0]).group(1))
        all_locs.append(mis_loc)
    return all_locs


def if_frameshift(data):
    return "frameshift_variant" in data


def if_missense(d):
    return "missense" in d


def check_missense_overlap(target, pep, loc):
    possible_left = max(0, loc - len(pep))
    possible_right = min(len(target), loc + len(pep) - 1)
    possible_area = target[possible_left:possible_right]
    return (pep.lower() in possible_area.lower())


def check_frameshift_overlap(target, pep, loc):
    possible_left = max(0, loc - len(pep))
    possible_area = target[possible_left:]
    return (pep.lower() in possible_area.lower())


# prosit_mut_dict, prosit_mut_pair = read_mut("./fig3_mutation_missense.txt")
# ft_mut_dict, ft_mut_pair = read_mut("./fig3_mutation_missense_ft.txt")

prosit_mut_dict, prosit_mut_pair = read_mut(
    neo_file_prosit)
ft_mut_dict, ft_mut_pair = read_mut(neo_file_ft_prosit)

# -------------------------------------


def vis_missense(mut_pair):
    targets = [p[2] for p in mut_pair]
    locs = extract_missense_loc([p[1:] for p in mut_pair])
    peps = [p[0] for p in mut_pair]
    core_types = [p[1] for p in mut_pair]
    for t, p, l, c in zip(targets, peps, locs, core_types):
        print('-------------------------------')
        assert check_missense_overlap(t, p, l, c)


# vis_missense(ft_mut_pair)
# -------------------------------------
FASTA = "/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/Mel15OP1_mut_only.fasta"
MUT_ALL = read_fasta(FASTA)
MUT_LOC, MUT_ALL = extract_loc(MUT_ALL)
prosit_supp_data = pd.read_csv("./data/fig3_prosit_mel15_mut.csv")
all_pep_supp = set([p.upper() for p in prosit_supp_data['Sequence']])

supp_mut_found_dict = {}
supp_mut_found_dict_list = defaultdict(list)
supp_mut_found_pair = []
not_f = 0
for p_p in all_pep_supp:
    found = False
    this_pack = None
    p_p = p_p
    for loc, pack in zip(MUT_LOC, MUT_ALL):
        if not found:
            this_pack = pack
        if (if_frameshift(pack[0]) and check_frameshift_overlap(pack[1], p_p, loc)) or check_missense_overlap(pack[1], p_p, loc):
            found = True
            starts = re.search(p_p.lower(), pack[1].lower()).start()
            supp_mut_found_dict_list[p_p].append((pack, starts))
            supp_mut_found_pair.append((p_p, pack[0], pack[1]))
    if not found:
        print("Not found", p_p)
        raise
    supp_mut_found_dict[p_p] = this_pack

all_supp_mut_types = []
all_supp_mut_proteins = []
for v in supp_mut_found_dict_list.values():
    all_supp_mut_types.extend([p[0][0] for p in v])
    all_supp_mut_proteins.extend([p[0][1] for p in v])
print("Supp Mut pep and type", len(supp_mut_found_dict_list),
      len(set(all_supp_mut_types)), len(set(all_supp_mut_proteins)))
print(set(all_supp_mut_proteins) - set(v[1]
      for v in supp_mut_found_dict.values()))
supp_mut_found_dict_missense = {
    k: v for k, v in supp_mut_found_dict.items() if if_missense(v[0])
}

missense_all_supp_mut_types = []
missense_all_supp_mut_proteins = []
for v in supp_mut_found_dict_list.values():
    missense_all_supp_mut_types.extend(
        [p[0][0] for p in v if if_missense(p[0][0])])
    missense_all_supp_mut_proteins.extend(
        [p[0][1] for p in v if if_missense(p[0][0])])

print("(missense)Supp Mut pep and type", len(supp_mut_found_dict_missense),
      len(set(missense_all_supp_mut_types)), len(set(missense_all_supp_mut_proteins)))
# -------------------------------------
prosit_pep, p_df = read_peptide(
    prosit_percolator_dir, threshold=0.03)
ft_pep, ft_df = read_peptide(
    ft_prosit_percolator_dir, threshold=0.03)
LOSS = prosit_pep - ft_pep
GAIN = ft_pep - prosit_pep
SHARE = ft_pep.intersection(prosit_pep)
print(len(LOSS), len(GAIN), len(SHARE))
print("Unique peptides:", len(set([p[0] for p in prosit_mut_pair])),
      len(set([p[0] for p in ft_mut_pair])))
print("Unique mut types:", len(set([p[1] for p in prosit_mut_pair])),
      len(set([p[1] for p in ft_mut_pair])))
print("Unique mut protein:", len(set([p[2] for p in prosit_mut_pair])),
      len(set([p[2] for p in ft_mut_pair])))

print("Prosit mut pep over LSG", len(set([p[0] for p in prosit_mut_pair if p[0] in LOSS])),
      len(set([p[0] for p in prosit_mut_pair if p[0] in SHARE])),
      len(set([p[0] for p in prosit_mut_pair if p[0] in GAIN])))
print("Prosit mut type over LSG", len(set(["##".join(p[1:]) for p in prosit_mut_pair if p[0] in LOSS])),
      len(set(["##".join(p[1:]) for p in prosit_mut_pair if p[0] in SHARE])),
      len(set(["##".join(p[1:]) for p in prosit_mut_pair if p[0] in GAIN])))
prosit_our_loss = set([p[0] for p in prosit_mut_pair if p[0] in LOSS])
print("ft mut pep over LSG", len(set([p[0] for p in ft_mut_pair if p[0] in LOSS])),
      len(set([p[0] for p in ft_mut_pair if p[0] in SHARE])),
      len(set([p[0] for p in ft_mut_pair if p[0] in GAIN])))
print("ft mut type over LSG", len(set(["##".join(p[1:]) for p in ft_mut_pair if p[0] in LOSS])),
      len(set(["##".join(p[1:]) for p in ft_mut_pair if p[0] in SHARE])),
      len(set(["##".join(p[1:]) for p in ft_mut_pair if p[0] in GAIN])))

mut_dict2df(prosit_mut_dict, "data/fig3_mut_prosit.csv")
mut_dict2df(ft_mut_dict, "data/fig3_mut_ft.csv")

reverse_prosit_dict_list = defaultdict(list)
reverse_ft_dict_list = defaultdict(list)

for p in prosit_mut_pair:
    reverse_prosit_dict_list[p[0]].append(p[1:])
for p in ft_mut_pair:
    reverse_ft_dict_list[p[0]].append(p[1:])
# -----------------------------------------------------------
gain_pep = set([p[0] for p in ft_mut_pair if p[0] in GAIN])
cols = ["Sequence", "FT q-value", "No Ft q-value",
        "Mutation Type", "Mutation Protein"]
p_df['proteinIds'] = p_df['proteinIds'].apply(lambda x: x.upper())
ft_df['proteinIds'] = ft_df['proteinIds'].apply(lambda x: x.upper())

data = []
for p in gain_pep:
    prosit_q = p_df[p_df['proteinIds'] == p]['q-value'].iloc[0]
    ft_q = ft_df[ft_df['proteinIds'] == p]['q-value'].iloc[0]
    for pack in reverse_ft_dict_list[p]:
        data.append([p, ft_q, prosit_q,
                     pack[0], pack[1]])
df = pd.DataFrame(columns=cols, data=data)
df.to_csv("data/fig3_mut_ft_gain_q_value.csv")


lost_pep = (all_pep_supp - set([p[0] for p in ft_mut_pair]))
cols = ["Sequence", "FT q-value", "No Ft q-value",
        "Mutation Type", "Mutation Protein"]
data = []
for p in lost_pep:
    prosit_q = p_df[p_df['proteinIds'] == p]['q-value'].iloc[0]
    ft_q = ft_df[ft_df['proteinIds'] == p]['q-value'].iloc[0]
    for pack in reverse_prosit_dict_list[p]:
        data.append([p, ft_q, prosit_q, pack[0], pack[1]])
df = pd.DataFrame(columns=cols, data=data)
df.to_csv("data/fig3_mut_ft_loss_q_value.csv")


lost_pep = set([p[0] for p in ft_mut_pair])
cols = ["Sequence", "FT q-value", "No Ft q-value",
        "Mutation Type", "Mutation Protein"]
data = []
for p in lost_pep:
    prosit_q = p_df[p_df['proteinIds'] == p]['q-value'].iloc[0]
    ft_q = ft_df[ft_df['proteinIds'] == p]['q-value'].iloc[0]
    for pack in reverse_ft_dict_list[p]:
        data.append([p, ft_q, prosit_q, pack[0], pack[1]])
df = pd.DataFrame(columns=cols, data=data)
df.to_csv("data/fig3_mut_ft_q_value.csv")
# ---------------------------------------------

prosit_our_mut = set([p[0] for p in prosit_mut_pair])
ft_our_mut = set([p[0] for p in ft_mut_pair])


def LSG(s1, s2):
    return s1 - s2, s1.intersection(s2), s2 - s1


def save_LSG(pep1, pep2, reverse_dict: Dict[str, List], file_prefix, second=None):
    L, S, G = LSG(pep1, pep2)
    for set_name, pep_set in zip(['loss', 'share', 'gain'], [L, S, G]):
        if len(pep_set) == 0:
            print(file_prefix, set_name, "pass")
            continue
        dict2df = {"Mutation Protein": [], "Mutation Type": [], "Sequence": []}
        type2pep_dict = defaultdict(list)
        for pep in sorted(pep_set):
            for pack in reverse_dict[pep]:
                type2pep_dict["#".join(pack)].append(pep)
        if len(type2pep_dict) == 0:
            for pep in sorted(pep_set):
                for pack in second[pep]:
                    type2pep_dict["#".join(pack)].append(pep)
        for pack in sorted(type2pep_dict.keys()):
            dict2df['Sequence'].append(";".join(type2pep_dict[pack]))
            dict2df['Mutation Type'].append(pack.split('#')[0])
            dict2df['Mutation Protein'].append(pack.split('#')[1])
        df = pd.DataFrame(dict2df)
        df = df.sort_values(
            by=['Mutation Protein', 'Mutation Type'], ignore_index=True)
        df = df[['Mutation Protein', 'Mutation Type', 'Sequence']]
        df.to_csv(f"data/{file_prefix}-{set_name}.csv")
        with open(f"data/{file_prefix}-{set_name}.pep", 'w') as f:
            for pep in sorted(pep_set):
                f.write(pep + '\n')


all_pep_supp_missense = set(
    [p for p in all_pep_supp if p in supp_mut_found_dict_missense])
print("Prosit Mut Pep", [len(p) for p in LSG(all_pep_supp, prosit_our_mut)])
print("Ft Mut Pep", [len(p) for p in LSG(all_pep_supp, ft_our_mut)])
print("(missense)Prosit Mut Pep", [len(p)
      for p in LSG(all_pep_supp_missense, prosit_our_mut)])
print("(missense)FT Mut Pep", [len(p)
      for p in LSG(all_pep_supp_missense, ft_our_mut)])


# ------------------------

save_LSG(all_pep_supp, prosit_our_mut,
         reverse_prosit_dict_list, "fig3_mut_prosit")
save_LSG(all_pep_supp, ft_our_mut,
         reverse_ft_dict_list, "fig3_mut_ft", second=reverse_prosit_dict_list)
# ---------------------------------------------------
all_pep_supp_types = set(all_supp_mut_proteins)
all_pep_supp_types_missense = set(missense_all_supp_mut_proteins)

prosit_our_mut_types = []
for p in prosit_our_mut:
    prosit_our_mut_types.extend([v[1] for v in reverse_prosit_dict_list[p]])
ft_our_mut_types = []
for p in ft_our_mut:
    ft_our_mut_types.extend([v[1] for v in reverse_ft_dict_list[p]])
prosit_our_mut_types = set(prosit_our_mut_types)
ft_our_mut_types = set(ft_our_mut_types)

print("Prosit Mut protein:", [len(p)
      for p in LSG(all_pep_supp_types, prosit_our_mut_types)])
print("ft Mut protein:", [len(p)
      for p in LSG(all_pep_supp_types, ft_our_mut_types)])

# print("(missense)Prosit Mut protein:", [len(p)
#       for p in LSG(all_pep_supp_types_missense, prosit_our_mut_types)])
# print("(missense)ft Mut protein:", [len(p)
#       for p in LSG(all_pep_supp_types_missense, ft_our_mut_types)])
# -----------------------------------------------
all_pep_supp_types = set(all_supp_mut_types)
all_pep_supp_types_missense = set(missense_all_supp_mut_types)

prosit_our_mut_types = []
for p in prosit_our_mut:
    prosit_our_mut_types.extend([v[0] for v in reverse_prosit_dict_list[p]])
ft_our_mut_types = []
for p in ft_our_mut:
    ft_our_mut_types.extend([v[0] for v in reverse_ft_dict_list[p]])
prosit_our_mut_types = set(prosit_our_mut_types)
ft_our_mut_types = set(ft_our_mut_types)
# for thistype in (all_pep_supp_types - prosit_our_mut_types):
#     for pack in supp_mut_found_pair:
#         if thistype == pack[1]:
#             print(thistype, pack[2])
#             print(pack[0])
#             print("---------------------------------")
print("Prosit Mut types:", [len(p)
      for p in LSG(all_pep_supp_types, prosit_our_mut_types)])
print("ft Mut types:", [len(p)
      for p in LSG(all_pep_supp_types, ft_our_mut_types)])
# ------------------------------
# for p_p in prosit_our_loss:
#     found = False
#     p_p = p_p
#     p_p = p_p.lower()
#     for pack in MUT_ALL:
#         if p_p in pack[1].lower():
#             found = True
#             print(p_p)
#             print(pack[0])
#             print("------------------")
#             break
#     if not found:
#         print("Not found", p_p)


# ----------------------------
print(len(all_pep_supp), not_f)
print([len(p) for p in LSG(all_pep_supp, prosit_pep)])
print([len(p) for p in LSG(all_pep_supp, ft_pep)])

LOST = LSG(all_pep_supp, prosit_pep)[0]

MUT_LOST = LSG(all_pep_supp, prosit_our_mut)[0]
print([len(p) for p in LSG(LOST, MUT_LOST)])

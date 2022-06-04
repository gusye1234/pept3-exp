import pandas as pd
import sys
import re
from os.path import join
from tqdm import tqdm


def read_peptide(check_dir, threshold=0.01):
    df = pd.read_csv(join(check_dir, "prosit_target.peptides"), sep='\t')
    f_p_peptides = df[df['q-value'] < threshold]['peptide'].apply(
        lambda x: x.strip("_").strip(".")).unique()
    f_p_peptides = [p.upper() for p in f_p_peptides]
    return set(f_p_peptides)


def if_mutation(d):
    return "|" in d


def read_mut(mut_fasta):
    mut_pair = []
    with open(mut_fasta) as f:
        for l in f:
            l = l.strip()
            if l.startswith(">"):
                mut_pair.append([l])
            else:
                mut_pair[-1].append(l)
    return mut_pair


def filter_prefixs(data, prefixs):
    new_data = []

    def check(f: str):
        for p in prefixs:
            if f.startswith(p):
                return True
        return False

    for pack in data:
        mut_flag = pack[0]
        for f in mut_flag.split('|'):
            if check(f):
                new_data.append(pack)
                break
    return new_data


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
    print(f"{len(data)}, {can_not_parse}, {out_bound}")
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


def check_missense_overlap(target, pep, loc):
    possible_left = max(0, loc - len(pep))
    possible_right = min(len(target), loc + len(pep) - 1)
    possible_area = target[possible_left:possible_right]
    return (pep.lower() in possible_area.lower())


def check_frameshift_overlap(target, pep, loc):
    possible_left = max(0, loc - len(pep))
    possible_area = target[possible_left:]
    return (pep.lower() in possible_area.lower())


def if_frameshift(data):
    return "frameshift_variant" in data


def if_missense(data):
    return "missense" in data


def vis_match(pep, mut_loc, mut_core, f):
    starts = re.search(pep.lower(), mut_core[1].lower()).start()
    f.write('\n'.join([
        mut_core[0].split("|")[5],
        " " * (mut_loc - 1) + "|" + str(mut_loc),
        mut_core[1],
        " " * starts + pep,
        '------------------'
    ]) + "\n")
    f.flush()


CHECK = "/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/forPride/rescoring_for_paper_2/percolator"
misfilename = "data/fig3_mutation_our_prosit.txt"
# CHECK = "/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/percolator_hdf5_0.1"
# misfilename = "data/fig3_mutation_our_ft.txt"
FASTA = "/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/Mel15OP1_mut_only.fasta"
print(f"Save to {misfilename}")
PEPTIDE = read_peptide(CHECK, threshold=0.03)
# -----------------------------------------
protein_groups = pd.read_csv(
    '/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/forPride/txt/peptides.txt', sep='\t')
protein_groups = protein_groups[protein_groups['Reverse'] != '+']
protein_groups = protein_groups[protein_groups['Proteins'].apply(
    lambda x: all([if_mutation(p) for p in x.split(";")]))]
LEGAL_PEPS = set(protein_groups['Sequence'])
print(len(LEGAL_PEPS))
PEPTIDE = PEPTIDE.intersection(LEGAL_PEPS)
# ----------------------------------------
print(f"total {len(PEPTIDE)} peptides to search")
MUT_DATA = filter_prefixs(
    read_mut(FASTA), ["missense", "frameshift_variant", "non_coding_transcript_exon_variant"])
MUT_LOC, MUT_DATA = extract_loc(MUT_DATA)
print(f"Total {len(MUT_DATA)} to check")
mut_pep_count = []

P_bar = tqdm(PEPTIDE)
with open(f"log_{misfilename}", 'w') as f:
    for pep in P_bar:
        P_bar.set_description(f"Found {len(mut_pep_count)}")
        P_bar.refresh()
    # for pep in PEPTIDE:
        for mut_loc, mut_core in zip(MUT_LOC, MUT_DATA):
            core_pep = mut_core[1]
            if if_frameshift(mut_core[0]) and check_frameshift_overlap(core_pep, pep, mut_loc):
                vis_match(pep, mut_loc, mut_core, f)
                mut_pep_count.append((pep, mut_core[0], mut_core[1]))
            elif check_missense_overlap(core_pep, pep, mut_loc):
                vis_match(pep, mut_loc, mut_core, f)
                mut_pep_count.append((pep, mut_core[0], mut_core[1]))
with open(misfilename, 'w') as f:
    for p in mut_pep_count:
        f.write("\n".join(p))
        f.write("\n---------------------------\n\n")

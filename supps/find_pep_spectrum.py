import torch
import os
import pandas as pd
import math
import sys
sys.path.append("..")
from pept3 import model
from pept3.tools import generate_from_msms, get_irt_all

ft_peptide_file = "/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/3fold_hdf5_0.1/prosit_target.peptides"
def peptides2dict(tabfile):
    df = pd.read_csv(tabfile, sep='\t')
    assert len(df['peptide']) == len(df['peptide'].unique())
    f_p_peptides = df['peptide'].apply(
        lambda x: x.strip("_").strip(".")).unique()
    f_p_dict = {
        k.strip("_").strip(".").upper(): (v, z) for k, v, z in zip(df['peptide'], df['q-value'], df["PSMId"])
    }
    return f_p_peptides, f_p_dict

_, fp_dict = peptides2dict(ft_peptide_file)

def star_move(seq):
    peps = seq.replace("*", "")
    return peps

which = "Mel15"
msms_file = "/data2/yejb/prosit/boosting_bp/txt_Mel15/msms.txt"
raw_dir = "/data2/yejb/prosit/boosting_bp/annotation_Mel15"

table = pd.read_csv("./data/mel15_ft_added.csv")

table.rename(columns={"Experiments": "Patient",
              "Mutation Type": "Genomic alterations",
              "Finetuned Prosit peptide Q-value":"Q-value"}, inplace=True)
table['Patient'] = table["Patient"].apply(lambda x: x.split("_")[0])
seq_loc = []
for seq, loc in zip(table["Neo-Epitope"], table["Position"]):
    seq_loc.append(star_move(seq))
table["Sequence"] = seq_loc

alleles = []
affinity = []
allele_name = ['HLA-A03:01',
               'HLA-A68:01',
               'HLA-B27:05',
               'HLA-B35:03']
columns = [f"NetMHCpan prediction \n{allele}\n(score;%rank;binding_level)"
           for allele in allele_name]
for p in zip(*(table[c] for c in columns)):
    p = [(l, n) for l, n in zip(p, allele_name)]
    p = sorted(p, key=lambda x: float(x[0].split(";")[1]) if isinstance(x[0], str) else float("inf"))    
    alleles.append(p[0][1])
    affinity.append(";".join(p[0][0].split(";")[1:]).replace("<=", ""))
table["Allele"] = alleles
table["%rank;bindLevel"] = affinity

new_tab = table[["Patient", "Gene name", "Transcript ID", "Genomic alterations", "Amino-acid change", "Sequence", "Position", "Allele", "%rank;bindLevel", "Q-value"]]
new_tab = new_tab.sort_values(by="Q-value", key=lambda x: x.apply(lambda y: float(y)))
new_tab = new_tab.iloc[:10]

seqs = new_tab['Sequence'].to_list()
psmids = []
for seq in seqs:
    seq = seq.upper()
    assert fp_dict[seq][0] < 0.03 # our threshold
    psmids.append(fp_dict[seq][1])

print(psmids)

def split_name(pid):
    fields = pid.split("-")
    charge = fields[-1]
    protein = fields[-2]
    scan_number = fields[-3]
    rawfile = "-".join(fields[:-3])
    return rawfile, scan_number, protein, charge

def locate_maxquant(psmids, msms_file):
    msms_data = pd.read_csv(msms_file, sep='\t')
    datas = []
    old_format = [] 
    for pid in psmids:
        rawfile, scan_number, protein, charge = split_name(pid)
        inter = msms_data[msms_data["Raw file"] == rawfile]
        inter = inter[inter["Scan number"] == int(scan_number)]
        assert len(inter) == 1
        datas.append(inter)
        old_format.append(inter.iloc[0].to_list())
    return pd.concat(datas), old_format

mq_result, old_mq_format = locate_maxquant(psmids, msms_file)

run_model = model.PrositIRT()
run_model.load_state_dict(torch.load(
    f"../checkpoints/irt/best_valid_irt_{run_model.comment()}-1024.pth", map_location="cpu"))
prosit_irt = run_model.eval()
data_for_rt = generate_from_msms(old_mq_format, list(mq_result.keys()))
irts = get_irt_all(prosit_irt, data_for_rt)
new_tab["Prosit_IRT"] = irts
new_tab = pd.concat([new_tab.reset_index(), mq_result.reset_index()], axis=1)

def locate_mass(psmids, raw_dir):
    raw_files = os.listdir(raw_dir)
    raw_files = [p for p in raw_files if p.endswith(".csv")]
    datas = []
    for pid in psmids:               
        rawfile, scan_number, protein, charge = split_name(pid)
        rawfile = rawfile+".csv"
        assert rawfile in raw_files
        raw_df = pd.read_csv(os.path.join(raw_dir, rawfile))
        raw_row = raw_df[raw_df['scan_number'] == int(scan_number)]
        assert len(raw_row) == 1
        datas.append(raw_row)
    return pd.concat(datas).reset_index()
raw_result = locate_mass(psmids, raw_dir)
new_tab = pd.concat([new_tab, raw_result], axis=1)
new_tab.to_csv("./data/exp_peps.csv", index=False, sep='\t')
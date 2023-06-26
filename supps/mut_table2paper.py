import pandas as pd
import math

def star_latex(seq):
    peps = seq.split("*")
    assert len(peps) == 3
    new_peps = "".join([
        peps[0], "\textit{\textbf{",
        peps[1], "}}", peps[2]
    ])
    return new_peps


table = pd.read_csv("./data/mel15_ft_added.csv")


table.rename(columns={"Experiments": "Patient",
              "Mutation Type": "Genomic alterations",
              "Finetuned Prosit peptide Q-value":"Q-value"}, inplace=True)

table['Patient'] = table["Patient"].apply(lambda x: x.split("_")[0])

seq_loc = []
for seq, loc in zip(table["Neo-Epitope"], table["Position"]):
    seq_loc.append(star_latex(seq)+loc)
table["Sequence(Position)"] = seq_loc

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

new_tab = table[["Patient", "Gene name", "Transcript ID", "Genomic alterations", "Amino-acid change", "Sequence(Position)", "Allele", "%rank;bindLevel", "Q-value"]]
# from IPython import embed
# embed()
# new_tab = new_tab.sort_values(by="%rank;bindLevel", key=lambda x: x.apply(lambda y: float(y.split(";")[0])))
new_tab = new_tab.sort_values(by="Q-value", key=lambda x: x.apply(lambda y: float(y)))
new_tab["Genomic alterations"] = new_tab["Genomic alterations"].apply(lambda x: x.replace("_variant", "").replace("_", "-"))
new_tab["Amino-acid change"] = new_tab["Amino-acid change"].apply(lambda x: x.replace(">", "$>$"))
print(new_tab.to_latex(index=False, escape=False), file=open("see.txt", 'w'))
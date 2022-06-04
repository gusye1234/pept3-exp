import os
import pandas as pd


def file2allele(f):
    f = os.path.basename(os.path.splitext(f)[0])
    return f"HLA-{f[0]}*{f[1:3]}:{f[3:]}"


def read_hla_seq(f):
    df = pd.read_csv(f)
    result = {}
    for _, row in df.iterrows():
        result[row['HLA']] = row['HLA_sequence']
    return result


def add_prefix(f, name):
    dirname = os.path.dirname(f)
    fname = os.path.basename(f)
    return os.path.join(dirname, name + fname)


check_dir = "/data1/yejb/prosit/figure3/supply_origin/ft_nonft"
# check_dir = "/data1/yejb/prosit/figure3/supply_origin/prosit_add"
# check_dir = "/data1/yejb/prosit/figure3/supply_origin/finetuned_add"
out_dir = os.path.join(check_dir, "transphla")
print(f"{check_dir} \n -> {out_dir}")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

all_peps = sorted([f for f in os.listdir(
    "/data1/yejb/prosit/figure3/supply_origin/prosit_add") if f[0] in "ABCG"])
all_peps = [os.path.join(check_dir, f) for f in all_peps]
all_hlas = read_hla_seq("./common_hla_sequence.csv")

print(len(all_hlas), len(all_peps))
for add_name in ['sub-', 'add-', 'share-']:
    for pep in all_peps:
        allele_name = file2allele(pep)
        out_name = os.path.join(
            out_dir, f"peptide-{add_name}{allele_name}.fasta")
        hla_out_name = os.path.join(
            out_dir, f"hla-{add_name}{allele_name}.fasta")
        part_pep = add_prefix(pep, add_name)
        if not allele_name in all_hlas:
            print(allele_name, "not found")
            continue
        with open(part_pep) as f:
            read_pep = [l.strip() for l in f]
        with open(out_name, 'w') as f:
            for pep in read_pep:
                f.write(f">{pep}\n")
                f.write(f"{pep}\n")
        with open(hla_out_name, 'w') as f:
            hla_seq = all_hlas[allele_name]
            for _ in read_pep:
                f.write(f">{allele_name}\n")
                f.write(f"{hla_seq}\n")

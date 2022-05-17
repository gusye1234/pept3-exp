import os
from tqdm import tqdm


def file2allele(f):
    f = os.path.basename(os.path.splitext(f)[0])
    return f"HLA-{f[:3]}:{f[3:]}"


def add_prefix(f, name):
    dirname = os.path.dirname(f)
    fname = os.path.basename(f)
    return os.path.join(dirname, name + fname)


# check_dir = "/data1/yejb/prosit/figure3/supply_origin/ft_nonft"
check_dir = "/data1/yejb/prosit/figure3/supply_origin/prosit_add"
out_dir = os.path.join(check_dir, "netMHCpan_out")
print(f"{check_dir} \n -> {out_dir}")
all_peps = sorted([f for f in os.listdir(
    "/data1/yejb/prosit/figure3/supply_origin/prosit_add") if f[0] in "ABCG"])
# print(all_peps)
all_peps = [os.path.join(check_dir, f) for f in all_peps]

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for add_name in ['sub-', 'add-', 'share-']:
    # if status != 0:
    #     print(f"Fail {allele_name} with {status}")
    #     exit()
    for pep in tqdm(all_peps):
        allele_name = file2allele(pep)
        out_name = os.path.join(out_dir, f"{allele_name}.out")
        part_pep = add_prefix(pep, add_name)
        out_name = os.path.join(out_dir, f"{add_name}{allele_name}.out")
        status = os.system(f"cd /home/yejb/code_repo/netMHCpan-4.1 && \
                       ./netMHCpan -a {allele_name} -p {part_pep} \
                       > {out_name}")
        if status != 0:
            print(f"Fail {add_name}{allele_name} with {status}")
            exit()

import os
from tqdm import tqdm


check_dirs = ["/home/yejb/code_repo/MST/figs/data/fig3_mut_ft-gain.pep",
              "/home/yejb/code_repo/MST/figs/data/fig3_mut_ft-loss.pep",
              "/home/yejb/code_repo/MST/figs/data/fig3_mut_ft-share.pep"]
check_alleles = ['HLA-A03:01',
                 'HLA-A68:01',
                 'HLA-B27:05',
                 'HLA-B35:03']

OUT = "/home/yejb/code_repo/MST/figs/data/netmhcpan"
if not os.path.exists(OUT):
    os.mkdir(OUT)
for check_dir in check_dirs:
    for allele in check_alleles:
        out_dir = check_dir.split('.')[0] + f"-{allele}-netmhcpan.out"
        out_dir = os.path.join(OUT, os.path.split(out_dir)[1])
        print(f"{check_dir} \n -> {out_dir}")
        status = os.system(f"cd /home/yejb/code_repo/netMHCpan-4.1 && \
                            ./netMHCpan -a {allele} -p {check_dir} \
                            > {out_dir}")
        if status != 0:
            print(f"Fail {check_dir} {allele} with {status}")
            exit()

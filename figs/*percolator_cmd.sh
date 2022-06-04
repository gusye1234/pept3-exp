#!/bin/bash
# save_tab=/data1/yejb/prosit/figure3/forPRIDE/IAA_noIAA
save_tab=/data1/yejb/prosit/figure3/percolator/IAA_noIAA
name=prosit

echo "Save results to ${save_tab}/${name}*"
percolator -v 0 --weights ${save_tab}/${name}_weights.csv \
                 -Y --testFDR 0.01 --trainFDR 0.01 \
                --results-psms ${save_tab}/${name}_target.psms \
                --results-peptides ${save_tab}/${name}_target.peptides \
                --decoy-results-psms ${save_tab}/${name}_decoy.psms \
                --decoy-results-peptides ${save_tab}/${name}_decoy.peptides \
                ${save_tab}/all_${name}_0.01.tab > ${save_tab}/${name}.log
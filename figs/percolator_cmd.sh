#!/bin/bash

place="/data/prosit/figs/figure6/sprot_all/percolator/try/autos"
fdr=0.01

name="spectral_all"
echo "-------------------------------------------------------------"
echo "===${name}==="
percolator --weights ${place}/${name}_weights.csv \
--post-processing-tdc --only-psms --testFDR ${fdr} \
--results-psms ${place}/${name}_target.psms \
--decoy-results-psms ${place}/${name}_decoy.psms \
${place}/${name}.tab
echo " "

# name="sa"
# echo "-------------------------------------------------------------"
# echo "===${name}==="
# percolator --weights ${place}/${name}_weights.csv \
# --post-processing-tdc --only-psms --testFDR ${fdr} \
# --results-psms ${place}/${name}_target.psms \
# --decoy-results-psms ${place}/${name}_decoy.psms \
# ${place}/${name}.tab
# echo " "

# name="andromeda"
# echo "-------------------------------------------------------------"
# echo "===${name}==="
# percolator --weights ${place}/${name}_weights.csv \
# --post-processing-tdc --only-psms --testFDR ${fdr} \
# --results-psms ${place}/${name}_target.psms \
# --decoy-results-psms ${place}/${name}_decoy.psms \
# ${place}/${name}.tab
# echo " "
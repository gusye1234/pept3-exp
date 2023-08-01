import os
import sys
from time import time
import pandas as pd
import pickle
from ms2pip.ms2pipC import MS2PIP
from converter import main_convert
from tools import ms2pip_result_convert, fdr_test

sys.path.append("../")
import pept3


Test=False

params = {
     "ms2pip": {
         "ptm": [
             "Oxidation,15.994915,opt,M",
             "Carbamidomethyl,57.021464,opt,C",
             "Acetyl,42.010565,opt,N-term"
         ],
         "frag_method": "HCD2021",
         "frag_error": 0.02, # not used
         "out": "csv",
         "sptm": [], "gptm": [],
     }
 }

for which in ["trypsin", 'chymo', "lysc", "gluc"]:
    print("Running", which)
    save_tab = f"/data2/yejb/prosit/figs/fig235/{which}/ms2pip"
    msms_dir = f"/data2/yejb/prosit/figs/fig235/{which}/maxquant/combined/txt"
    raw_dir = f"/data2/yejb/prosit/figs/fig235/{which}/raw"
    if not os.path.exists(save_tab):
        os.mkdir(save_tab)
    # peprec_file = main_convert(msms_dir, save_tab, overwrite=True)
    # ms2pip = MS2PIP(peprec_file, 
    #                 params=params, 
    #                 return_results=False, 
    #                 num_cpu=4,
    #                 output_filename=os.path.join(save_tab, "spectrum.predict"),
    #                 add_retention_time=True)
    # ms2pip.run()

    totest = ["andromeda", "sa", "prosit_combined", "prosit_best"]
    if Test:
        pkl_file = os.path.join(save_tab, "lookup_ms2pip.pkl")
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                lookup_dict = pickle.load(f)
        else:
            lookup_dict = ms2pip_result_convert(save_tab)
            with open(pkl_file, 'wb') as f:
                pickle.dump(lookup_dict, f)
        
        fdr_test(lookup_dict, 
                os.path.join(msms_dir, "msms.txt"),
                raw_dir,
                save_tab,
                need_irt=True,
                totest=totest,
                need_all=False
                )
    
    print(" start percolator... ")
    fdr_threshold = 0.1
    show_fdr=[0.1, 0.01, 0.001, 0.0001]
    record = {}
    record['fdrs'] = [100 * i for i in show_fdr]
    for name in totest:
        start = time()
        if Test:
            os.system(f"percolator -v 0 --weights {save_tab}/{name}_weights.csv \
                    --post-processing-tdc --only-psms --testFDR {fdr_threshold} \
                    --results-psms {save_tab}/{name}_target.psms \
                    --decoy-results-psms {save_tab}/{name}_decoy.psms \
                    {save_tab}/{name}.tab")
        target_tab = pd.read_csv(os.path.join(
            save_tab, f"{name}_target.psms"), sep='\t')
        record[name] = []
        for fdr in show_fdr:
            record[name].append((target_tab['q-value'] < fdr).sum())
        print(f"{name}:{time()-start:.1f}", end='-')
        
    print(pd.DataFrame(record))
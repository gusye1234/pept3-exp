import os
from ms2pip.ms2pipC import MS2PIP
from converter import main_convert

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
    if not os.path.exists(save_tab):
        os.mkdir(save_tab)
    peprec_file = main_convert(msms_dir, save_tab)
    ms2pip = MS2PIP(peprec_file, 
                    params=params, 
                    return_results=False, 
                    num_cpu=4,
                    output_filename=os.path.join(save_tab, "spectrum.predict"))
    ms2pip.run()
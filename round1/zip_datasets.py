import os
import zipfile
import pandas as pd
from tqdm import tqdm

def filter_files(root, file,
                 root_name = ["no_finetuned_3fold", "finetuned_3fold_0.1"],
                 file_name = ["fixed_features.tab"]):
    if file in file_name:
        return True
    if any([name in root for name in root_name]):
        return True
    return False

def zipdir(path, ziph,
           root_name = ["no_finetuned_3fold", "finetuned_3fold_0.1"],
           file_name = ["fixed_features.tab"]):
    # ziph is zipfile handle
    for root, dirs, files in tqdm(os.walk(path)):
        for file in files:
            if filter_files(root, file,
                            root_name=root_name,
                            file_name=file_name):        
                # print(os.path.basename(root), file)
                print(root, file)
                ziph.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), 
                                           os.path.join(path, '..')))

# for which in ["trypsin", 'chymo', "lysc", "gluc"]:
#     print(which)
#     # for which in ["trypsin", 'lysc']:
#     save_tab = f"/data2/yejb/prosit/figs/fig235/{which}"
#     with zipfile.ZipFile(f"/data2/yejb/prosit/figs/fig235/bekker-{which}.zip", 'w') as zipf:
#         zipdir(save_tab, zipf)


# for which in ["sprot_human", "IGC", "sprot_all", "sprot_bacteria_human"]:
#     print("-------------------------------")
#     print(which)
#     root_dir = f"/data2/yejb/prosit/figs/figure6/{which}"
    
#     with zipfile.ZipFile(f"/data2/yejb/prosit/figs/figure6/meta-{which}.zip", 'w') as zipf:
#         zipdir(root_dir, zipf)    

root_dir = f"/data2/yejb/prosit/figs/figure5/"
with zipfile.ZipFile(f"/data2/yejb/prosit/figs/figure5/davis.zip", 'w') as zipf:
    zipdir(root_dir, zipf)    
# hla_mel = pd.read_csv("../figs/data/HLA_Mel.csv")
# hla_mel = hla_mel[hla_mel['Experiment'].apply(lambda x: x.endswith("HLA-I"))]
# Mels = hla_mel['Experiment'].unique()
# root_dir = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/"

# all_mels = list(Mels)+[n+"_ori" for n in Mels]
# with zipfile.ZipFile(f"/data/yejb/prosit/figs/boosting/figs/HLA_Mels.zip", 'w') as zipf:
#     zipdir(root_dir, zipf,
#            root_name=['prosit_hcd',
#                       'prosit_l1'],
#            file_name=[
#                'fixed_features.tab',
#            ])    
    
# root_dir = f"/data/yejb/prosit/figs/boosting/figs/Figure_5_Mel15/"
# with zipfile.ZipFile(f"/data/yejb/prosit/figs/boosting/figs/HLA_Mel15.zip", 'w') as zipf:
#     zipdir(root_dir, zipf,
#            root_name=['3fold_hdf5_0.1'],
#            file_name=[
#                'fixed_features.tab',
#                'Mel15OP1_Ensemble.fasta',
#                'Mel15OP1_mut_only.fasta'
#            ]
#            )    
    
    
# # root_dir = f"/data1/yejb/prosit/figure3/"
# # with zipfile.ZipFile(f"/data1/yejb/prosit/figure3/HLA_Allele.zip", 'w') as zipf:
# #     zipdir(root_dir, zipf,
# #            root_name=['prosit_hcd/3fold_hdf5_allele_0.1',
# #                       'prosit_l1/3fold_hdf5_allele_0.1'],
# #            file_name=[
# #                'fixed_features.tab',
# #                'Mel15OP1_Ensemble.fasta',
# #                'Mel15OP1_mut_only.fasta'
# #            ]
# #            )    
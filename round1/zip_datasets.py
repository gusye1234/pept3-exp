import os
import zipfile
from tqdm import tqdm

def filter_files(root, file):
    if file in ["fixed_features.tab"]:
        return True
    root = os.path.basename(root)
    if root in ["no_finetuned_3fold", "finetuned_3fold_0.1"]:
        return True
    return False

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in tqdm(os.walk(path)):
        for file in files:
            if filter_files(root, file):        
                # print(os.path.basename(root), file)
                ziph.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), 
                                           os.path.join(path, '..')))

for which in ["trypsin", 'chymo', "lysc", "gluc"]:
    print(which)
    # for which in ["trypsin", 'lysc']:
    save_tab = f"/data2/yejb/prosit/figs/fig235/{which}"
    with zipfile.ZipFile(f"/data2/yejb/prosit/figs/fig235/bekker-{which}.zip", 'w') as zipf:
        zipdir(save_tab, zipf)


for which in ["sprot_human", "IGC", "sprot_all", "sprot_bacteria_human"]:
    print("-------------------------------")
    print(which)
    root_dir = f"/data2/yejb/prosit/figs/figure6/{which}"
    
    with zipfile.ZipFile(f"/data2/yejb/prosit/figs/figure6/meta-{which}.zip", 'w') as zipf:
        zipdir(root_dir, zipf)    
    

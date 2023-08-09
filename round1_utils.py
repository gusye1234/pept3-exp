import shutil
from os import path
from tqdm import tqdm
import pandas as pd
from pept3.helper import read_m_r_ions, read_name


def save_triple(m_r, m_i_delta, m_i, save2):
    with open(save2, 'w') as f:
        for m1, ion_delta, ion in zip(m_r, m_i_delta, m_i):
            mr_line = str(m1)
            delta_line = str(ion_delta)
            ion_line = str(ion)
            f.write(
                "\t".join([mr_line, delta_line, ion_line]) + "\n")

def merge_decoy(original_msms_tab, 
                original_feat_tab,
                from_msms_tab, 
                from_feat_tab,
                merge_tab,
                sample_num=20000):
    ori_feat = path.join(original_feat_tab, "fixed_features.tab")
    ori_msms = path.join(original_msms_tab, "msms.txt")
    msms_name = read_name(ori_msms)
    ori_ions = path.join(original_msms_tab, "msms_ions.txt")
    
    from_feat = path.join(from_feat_tab, "fixed_features.tab")    
    from_msms = path.join(from_msms_tab, "msms.txt")
    from_ions = path.join(from_msms_tab, "msms_ions.txt")
    
    merge_feat = path.join(merge_tab, "fixed_features.tab")    
    merge_msms = path.join(merge_tab, "msms.txt")
    merge_ions = path.join(merge_tab, "msms_ions.txt")
    merge_ids = path.join(merge_tab, "fake_decoys.txt")
    
    if not path.exists(merge_msms):
        shutil.copyfile(ori_msms, merge_msms)
    
    ori_feat_pd = pd.read_csv(ori_feat, sep='\t')
    all_target_peptides = set(ori_feat_pd[ori_feat_pd['Label']==1]['Peptide'])
    
    from_feat_pd = pd.read_csv(from_feat, sep='\t').sample(frac=1).reset_index(drop=True)
    from_feat_pd = from_feat_pd[from_feat_pd['Label']==-1].reset_index(drop=True)
    print(f"Total decoy {len(from_feat_pd)}")
    from_feat_pd = from_feat_pd.iloc[:(sample_num*2)]
    bias = ori_feat_pd['SpecId'].max()
    
    fake_before_ids = set(from_feat_pd['SpecId'].to_list())
    false_target_ids = set(from_feat_pd.iloc[:sample_num]['SpecId'].to_list())
    from_feat_pd['SpecId'] += bias
    fake_ids = set(from_feat_pd['SpecId'].to_list())
    from_feat_pd.loc[:sample_num-1, 'Label'] = 1
    print(f"Adding {len(from_feat_pd)} decoys to {len(ori_feat_pd)} target+decoy")
    merge_feat_pd = pd.concat([ori_feat_pd, from_feat_pd],
                              ignore_index=True,
                              axis=0)
    merge_feat_pd.to_csv(merge_feat, sep='\t', index=False)
    
    m_r, m_i_delta, m_i = read_m_r_ions(from_ions)
    new_mr, new_mi_delta, new_mi = [], [], []
    
    false_target_num, false_decoy_num = 0, 0
    for m1, m2, m3 in tqdm(zip(m_r, m_i_delta, m_i)):
        if int(m1[0][msms_name.index("id")] ) in fake_before_ids:
            assert len(m1[0][msms_name.index("Reverse")].strip())
            if int(m1[0][msms_name.index("id")]) in false_target_ids:
                m1[0][msms_name.index("Reverse")] = ""
                false_target_num += 1
            else:
                false_decoy_num += 1
            m1[0][msms_name.index("id")] = \
            int(m1[0][msms_name.index("id")]) + bias
            new_mr.append(m1)
            new_mi_delta.append(m2)
            new_mi.append(m3)
    print(f"Add {false_target_num} false_target, {false_decoy_num} false decoy")
    del m_r, m_i_delta, m_i
    assert len(new_mr) == 2*sample_num, f"Only {len(new_mr)} is found, expect {sample_num}"
    ori_ions_obj = read_m_r_ions(ori_ions)
    ori_ions_obj = [o+o1 for o, o1 in zip(ori_ions_obj,
                                          [new_mr, new_mi_delta, new_mi])]
    save_triple(*ori_ions_obj, merge_ions)
    
    with open(merge_ids, 'w') as f:
        f.write(f"{from_msms}\t{merge_msms}\n")    
        for i in fake_ids:
            f.write(f"{i-bias}\t{i}\n")    
            
    
    
    
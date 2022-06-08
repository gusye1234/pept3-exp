import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

IAA_dir = "/data1/yejb/prosit/figure3/forPRIDE/IAA"
noIAA_dir = "/data1/yejb/prosit/figure3/forPRIDE/noIAA"
check_dir = "/data1/yejb/prosit/figure3/forPRIDE"

save_dir = os.path.join(check_dir, "Alleles")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
IAA_hdf5_data = os.path.join(IAA_dir, "data.hdf5")
noIAA_hdf5_data = os.path.join(noIAA_dir, "data.hdf5")

IAA_csv = os.path.join(IAA_dir, "percolator/features.csv")
noIAA_csv = os.path.join(noIAA_dir, "percolator/features.csv")

IAA_prosit = os.path.join(IAA_dir, "percolator/prosit.tab")
noIAA_prosit = os.path.join(noIAA_dir, "percolator/prosit.tab")

IAA_maxquant = os.path.join(IAA_dir, "msms.tx")
noIAA_maxquant = os.path.join(noIAA_dir, "msms.txt")

alleles_rawfile = {}
with open("./data/allele_raw.txt") as f:
    for l in f:
        pack = l.strip().split("\t")
        alleles_rawfile[pack[0]] = set(pack[1:])


def load_hdf5(hdf5):
    HDF5 = h5py.File(hdf5, 'r')
    Rawfiles = np.array(HDF5['rawfile'])
    HDF5_dict = {
        key: np.array(HDF5[key])
        for key in HDF5.keys()
    }
    return HDF5, Rawfiles, HDF5_dict


# IAA_HDF5, IAA_Rawfiles, IAA_HDF5_dict = load_hdf5(IAA_hdf5_data)
# noIAA_HDF5, noIAA_Rawfiles, noIAA_HDF5_dict = load_hdf5(noIAA_hdf5_data)

# c_count = 0


# def pick_bool_index(alleles_rawfile, Rawfiles, MEL):
#     pick_index = np.zeros(len(Rawfiles)).astype("bool")
#     for raw in alleles_rawfile[MEL]:
#         pick_index = np.logical_or(
#             pick_index, (Rawfiles == raw.encode()))
#     return pick_index


# for MEL in tqdm(alleles_rawfile.keys()):
#     SAVE = os.path.join(save_dir, MEL)
#     if not os.path.exists(SAVE):
#         os.mkdir(SAVE)
#     IAA_pick_index = pick_bool_index(alleles_rawfile, IAA_Rawfiles, MEL)
#     noIAA_pick_index = pick_bool_index(alleles_rawfile, noIAA_Rawfiles, MEL)

#     # print(f"{MEL} for {np.sum(IAA_pick_index) + np.sum(noIAA_pick_index)}")
#     c_count += np.sum(IAA_pick_index) + np.sum(noIAA_pick_index)
#     with h5py.File(os.path.join(SAVE, "data.hdf5"), 'w') as f:
#         iaa_raw = IAA_HDF5_dict['rawfile'][IAA_pick_index]
#         noiaa_raw = noIAA_HDF5_dict['rawfile'][noIAA_pick_index]
#         raws = np.concatenate([iaa_raw, noiaa_raw], axis=0)

#         iaa_num = IAA_HDF5_dict['scan_number'][IAA_pick_index]
#         noiaa_num = noIAA_HDF5_dict['scan_number'][noIAA_pick_index]
#         scannums = np.concatenate([iaa_num, noiaa_num], axis=0)

#         order_list = [(i, p[0], p[1])
#                       for i, p in enumerate(zip(raws, scannums))]
#         order_list.sort(key=lambda x: (x[1].decode(), x[2]))
#         order = np.array([p[0] for p in order_list]).astype('int')
#         for h5key in IAA_HDF5.keys():
#             keydata_iaa = IAA_HDF5_dict[h5key][IAA_pick_index]
#             keydata_noiaa = noIAA_HDF5_dict[h5key][noIAA_pick_index]
#             keydata = np.concatenate([keydata_iaa, keydata_noiaa])[order]
#             f.create_dataset(h5key, keydata.shape, keydata.dtype, keydata)
# print(f"HDF5: {len(IAA_Rawfiles) + len(noIAA_Rawfiles)} - {c_count}")

# IAA_FEAT = pd.read_csv(IAA_csv)
# noIAA_FEAT = pd.read_csv(noIAA_csv)

# iaa_index_tab = IAA_FEAT.set_index('raw_file')
# noiaa_index_tab = noIAA_FEAT.set_index('raw_file')
# c_count = 0
# for MEL in tqdm(alleles_rawfile.keys()):
#     SAVE = os.path.join(save_dir, MEL)
#     SAVE = os.path.join(SAVE, "percolator")
#     if not os.path.exists(SAVE):
#         os.mkdir(SAVE)
#     df = None
#     for raw in alleles_rawfile[MEL]:
#         iaa_target = iaa_index_tab.filter(like=raw, axis=0)
#         noiaa_target = noiaa_index_tab.filter(like=raw, axis=0)
#         target = pd.concat([iaa_target, noiaa_target], ignore_index=False)
#         if df is None:
#             df = target
#         else:
#             df = pd.concat([df, target], ignore_index=False)
#     # print(f"{MEL} for {len(df)}")
#     before_keys = list(df.keys())
#     df.reset_index(inplace=True)
#     df = df.drop_duplicates(subset=['raw_file', "scan_number"], keep='last')
#     c_count += len(df)
#     df = df.sort_values(by=['raw_file', "scan_number"])
#     df.to_csv(os.path.join(SAVE, 'features.csv'), index=False)
# print(f"Features: {len(IAA_FEAT) + len(noIAA_FEAT)} - {c_count}")

# IAA_PROSIT = pd.read_csv(IAA_prosit, sep='\t')
# noIAA_PROSIT = pd.read_csv(noIAA_prosit, sep='\t')

# iaa_index_tab = IAA_PROSIT.set_index('SpecId')
# noiaa_index_tab = noIAA_PROSIT.set_index('SpecId')
# c_count = 0
# for MEL in tqdm(alleles_rawfile.keys()):
#     SAVE = os.path.join(save_dir, MEL)
#     SAVE = os.path.join(SAVE, "percolator")
#     if not os.path.exists(SAVE):
#         os.mkdir(SAVE)
#     df = None
#     for raw in alleles_rawfile[MEL]:
#         iaa_target = iaa_index_tab.filter(like=raw, axis=0)
#         noiaa_target = noiaa_index_tab.filter(like=raw, axis=0)
#         target = pd.concat([iaa_target, noiaa_target], ignore_index=False)
#         if df is None:
#             df = target
#         else:
#             df = pd.concat([df, target], ignore_index=False)
#     # print(f"{MEL} for {len(df)}")
#     before_keys = list(df.keys())
#     df.reset_index(inplace=True)
#     df = df.drop_duplicates(subset=['SpecId', "ScanNr"], keep='last')
#     df = df.sort_values(by=['SpecId', "ScanNr"])
#     c_count += len(df)
#     df.to_csv(os.path.join(SAVE, 'prosit.tab'), sep='\t', index=False)
# print(f"Prosit: {len(IAA_PROSIT) + len(noIAA_PROSIT)} - {c_count}")


IAA_MAXQUANT = pd.read_csv(IAA_maxquant, sep='\t')
noIAA_MAXQUANT = pd.read_csv(noIAA_maxquant, sep='\t')

iaa_index_tab = IAA_MAXQUANT.set_index('Raw File')
noiaa_index_tab = noIAA_MAXQUANT.set_index('Raw File')
c_count = 0
for MEL in tqdm(alleles_rawfile.keys()):
    SAVE = os.path.join(save_dir, MEL)
    if not os.path.exists(SAVE):
        os.mkdir(SAVE)
    df = None
    for raw in alleles_rawfile[MEL]:
        iaa_target = iaa_index_tab.filter(like=raw, axis=0)
        noiaa_target = noiaa_index_tab.filter(like=raw, axis=0)
        target = pd.concat([iaa_target, noiaa_target], ignore_index=False)
        if df is None:
            df = target
        else:
            df = pd.concat([df, target], ignore_index=False)
    # print(f"{MEL} for {len(df)}")
    before_keys = list(df.keys())
    df.reset_index(inplace=True)
    c_count += len(df)
    df.to_csv(os.path.join(SAVE, 'msms.txt'), sep='\t', index=False)
print(f"Prosit: {len(IAA_MAXQUANT) + len(noIAA_MAXQUANT)} - {c_count}")

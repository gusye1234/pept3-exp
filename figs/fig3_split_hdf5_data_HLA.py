import os
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict

check_dir = "/data/yejb/prosit/figs/boosting/figs/Figure_5_HLA_1/forPride/rescoring_for_paper_2"

save_dir = os.path.join(check_dir, "Mels")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
hdf5_data = os.path.join(check_dir, "data.hdf5")
csv_data = os.path.join(check_dir, "percolator/features.csv")
prosit_data = os.path.join(check_dir, "percolator/prosit.tab")

hla_mel = pd.read_csv("./HLA_Mel.csv")
hla_mel = hla_mel[hla_mel['Experiment'].apply(lambda x: x.endswith("HLA-I"))]
naming_map = {
    hla_mel.iloc[row]['Raw file']: hla_mel.iloc[row]['Experiment']
    for row in range(len(hla_mel))
}
mel2raw = defaultdict(list)

for rawf, mel in naming_map.items():
    mel2raw[mel].append(rawf)

# HDF5 = h5py.File(hdf5_data, 'r')
# Rawfiles = np.array(HDF5['rawfile'])
# HDF5_dict = {
#     key: np.array(HDF5[key])
#     for key in HDF5.keys()
# }
# c_count = 0
# for MEL in mel2raw.keys():
#     SAVE = os.path.join(save_dir, MEL)
#     if not os.path.exists(SAVE):
#         os.mkdir(SAVE)
#     pick_index = np.zeros(len(Rawfiles)).astype("bool")
#     for raw in mel2raw[MEL]:
#         pick_index = np.logical_or(pick_index, (Rawfiles == raw.encode()))
#     print(f"{MEL} for {np.sum(pick_index)}")
#     c_count += np.sum(pick_index)
#     with h5py.File(os.path.join(SAVE, "data.hdf5"), 'w') as f:
#         order_list = [(i, p[0], p[1]) for i, p in enumerate(
#             zip(HDF5_dict['rawfile'][pick_index], HDF5_dict['scan_number'][pick_index]))]
#         order_list.sort(key=lambda x: (x[1].decode(), x[2]))
#         order = np.array([p[0] for p in order_list]).astype('int')
#         for h5key in HDF5.keys():
#             keydata = HDF5_dict[h5key][pick_index][order]
#             f.create_dataset(h5key, keydata.shape, keydata.dtype, keydata)
# print(f"HDF5: {len(Rawfiles)} - {c_count}")

FEAT = pd.read_csv(csv_data)
index_tab = FEAT.set_index('raw_file')
c_count = 0
for MEL in mel2raw.keys():
    SAVE = os.path.join(save_dir, MEL)
    SAVE = os.path.join(SAVE, "percolator")
    if not os.path.exists(SAVE):
        os.mkdir(SAVE)
    df = None
    for raw in mel2raw[MEL]:
        target = index_tab.filter(like=raw, axis=0)
        if df is None:
            df = target
        else:
            df = pd.concat([df, target], ignore_index=False)
    print(f"{MEL} for {len(df)}")
    c_count += len(df)
    before_keys = list(df.keys())
    df.reset_index(inplace=True)
    df = df.sort_values(by=['raw_file', "scan_number"])
    df.to_csv(os.path.join(SAVE, 'features.csv'), index=False)
print(f"Features: {len(FEAT)} - {c_count}")

PROSIT = pd.read_csv(prosit_data, sep='\t')
index_tab = PROSIT.set_index('SpecId')
c_count = 0
for MEL in mel2raw.keys():
    SAVE = os.path.join(save_dir, MEL)
    SAVE = os.path.join(SAVE, "percolator")
    if not os.path.exists(SAVE):
        os.mkdir(SAVE)
    df = None
    for raw in mel2raw[MEL]:
        target = index_tab.filter(like=raw, axis=0)
        if df is None:
            df = target
        else:
            df = pd.concat([df, target], ignore_index=False)
    print(f"{MEL} for {len(df)}")
    c_count += len(df)
    before_keys = list(df.keys())
    df.reset_index(inplace=True)
    df = df.sort_values(by=['SpecId', "ScanNr"])
    df.to_csv(os.path.join(SAVE, 'prosit.tab'), sep='\t', index=False)
print(f"Prosit: {len(PROSIT)} - {c_count}")

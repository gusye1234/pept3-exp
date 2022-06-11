from cmath import nan
from re import S
import pandas as pd


def get_q_values(feat_tab: pd.DataFrame, ids):
    ids = set(ids)
    index = feat_tab['PSMId'].isin(ids)
    filter_tab = feat_tab[index]
    filter_tab = filter_tab.sort_values(by=['PSMId'])
    return filter_tab


FEATS = ['sa', 'prosit_best']
MODELS = ['pdeep2', 'prosit_l1']

writer = pd.ExcelWriter("data/supp2.xlsx", engine='openpyxl')
# --------------------------------

for which, save_name in zip(["trypsin", 'chymo', "lysc", "gluc"], ["Trypsin", 'Chymotrypsin', "Lys-C", "Glu-C"]):
    print(which)
    assembles = {}
    for frag_model in MODELS:
        for feat in FEATS:
            save_tab = f"/data/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/no_finetuned_twofold/{feat}_target.psms"
            ft_save_tab = f"/data/prosit/figs/fig235/{which}/percolator_up/try/{frag_model}/finetuned_twofold/{feat}_target.psms"
            assembles[f"{frag_model}-{feat}-noft"] = pd.read_csv(
                save_tab, sep='\t')
            assembles[f"{frag_model}-{feat}-ft"] = pd.read_csv(
                ft_save_tab, sep='\t')

    andromeda = pd.read_csv(
        f"/data/prosit/figs/fig235/{which}/percolator_up/try/prosit_l1/no_finetuned_twofold/andromeda_target.psms", sep='\t')
    ok_ids = set(andromeda[andromeda['q-value'] < 0.01]['PSMId'])
    allow_ids = set(andromeda['PSMId'])
    for k, v in assembles.items():
        ok_ids = ok_ids.union(set(v[v['q-value'] < 0.01]['PSMId']))
        allow_ids = allow_ids.intersection(set(v['PSMId']))
    ok_ids = ok_ids.intersection(allow_ids)
    andromeda_1 = get_q_values(andromeda, ok_ids)
    final_tab = andromeda_1[['PSMId', 'peptide', 'q-value']]
    final_tab = final_tab.rename(
        columns={"q-value": 'andromeda q.value', "peptide": "Modified Sequence"})
    for frag_model in MODELS:
        for feat in FEATS:
            filter_tab_noft = get_q_values(
                assembles[f"{frag_model}-{feat}-noft"], ok_ids)
            filter_tab_ft = get_q_values(
                assembles[f"{frag_model}-{feat}-ft"], ok_ids)
            assert len(final_tab) == len(filter_tab_ft)
            assert len(final_tab) == len(filter_tab_noft)
            assert all([p1 == p2 for p1, p2 in zip(
                final_tab['PSMId'], filter_tab_ft['PSMId'])])
            assert all([p1 == p2 for p1, p2 in zip(
                final_tab['PSMId'], filter_tab_noft['PSMId'])])
            final_tab[f"{frag_model}-{feat}-noft q.value"] = filter_tab_noft['q-value']
            final_tab[f"{frag_model}-{feat}-ft q.value"] = filter_tab_ft['q-value']
    print(len(final_tab))
    final_tab.replace("", float("NaN"), inplace=True)
    final_tab.dropna(inplace=True)
    print(len(final_tab))
    final_tab.to_excel(
        writer, sheet_name=f"{save_name} identification", index=False)

    try_fdrs = [(0.005 + 0.0005 * i) for i in range(10)]
    try_fdrs += [(0.01 + 0.005 * i) for i in range(11)]
    fdr_cols = ["q-value(%)", 'andromeda']
    for frag_model in MODELS:
        for feat in FEATS:
            fdr_cols.append(f"{frag_model}-{feat}-noft")
            fdr_cols.append(f"{frag_model}-{feat}-ft")
    fdr_core_data = []
    for fdr in try_fdrs:
        fdr_row = [fdr * 100]
        fdr_row.append((andromeda['q-value'] < fdr).sum())
        for frag_model in MODELS:
            for feat in FEATS:
                fdr_row.append(
                    (assembles[f"{frag_model}-{feat}-noft"]['q-value'] < fdr).sum())
                fdr_row.append(
                    (assembles[f"{frag_model}-{feat}-ft"]['q-value'] < fdr).sum())
        fdr_core_data.append(fdr_row)
    fdr_df = pd.DataFrame(columns=fdr_cols, data=fdr_core_data)
    fdr_df.to_excel(
        writer, sheet_name=f"{save_name} fdr comparison", index=False)


writer.save()

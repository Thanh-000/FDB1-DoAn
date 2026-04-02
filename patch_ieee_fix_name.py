import json

with open('MVS_XAI_Colab_IEEE_CIS.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code': continue
    cid = cell.get('metadata',{}).get('id','')
    if cid == 'run_pipeline':
        src = ''.join(cell['source'])
        src = src.replace('select_tabular_features(df_proc)', 'select_tabular_cols(df_proc)')
        src = src.replace('extract_tabular_view(df_proc, selected)', 'extract_tabular_view(df_proc, selected)')
        src = src.replace('top3 = stage6_xai(best_lgb, Xt_tr[:2000], Xs, selected)',
                          'top3 = stage6_xai(best_lgb, Xt_tr[:2000], Xs, selected)')
        cell['source'] = [line + '\n' for line in src.split('\n')]
        if cell['source'][-1].strip() == '':
            cell['source'] = cell['source'][:-1]
        print('[FIXED] select_tabular_features -> select_tabular_cols')

with open('MVS_XAI_Colab_IEEE_CIS.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('=== IEEE-CIS notebook fixed! ===')

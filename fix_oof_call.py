import json

files = ['MVS_XAI_Colab_IEEE_CIS.ipynb', 'MVS_XAI_Colab_DataPrep_Phase1.ipynb']

for fname in files:
    with open(fname, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            for j, line in enumerate(source):
                if 'models = generate_oof_train(Xt_sm, y_sm,' in line:
                    source[j] = line.replace('generate_oof_train(Xt_sm, y_sm, Xs_tr[:cut], y_tr[:cut])',
                                             'generate_oof_train(Xt_tr[:cut], Xs_tr[:cut], y_tr[:cut])')
                elif 'del Xt_sm, y_sm; gc.collect()' in line:
                    source[j] = line.replace('del Xt_sm, y_sm; gc.collect()', 'import gc; gc.collect()')
                elif 'Xt_sm, y_sm = Xt_tr[:cut], y_tr[:cut]' in line:
                     # Delete this line entirely - actually just comment it so we don't break indices
                     source[j] = '    # Xt_sm, y_sm assignment removed\n'
            
            nb['cells'][i]['source'] = source
            
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print(f"Fixed generate_oof_train calls in {fname}")

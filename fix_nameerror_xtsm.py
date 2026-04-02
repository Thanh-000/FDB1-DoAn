import json

files = ['MVS_XAI_Colab_IEEE_CIS.ipynb', 'MVS_XAI_Colab_DataPrep_Phase1.ipynb']

for fname in files:
    with open(fname, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            for j, line in enumerate(source):
                if '{Xt_sm.shape[0]}' in line:
                    source[j] = line.replace('{Xt_sm.shape[0]}', '{cut}')
            
            nb['cells'][i]['source'] = source
            
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print(f"Fixed Xt_sm in print statement: {fname}")

import json

files = ['MVS_XAI_Colab_IEEE_CIS.ipynb', 'MVS_XAI_Colab_DataPrep_Phase1.ipynb']
for fname in files:
    with open(fname, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        if 'import pandas as pd' in src:
            if 'drive.mount' not in src:
                import_lines = [
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')\n",
                    "\n"
                ]
                nb['cells'][i]['source'] = import_lines + nb['cells'][i]['source']
                with open(fname, 'w', encoding='utf-8') as f_out:
                    json.dump(nb, f_out, indent=1, ensure_ascii=False)
                print(f'Added drive.mount to {fname}')
            break

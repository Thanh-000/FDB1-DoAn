import json
import re

def patch_gpu(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            lines = cell['source']
            new_lines = []
            for line in lines:
                # 1. Enable GPU for XGBoost
                if "xgb.XGBClassifier(" in line and "device=" not in line:
                    line = line.replace("tree_method='hist'", "tree_method='hist', device='cuda'")
                
                # 2. Remove use_cudnn=False so LSTM can run fast on GPU
                if "use_cudnn=False" in line:
                    line = line.replace(", use_cudnn=False", "")
                
                new_lines.append(line)
            cell['source'] = new_lines
            
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

try:
    patch_gpu('MVS_XAI_Colab_IEEE_CIS.ipynb')
    patch_gpu('MVS_XAI_Colab_DataPrep_Phase1.ipynb')
    print("GPU acceleration enabled for XGBoost and LSTM!")
except Exception as e:
    print("Error:", e)

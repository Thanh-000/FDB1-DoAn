import json

files = ['MVS_XAI_Colab_IEEE_CIS.ipynb', 'MVS_XAI_Colab_DataPrep_Phase1.ipynb']

for nb_path in files:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        
        # 1. Clean up imports
        if 'import KMeansSMOTE, SMOTE' in src:
            src = src.replace('from imblearn.over_sampling import KMeansSMOTE, SMOTE\n', '')
            
        # 2. Update markdown tables referencing SMOTE
        if '| K-Means SMOTE |' in src:
            src = src.replace('K-Means SMOTE', 'Native Weights/Focal Loss')
        if '+ K-Means SMOTE |' in src:
            src = src.replace('+ K-Means SMOTE', '+ Native Weights')
            
        # 3. generate_oof_train signature and body cleanup
        if 'def generate_oof_train(' in src:
            lines = src.split('\n')
            new_lines = []
            for line in lines:
                if 'def generate_oof_train(' in line:
                    new_lines.append("def generate_oof_train(X_tab_tr, X_seq_tr, y_tr):")
                else:
                    line = line.replace('y_smote', 'y_tr')
                    line = line.replace('y_raw', 'y_tr')
                    new_lines.append(line)
            src = '\n'.join(new_lines)

        # 4. stage2_walk_forward cleanup
        if 'def stage2_walk_forward(' in src:
            lines = src.split('\n')
            new_lines = []
            skip = False
            for line in lines:
                if '# SMOTE on 80% base subset only' in line:
                    skip = True
                    new_lines.append("    print(f'    Base train size: {cut}')")
                elif 'models = generate_oof_train(Xt_sm' in line:
                    skip = False
                    new_lines.append("    models = generate_oof_train(Xt_tr[:cut], Xs_tr[:cut], y_tr[:cut])")
                elif 'del Xt_sm, y_sm; gc.collect()' in line:
                    new_lines.append("    import gc; gc.collect()")
                    continue
                elif skip:
                    continue
                else:
                    new_lines.append(line)
            src = '\n'.join(new_lines)
            
        nb['cells'][i]['source'] = [l + '\n' for l in src.split('\n')][:-1]

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Cleaned up SMOTE and unused artifacts in {nb_path}")

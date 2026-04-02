import json

nb_path = 'MVS_XAI_Colab_DataPrep_Phase1.ipynb'
out_path = 'MVS_XAI_Colab_DataPrep_Phase1.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src_list = cell.get('source', [])
    if isinstance(src_list, str):
        src_list = [src_list]
    src = ''.join(src_list)

    # 1. Update Title / Header
    if '# 💳 MVS-XAI' in src:
        new_src = src.replace('**(Tier 2 Optimized)**', '**(Tier 3 Optimized: Dest-Profile + Native Weights)**')
        if 'K-Means SMOTE' in new_src and 'balanced_subsample' in new_src:
            new_src = new_src.replace('K-Means SMOTE', 'scale_pos_weight')
        nb['cells'][i]['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in new_src.split('\n')]
        
    # 2. Add Tier 3 to stage1_preprocessing
    elif 'def stage1_preprocessing(df):' in src:
        lines = src.split('\n')
        new_lines = []
        for line in lines:
            if "num_cols = ['amount'," in line:
                new_lines.extend([
                    "    # Tier 3: Destination Account Profiling (Mule Account detection)",
                    "    print('    [Tier 3] Adding Dest-Profile Features...')",
                    "    if 'nameDest' in df.columns:",
                    "        dest_freq = df['nameDest'].value_counts()",
                    "        df['dest_txn_count'] = df['nameDest'].map(dest_freq).astype(np.float32)",
                    "    "
                ])
                new_lines.append(line)
            elif "amountLogRatio', 'hourOfDay']" in line:
                new_lines.append(line.replace("]", ", 'dest_txn_count']"))
            else:
                new_lines.append(line)
        nb['cells'][i]['source'] = [l + ('\n' if not l.endswith('\n') else '') for l in new_lines][:-1]

    # 3. Update extract_sequential_view window size
    elif 'SEQ_WINDOW =' in src and 'def extract_sequential_view' in src:
        src = src.replace('SEQ_WINDOW = 10', 'SEQ_WINDOW = 5  # Tier 3: Reduced window for PaySim')
        nb['cells'][i]['source'] = [l + '\n' for l in src.split('\n')]

    # 4. Update generate_oof_train (add scale_pos_weight for XGB/LGB, remove CatBoost print)
    elif 'def generate_oof_train(' in src:
        src = src.replace(
            "xc = xgb.XGBClassifier(n_estimators=800, max_depth=8, learning_rate=0.03,",
            "ratio = float((y_smote==0).sum()) / max(float((y_smote==1).sum()), 1.0)\n    xc = xgb.XGBClassifier(n_estimators=800, max_depth=8, learning_rate=0.03, scale_pos_weight=ratio,"
        )
        src = src.replace(
            "lc = lgb.LGBMClassifier(n_estimators=800, max_depth=10, learning_rate=0.03,",
            "lc = lgb.LGBMClassifier(n_estimators=800, max_depth=10, learning_rate=0.03, scale_pos_weight=ratio,"
        )
        src = src.replace("print('    [CatBoost] Training...')\n", "")
        nb['cells'][i]['source'] = [l + '\n' for l in src.split('\n')]

    # 5. Remove SMOTE locally in stage2_walk_forward
    elif '# SMOTE on 80% base subset only' in src:
        lines = src.split('\n')
        new_lines = []
        skip = False
        for line in lines:
            if '# SMOTE on 80% base subset only' in line:
                new_lines.append(line)
                new_lines.append("    print('  [Tier 3] SMOTE Removed. Using scale_pos_weight natively.')")
                new_lines.append("    Xt_sm, y_sm = Xt_tr[:cut], y_tr[:cut]")
                new_lines.append("    print(f'    Base train size: {Xt_sm.shape[0]} (SMOTE disabled)')")
                skip = True
            elif 'After SMOTE:' in line:
                skip = False
                continue
            elif skip:
                continue
            else:
                new_lines.append(line)
        nb['cells'][i]['source'] = [l + '\n' for l in new_lines]


with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully applied Tier 3 optimizations to {out_path}!")

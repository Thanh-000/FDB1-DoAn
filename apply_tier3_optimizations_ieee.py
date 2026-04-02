import json
import re

nb_path = 'MVS_XAI_Colab_IEEE_CIS.ipynb'
out_path = 'MVS_XAI_Colab_IEEE_CIS.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src_list = cell.get('source', [])
    if isinstance(src_list, str):
        src_list = [src_list]
    src = ''.join(src_list)

    # 1. Update Title / Header
    if '# 💳 MVS-XAI' in src:
        new_src = src.replace('**(Tier 2 Optimized)**', '**(Tier 3 Optimized: Time-Window + PCA + Native Weights)**')
        if 'K-Means SMOTE' in new_src and 'balanced_subsample' in new_src:
            new_src = new_src.replace('K-Means SMOTE', 'scale_pos_weight')
        nb['cells'][i]['source'] = [line + ('\\n' if not line.endswith('\\n') else '') for line in new_src.split('\\n')]
        
    # 2. Add Tier 3 to stage1_preprocessing
    elif 'def stage1_preprocessing(df):' in src:
        lines = src.split('\n')
        new_lines = []
        for line in lines:
            if '# 1g. Min-Max Scaling' in line:
                new_lines.extend([
                    "    # 1f3. Tier 3: Time-Window Aggregations & Target Encoding",
                    "    print('  [Tier 3] Applying Time-Window (Velocity) & Target Encoding...')",
                    "    import re",
                    "    if 'card1' in df.columns:",
                    "        te_map = df.groupby('card1')['isFraud'].mean().to_dict()",
                    "        df['card1_fraud_rate'] = df['card1'].map(te_map).fillna(df['isFraud'].mean()).astype(np.float32)",
                    "    ",
                    "    if 'TransactionAmt' in df.columns and 'AccountID' in df.columns:",
                    "        group = df.groupby('AccountID')['TransactionAmt']",
                    "        df['Amt_mean_3'] = group.transform(lambda x: x.rolling(3, min_periods=1).mean()).astype(np.float32)",
                    "        df['Amt_std_3'] = group.transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0).astype(np.float32)",
                    "    ",
                    "    # 1f4. Tier 3: V-column Dimensionality Reduction (PCA)",
                    "    v_cols = [c for c in df.columns if re.match(r'^V\d+$', c)]",
                    "    if len(v_cols) > 20:",
                    "        print(f'  [Tier 3] Applying PCA down to 15 components on {len(v_cols)} V-columns...')",
                    "        from sklearn.decomposition import PCA",
                    "        try:",
                    "            pca = PCA(n_components=15, random_state=42)",
                    "            v_pca = pca.fit_transform(df[v_cols].fillna(0))",
                    "            for j in range(15):",
                    "                df[f'V_pca_{j}'] = v_pca[:, j].astype(np.float32)",
                    "            df.drop(columns=v_cols, inplace=True)",
                    "            num_cols = [c for c in num_cols if c not in v_cols]",
                    "            num_cols += [f'V_pca_{j}' for j in range(15)]",
                    "        except Exception as e:",
                    "            print(f'  [Tier 3 WARNING] PCA failed: {e}')",
                    "    ",
                    line
                ])
            else:
                new_lines.append(line)
        nb['cells'][i]['source'] = [l + ('\n' if not l.endswith('\n') else '') for l in new_lines][:-1]

    # 3. Update generate_oof_train (add scale_pos_weight for XGB/LGB, remove CatBoost print)
    elif 'def generate_oof_train(' in src:
        # Easy straight string replace
        
        # XGB
        src = src.replace(
            "xc = xgb.XGBClassifier(n_estimators=800, max_depth=8, learning_rate=0.03,",
            "ratio = float((y_smote==0).sum()) / max(float((y_smote==1).sum()), 1.0)\n    xc = xgb.XGBClassifier(n_estimators=800, max_depth=8, learning_rate=0.03, scale_pos_weight=ratio,"
        )
        # LGB
        src = src.replace(
            "lc = lgb.LGBMClassifier(n_estimators=800, max_depth=10, learning_rate=0.03,",
            "lc = lgb.LGBMClassifier(n_estimators=800, max_depth=10, learning_rate=0.03, scale_pos_weight=ratio,"
        )
        # CatBoost print
        src = src.replace("print('    [CatBoost] Training...')\n", "")
        
        nb['cells'][i]['source'] = [l + '\n' for l in src.split('\n')]

    # 4. Remove SMOTE locally in stage2_walk_forward
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

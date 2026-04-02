import json

def patch_all_improvements(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        cell_id = cell.get('metadata', {}).get('id', '')

        # ============================================================
        # 1. LSTM: epochs 5->15, batch_size 512->256
        # ============================================================
        if 'def train_lstm_oof' in src:
            src = src.replace('epochs=5, bs=512', 'epochs=15, bs=256')
            cell['source'] = [line + '\n' for line in src.split('\n')]
            if cell['source'][-1].strip() == '':
                cell['source'] = cell['source'][:-1]
            print(f'  [LSTM] epochs=15, bs=256')

        # ============================================================
        # 2. GAT: epochs 20->50, larger hidden dim 32->64
        # ============================================================
        if 'def train_gat_oof' in src:
            src = src.replace('epochs=20, lr=1e-3', 'epochs=50, lr=1e-3')
            src = src.replace('in_dim, hid=32', 'in_dim, hid=64')
            cell['source'] = [line + '\n' for line in src.split('\n')]
            if cell['source'][-1].strip() == '':
                cell['source'] = cell['source'][:-1]
            print(f'  [GAT] epochs=50, hid=64')

        # ============================================================
        # 3. XGBoost: n_estimators 300->500
        # ============================================================
        if 'def generate_oof' in src:
            src = src.replace(
                "xgb.XGBClassifier(n_estimators=300, max_depth=8",
                "xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05"
            )
            src = src.replace(
                "lgb.LGBMClassifier(n_estimators=300, max_depth=12",
                "lgb.LGBMClassifier(n_estimators=500, max_depth=12, learning_rate=0.05"
            )
            cell['source'] = [line + '\n' for line in src.split('\n')]
            if cell['source'][-1].strip() == '':
                cell['source'] = cell['source'][:-1]
            print(f'  [XGB/LGB] n_estimators=500, lr=0.05')

        # ============================================================
        # 4. Meta-Learner: stronger regularization C=0.1
        # ============================================================
        if 'def stage5_meta' in src:
            src = src.replace(
                "LogisticRegression(penalty='l2', max_iter=1000",
                "LogisticRegression(penalty='l2', C=0.1, max_iter=1000"
            )
            cell['source'] = [line + '\n' for line in src.split('\n')]
            if cell['source'][-1].strip() == '':
                cell['source'] = cell['source'][:-1]
            print(f'  [Meta] C=0.1 (stronger regularization)')

        # ============================================================
        # 5. LIME: more features + samples
        # ============================================================
        if 'def stage6_xai' in src:
            # LIME improvements
            src = src.replace(
                "le.explain_instance(X_sample[0], lgb_model.predict_proba, num_features=5)",
                "le.explain_instance(X_sample[0], lgb_model.predict_proba, num_features=10, num_samples=10000)"
            )
            # Anchors improvements
            src = src.replace(
                "anch.fit(X_tr[:1000], disc_perc=(25, 50, 75))",
                "anch.fit(X_tr[:2000], disc_perc=(10, 25, 50, 75, 90))"
            )
            cell['source'] = [line + '\n' for line in src.split('\n')]
            if cell['source'][-1].strip() == '':
                cell['source'] = cell['source'][:-1]
            print(f'  [LIME] num_features=10, num_samples=10000')
            print(f'  [Anchors] samples=2000, disc_perc=(10,25,50,75,90)')

    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

print('=== Patching IEEE-CIS ===')
patch_all_improvements('MVS_XAI_Colab_IEEE_CIS.ipynb')
print('\n=== Patching PaySim ===')
patch_all_improvements('MVS_XAI_Colab_DataPrep_Phase1.ipynb')
print('\nAll improvements applied!')

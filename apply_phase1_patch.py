"""
Phase 1 Patch: Remove BiLSTM + XGBoost Meta-Learner
Applied to BOTH: MVS_XAI_Colab_IEEE_CIS.ipynb & MVS_XAI_Colab_DataPrep_Phase1.ipynb
"""
import json

files = ['MVS_XAI_Colab_IEEE_CIS.ipynb', 'MVS_XAI_Colab_DataPrep_Phase1.ipynb']

for fname in files:
    print(f"\n{'='*50}\nPatching: {fname}\n{'='*50}")
    with open(fname, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes = 0
    
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        original = src
        
        # ============================================================
        # PATCH 1: generate_oof_train — Remove BiLSTM, keep 4 models
        # ============================================================
        if 'def generate_oof_train(' in src and 'def predict_oof(' in src:
            new_src = '''def generate_oof_train(X_tab_tr, y_tr):
    """Train 4 optimized models (BiLSTM removed — negative weight in ablation). Return trained models."""
    # MODEL 1: RF (300 trees + balanced)
    print('    [RF] Training...')
    rf = RandomForestClassifier(300, max_depth=20, min_samples_leaf=5,
        class_weight='balanced_subsample', n_jobs=-1, random_state=42)
    rf.fit(X_tab_tr, y_tr)

    # MODEL 2: XGB (800 trees + native weight)
    print('    [XGB] Training...')
    ratio = float((y_tr==0).sum()) / max(float((y_tr==1).sum()), 1.0)
    xc = xgb.XGBClassifier(n_estimators=800, max_depth=8, learning_rate=0.03, scale_pos_weight=ratio,
        tree_method='hist', device='cuda',
        colsample_bytree=0.7, subsample=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5, gamma=0.1,
        random_state=42, n_jobs=-1)
    xc.fit(X_tab_tr, y_tr)

    # MODEL 3: LGB (800 trees + native weight)
    print('    [LGB] Training...')
    lc = lgb.LGBMClassifier(n_estimators=800, max_depth=10, learning_rate=0.03, scale_pos_weight=ratio,
        colsample_bytree=0.7, subsample=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20, num_leaves=63,
        random_state=42, n_jobs=-1, verbose=-1)
    lc.fit(X_tab_tr, y_tr)

    # MODEL 4: CatBoost (auto balanced)
    cb = train_catboost(X_tab_tr, y_tr)

    return rf, xc, lc, cb

def predict_oof(models, X_tab):
    """Predict with 4 trained models. No re-training!"""
    rf, xc, lc, cb = models
    n = X_tab.shape[0]
    oof = np.zeros((n, 4))
    oof[:,0] = rf.predict_proba(X_tab)[:,1]
    oof[:,1] = xc.predict_proba(X_tab)[:,1]
    oof[:,2] = lc.predict_proba(X_tab)[:,1]
    oof[:,3] = cb.predict_proba(X_tab)[:,1]
    print(f'    OOF shape: {oof.shape}')
    return oof, lc

'''
            nb['cells'][i]['source'] = [l + '\n' for l in new_src.split('\n')][:-1]
            changes += 1
            print(f"  [Cell {i}] ✅ Replaced generate_oof_train + predict_oof (4 models, no BiLSTM)")

        # ============================================================
        # PATCH 2: stage5_meta — Replace LR with XGBoost Meta-Learner
        # ============================================================
        if 'def stage5_meta(' in src and 'LogisticRegression' in src:
            new_src = '''def stage5_meta(oof_tr, y_tr, oof_vl):
    print('[Stage 5] Meta-Learner (XGB) on OOF [Nx4]...')
    ratio_meta = float((y_tr==0).sum()) / max(float((y_tr==1).sum()), 1.0)
    m = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
        scale_pos_weight=ratio_meta, eval_metric='aucpr',
        tree_method='hist', device='cuda',
        random_state=42, n_jobs=-1)
    m.fit(oof_tr, y_tr)
    preds = m.predict_proba(oof_vl)[:,1]
    names = ['RF','XGB','LGB','CatBoost']
    importances = m.feature_importances_
    print(f'  Feature Importances: {dict(zip(names, importances.round(4)))}')
    return m, preds

'''
            nb['cells'][i]['source'] = [l + '\n' for l in new_src.split('\n')][:-1]
            changes += 1
            print(f"  [Cell {i}] ✅ Replaced LR Meta with XGBoost Meta (50 trees, depth=3)")

        # ============================================================
        # PATCH 3: Main training loop — Fix function call signatures
        # ============================================================
        if 'generate_oof_train(Xt_tr[:cut]' in src or 'predict_oof(models' in src:
            lines = src.split('\n')
            new_lines = []
            for line in lines:
                # Fix generate_oof_train call (remove Xs_tr arg)
                if 'models = generate_oof_train(Xt_tr[:cut], Xs_tr[:cut], y_tr[:cut])' in line:
                    line = line.replace('generate_oof_train(Xt_tr[:cut], Xs_tr[:cut], y_tr[:cut])',
                                       'generate_oof_train(Xt_tr[:cut], y_tr[:cut])')
                # Fix predict_oof calls (remove X_seq arg)
                if 'predict_oof(models, Xt_tr[cut:], Xs_tr[cut:])' in line:
                    line = line.replace('predict_oof(models, Xt_tr[cut:], Xs_tr[cut:])',
                                       'predict_oof(models, Xt_tr[cut:])')
                if 'predict_oof(models, Xt_vl, Xs_vl)' in line:
                    line = line.replace('predict_oof(models, Xt_vl, Xs_vl)',
                                       'predict_oof(models, Xt_vl)')
                # Fix visualization: model names and weights
                if "models_names = ['RF', 'XGB', 'LGB', 'BiLSTM', 'CatBoost']" in line:
                    line = line.replace("['RF', 'XGB', 'LGB', 'BiLSTM', 'CatBoost']",
                                       "['RF', 'XGB', 'LGB', 'CatBoost']")
                if "weights = meta.coef_[0]" in line:
                    line = line.replace("weights = meta.coef_[0]",
                                       "weights = meta.feature_importances_")
                if "Meta-Learner Weights (5 Models)" in line:
                    line = line.replace("Meta-Learner Weights (5 Models)",
                                       "Meta-Learner Importances (4 Models)")
                if 'Coefficient' in line and 'set_ylabel' in line:
                    line = line.replace('Coefficient', 'Importance')
                if 'axhline(0' in line:
                    # Remove the zero line (importances are always positive)
                    continue
                # Fix OOF markdown descriptions
                if 'OOF [Nx5]' in line and 'print' not in line and 'def ' not in line:
                    line = line.replace('OOF [Nx5]', 'OOF [Nx4]')
                # Fix pipeline complete title
                if 'Tier 2 Optimized' in line:
                    line = line.replace('Tier 2 Optimized', 'Tier 3 Phase1 — 4 Models + XGB Meta')
                new_lines.append(line)
            
            new_src = '\n'.join(new_lines)
            if new_src != src:
                nb['cells'][i]['source'] = [l + '\n' for l in new_src.split('\n')][:-1]
                changes += 1
                print(f"  [Cell {i}] ✅ Fixed function calls + visualization (4 models)")

        # ============================================================
        # PATCH 4: Markdown cells — update architecture descriptions
        # ============================================================
        if cell['cell_type'] == 'markdown':
            if '5 model names' in src or 'OOF [Nx5]' in src or 'Nx5' in src:
                src = src.replace('Nx5', 'Nx4')
                src = src.replace('5 model', '4 model')
                nb['cells'][i]['source'] = [l + '\n' for l in src.split('\n')][:-1]
                changes += 1
                print(f"  [Cell {i}] ✅ Updated markdown (5→4 models)")
            if 'BiLSTM' in src and '|' in src:
                # Table row mentioning BiLSTM in markdown
                lines = src.split('\n')
                new_lines = [l for l in lines if 'BiLSTM' not in l]
                new_src = '\n'.join(new_lines)
                if new_src != src:
                    nb['cells'][i]['source'] = [l + '\n' for l in new_src.split('\n')][:-1]
                    changes += 1
                    print(f"  [Cell {i}] ✅ Removed BiLSTM from markdown table")

    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"\n  📊 Total patches applied: {changes}")

print("\n\n✅ Phase 1 complete! Both notebooks patched.")

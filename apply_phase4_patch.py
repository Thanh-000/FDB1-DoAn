"""
Phase 4: Recall Boost + PCA Expansion + RF→ExtraTrees
Applied to IEEE-CIS (primary) and PaySim (sync)
"""
import json

def patch_file(fname, is_ieee):
    print(f"\n{'='*60}\nPhase 4 Patching: {fname}\n{'='*60}")
    with open(fname, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes = 0
    for i, cell in enumerate(nb.get('cells', [])):
        src = ''.join(cell.get('source', []))
        new_src = src

        # ============================================================
        # PATCH 1: Replace RF with ExtraTrees + Recall Multiplier (1.3x)
        # ============================================================
        if 'def generate_oof_train(X_tab_tr, y_tr):' in new_src and 'def predict_oof' in new_src:
            old_func = '''def generate_oof_train(X_tab_tr, y_tr):
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
    return oof, lc'''

            new_func = '''def generate_oof_train(X_tab_tr, y_tr):
    """Train 4 optimized models (Phase 4: ExtraTrees + Recall Boost). Return trained models."""
    ratio = float((y_tr==0).sum()) / max(float((y_tr==1).sum()), 1.0)
    RECALL_BOOST = 1.3  # Multiplier to push Recall up
    boosted_ratio = ratio * RECALL_BOOST
    
    # MODEL 1: ExtraTrees (replaces RF — more diversity, better minority recall)
    from sklearn.ensemble import ExtraTreesClassifier
    print('    [ExtraTrees] Training...')
    et = ExtraTreesClassifier(500, max_depth=25, min_samples_leaf=3,
        class_weight='balanced_subsample', n_jobs=-1, random_state=42)
    et.fit(X_tab_tr, y_tr)

    # MODEL 2: XGB (800 trees + boosted weight for recall)
    print('    [XGB] Training...')
    xc = xgb.XGBClassifier(n_estimators=800, max_depth=8, learning_rate=0.03,
        scale_pos_weight=boosted_ratio,
        tree_method='hist', device='cuda',
        colsample_bytree=0.7, subsample=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5, gamma=0.1,
        random_state=42, n_jobs=-1)
    xc.fit(X_tab_tr, y_tr)

    # MODEL 3: LGB (800 trees + boosted weight for recall)
    print('    [LGB] Training...')
    lc = lgb.LGBMClassifier(n_estimators=800, max_depth=10, learning_rate=0.03,
        scale_pos_weight=boosted_ratio,
        colsample_bytree=0.7, subsample=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20, num_leaves=63,
        random_state=42, n_jobs=-1, verbose=-1)
    lc.fit(X_tab_tr, y_tr)

    # MODEL 4: CatBoost (auto balanced)
    cb = train_catboost(X_tab_tr, y_tr)

    return et, xc, lc, cb

def predict_oof(models, X_tab):
    """Predict with 4 trained models (Phase 4). No re-training!"""
    et, xc, lc, cb = models
    n = X_tab.shape[0]
    oof = np.zeros((n, 4))
    oof[:,0] = et.predict_proba(X_tab)[:,1]
    oof[:,1] = xc.predict_proba(X_tab)[:,1]
    oof[:,2] = lc.predict_proba(X_tab)[:,1]
    oof[:,3] = cb.predict_proba(X_tab)[:,1]
    print(f'    OOF shape: {oof.shape}')
    return oof, lc'''

            if old_func in new_src:
                new_src = new_src.replace(old_func, new_func)
                changes += 1
                print(f"  [Cell {i}] ✅ RF→ExtraTrees + Recall Boost 1.3x")

        # ============================================================
        # PATCH 2: PCA 15 → 25 components (IEEE only)
        # ============================================================
        if is_ieee and 'n_components=15' in new_src and 'PCA' in new_src:
            new_src = new_src.replace('n_components=15', 'n_components=25')
            new_src = new_src.replace("Applying PCA down to 15 components", "Applying PCA down to 25 components")
            new_src = new_src.replace('for j in range(15):', 'for j in range(25):')
            new_src = new_src.replace("for j in range(15)]", "for j in range(25)]")
            changes += 1
            print(f"  [Cell {i}] ✅ PCA 15→25 components")

        # ============================================================
        # PATCH 3: Update Meta-Learner model names
        # ============================================================
        if "names = ['RF','XGB','LGB','CatBoost']" in new_src:
            new_src = new_src.replace("names = ['RF','XGB','LGB','CatBoost']",
                                      "names = ['ExtraTrees','XGB','LGB','CatBoost']")
            changes += 1
            print(f"  [Cell {i}] ✅ Meta names RF→ExtraTrees")

        # ============================================================
        # PATCH 4: Update Visualization model names
        # ============================================================
        if "models_names = ['RF', 'XGB', 'LGB', 'CatBoost']" in new_src:
            new_src = new_src.replace("models_names = ['RF', 'XGB', 'LGB', 'CatBoost']",
                                      "models_names = ['ExtraTrees', 'XGB', 'LGB', 'CatBoost']")
            changes += 1
            print(f"  [Cell {i}] ✅ Visualization names RF→ExtraTrees")

        # ============================================================
        # PATCH 5: Update imports — add ExtraTreesClassifier
        # ============================================================
        if 'from sklearn.ensemble import RandomForestClassifier' in new_src:
            new_src = new_src.replace(
                'from sklearn.ensemble import RandomForestClassifier',
                'from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier')
            changes += 1
            print(f"  [Cell {i}] ✅ Added ExtraTreesClassifier import")

        if src != new_src:
            nb['cells'][i]['source'] = [l + '\n' for l in new_src.split('\n')][:-1]

    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"\n  📊 Total patches: {changes}")

patch_file('MVS_XAI_Colab_IEEE_CIS.ipynb', is_ieee=True)
patch_file('MVS_XAI_Colab_DataPrep_Phase1.ipynb', is_ieee=False)
print("\n🚀 Phase 4 Complete!")

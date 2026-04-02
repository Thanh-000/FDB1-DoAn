"""
Phase 5: Structural Overhaul
1. Revert ExtraTrees→RF, PCA 25→15 (Phase 3 was better)
2. Full-Data OOF Stacking (internal 3-fold CV replaces 80/20 meta-holdout)
3. Card-level aggregation features
"""
import json

def patch_file(fname, is_ieee):
    print(f"\n{'='*60}\nPhase 5: {fname}\n{'='*60}")
    with open(fname, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes = 0
    for i, cell in enumerate(nb.get('cells', [])):
        src = ''.join(cell.get('source', []))
        new_src = src

        # ============================================================
        # REVERT 1: ExtraTrees → RF (Phase 3 config)
        # ============================================================
        if 'def generate_oof_train(X_tab_tr, y_tr):' in new_src and 'def predict_oof' in new_src:
            old_func = '''def generate_oof_train(X_tab_tr, y_tr):
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

            new_func = '''def generate_oof_train(X_tab_tr, y_tr):
    """Train 4 models (Phase 5: Full-OOF Stacking). Return trained models."""
    ratio = float((y_tr==0).sum()) / max(float((y_tr==1).sum()), 1.0)

    # MODEL 1: RF (300 trees + balanced)
    print('    [RF] Training...')
    rf = RandomForestClassifier(300, max_depth=20, min_samples_leaf=5,
        class_weight='balanced_subsample', n_jobs=-1, random_state=42)
    rf.fit(X_tab_tr, y_tr)

    # MODEL 2: XGB (800 trees + native weight)
    print('    [XGB] Training...')
    xc = xgb.XGBClassifier(n_estimators=800, max_depth=8, learning_rate=0.03,
        scale_pos_weight=ratio,
        tree_method='hist', device='cuda',
        colsample_bytree=0.7, subsample=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5, gamma=0.1,
        random_state=42, n_jobs=-1)
    xc.fit(X_tab_tr, y_tr)

    # MODEL 3: LGB (800 trees + native weight)
    print('    [LGB] Training...')
    lc = lgb.LGBMClassifier(n_estimators=800, max_depth=10, learning_rate=0.03,
        scale_pos_weight=ratio,
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

            if old_func in new_src:
                new_src = new_src.replace(old_func, new_func)
                changes += 1
                print(f"  [Cell {i}] ✅ Reverted ExtraTrees→RF + removed Recall Boost")

        # ============================================================
        # REVERT 2: PCA 25 → 15 (IEEE only)
        # ============================================================
        if is_ieee and 'n_components=25' in new_src:
            new_src = new_src.replace('n_components=25', 'n_components=15')
            new_src = new_src.replace("Applying PCA down to 25 components", "Applying PCA down to 15 components")
            new_src = new_src.replace('for j in range(25):', 'for j in range(15):')
            new_src = new_src.replace("for j in range(25)]", "for j in range(15)]")
            changes += 1
            print(f"  [Cell {i}] ✅ Reverted PCA 25→15")

        # ============================================================
        # REVERT 3: Meta names back to RF
        # ============================================================
        if "names = ['ExtraTrees','XGB','LGB','CatBoost']" in new_src:
            new_src = new_src.replace("names = ['ExtraTrees','XGB','LGB','CatBoost']",
                                      "names = ['RF','XGB','LGB','CatBoost']")
            changes += 1
            print(f"  [Cell {i}] ✅ Meta names reverted to RF")

        if "models_names = ['ExtraTrees', 'XGB', 'LGB', 'CatBoost']" in new_src:
            new_src = new_src.replace("models_names = ['ExtraTrees', 'XGB', 'LGB', 'CatBoost']",
                                      "models_names = ['RF', 'XGB', 'LGB', 'CatBoost']")
            changes += 1
            print(f"  [Cell {i}] ✅ Viz names reverted to RF")

        # ============================================================
        # PATCH 4: MAIN LOOP — Full-Data OOF Stacking (replaces 80/20 holdout)
        # ============================================================
        if 'for fold, (tr_i, vl_i) in enumerate(cv.split(X_tab, y)):' in new_src:
            old_loop_body = """    # === META-HOLDOUT SPLIT (80/20) ===
    cut = int(len(y_tr) * 0.8)
    print(f'  [Meta-Holdout] Base train: {cut}, Meta holdout: {len(y_tr)-cut}')

    # SMOTE on 80% base subset only
    print('  [Tier 3] SMOTE Removed. Using scale_pos_weight natively.')
    # Xt_sm, y_sm assignment removed
    print(f'    Base train size: {cut} (SMOTE disabled)')

    # Stage 4: Train ONCE on 80%
    print('  [Stage 4] Training models on base subset...')
    models = generate_oof_train(Xt_tr[:cut], y_tr[:cut])

    # Predict on 20% HOLDOUT
    print('  [Stage 4] Predicting OOF on meta-holdout (HONEST)...')
    oof_hold, _ = predict_oof(models, Xt_tr[cut:])

    # Predict on VALIDATION
    print('  [Stage 4] Predicting OOF on validation...')
    oof_vl, best_lgb = predict_oof(models, Xt_vl)

    # Stage 5: Meta-learner
    meta, meta_p = stage5_meta(oof_hold, y_tr[cut:], oof_vl)"""

            new_loop_body = """    # === FULL-DATA OOF STACKING (Internal 3-Fold CV) ===
    from sklearn.model_selection import StratifiedKFold as SKF_inner
    inner_cv = SKF_inner(n_splits=3, shuffle=True, random_state=42)
    n_tr = len(y_tr)
    oof_train = np.zeros((n_tr, 4))
    print(f'  [Full-OOF] Generating OOF on {n_tr} samples via 3-fold internal CV...')
    
    for ifold, (itr, ivl) in enumerate(inner_cv.split(Xt_tr, y_tr)):
        print(f'    [Inner {ifold+1}/3] Train: {len(itr)}, Val: {len(ivl)}')
        inner_models = generate_oof_train(Xt_tr[itr], y_tr[itr])
        oof_part, _ = predict_oof(inner_models, Xt_tr[ivl])
        oof_train[ivl] = oof_part
        del inner_models; import gc; gc.collect()
    
    # Train FINAL base models on ALL training data
    print('  [Stage 4] Training FINAL models on FULL training fold...')
    models = generate_oof_train(Xt_tr, y_tr)
    
    # Predict on VALIDATION with full-trained models
    print('  [Stage 4] Predicting OOF on validation...')
    oof_vl, best_lgb = predict_oof(models, Xt_vl)

    # Stage 5: Meta-learner trained on FULL OOF (not just 20%!)
    meta, meta_p = stage5_meta(oof_train, y_tr, oof_vl)"""

            if old_loop_body in new_src:
                new_src = new_src.replace(old_loop_body, new_loop_body)
                changes += 1
                print(f"  [Cell {i}] ✅ Replaced 80/20 Meta-Holdout with Full-Data OOF Stacking (3-fold internal CV)")

        # ============================================================
        # PATCH 5 (IEEE only): Add card-level aggregation features
        # ============================================================
        if is_ieee and "# 1f2. Advanced Interaction Features (Tier 1 & Kaggle Grandmaster)" in new_src:
            target = """    if 'TransactionDT' in df.columns and 'AccountID' in df.columns:
        df['DT_diff'] = df.groupby('AccountID')['TransactionDT'].diff().fillna(86400).astype(np.float32)
        df['UID_count'] = df.groupby('AccountID')['TransactionDT'].transform('count').astype(np.float32)
        df['UID_Amt_mean'] = df.groupby('AccountID')['TransactionAmt'].transform('mean').astype(np.float32)
        df['UID_Amt_std'] = df.groupby('AccountID')['TransactionAmt'].transform('std').fillna(0).astype(np.float32)"""

            replacement = """    if 'TransactionDT' in df.columns and 'AccountID' in df.columns:
        df['DT_diff'] = df.groupby('AccountID')['TransactionDT'].diff().fillna(86400).astype(np.float32)
        df['UID_count'] = df.groupby('AccountID')['TransactionDT'].transform('count').astype(np.float32)
        df['UID_Amt_mean'] = df.groupby('AccountID')['TransactionAmt'].transform('mean').astype(np.float32)
        df['UID_Amt_std'] = df.groupby('AccountID')['TransactionAmt'].transform('std').fillna(0).astype(np.float32)
    
    # Card-level aggregation features (high-value for fraud detection)
    for card_col in ['card1', 'card2', 'card5']:
        if card_col in df.columns and 'TransactionAmt' in df.columns:
            df[f'{card_col}_count'] = df.groupby(card_col)['TransactionAmt'].transform('count').astype(np.float32)
            df[f'{card_col}_amt_mean'] = df.groupby(card_col)['TransactionAmt'].transform('mean').astype(np.float32)
            df[f'{card_col}_amt_std'] = df.groupby(card_col)['TransactionAmt'].transform('std').fillna(0).astype(np.float32)"""

            if target in new_src:
                new_src = new_src.replace(target, replacement)
                changes += 1
                print(f"  [Cell {i}] ✅ Added card-level aggregation features (card1/card2/card5)")

        # ============================================================
        # PATCH 6 (IEEE only): Add card-agg features to select_tabular_cols
        # ============================================================
        if is_ieee and "def select_tabular_cols(df):" in new_src:
            target_extra = """             'Amt_cents', 'DT_diff', 'UID_count', 'UID_Amt_mean', 'UID_Amt_std', 'D1n']"""
            replacement_extra = """             'Amt_cents', 'DT_diff', 'UID_count', 'UID_Amt_mean', 'UID_Amt_std', 'D1n']
    
    # Add card-level agg features
    card_agg_cols = [c for c in df.columns if any(c.startswith(p) for p in ['card1_count', 'card1_amt', 'card2_count', 'card2_amt', 'card5_count', 'card5_amt'])]
    extra.extend(card_agg_cols)"""

            if target_extra in new_src:
                new_src = new_src.replace(target_extra, replacement_extra)
                changes += 1
                print(f"  [Cell {i}] ✅ Added card-agg features to tabular selection")

        if src != new_src:
            nb['cells'][i]['source'] = [l + '\n' for l in new_src.split('\n')][:-1]

    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"\n  📊 Total patches: {changes}")

patch_file('MVS_XAI_Colab_IEEE_CIS.ipynb', is_ieee=True)
patch_file('MVS_XAI_Colab_DataPrep_Phase1.ipynb', is_ieee=False)
print("\n🚀 Phase 5 Complete!")

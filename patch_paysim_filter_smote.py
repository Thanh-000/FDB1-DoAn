import json

with open('MVS_XAI_Colab_DataPrep_Phase1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    cell_id = cell.get('metadata', {}).get('id', '')

    # 1. DATA LOADING: Full load → Filter TRANSFER + CASH_OUT
    if cell_id == 'main_load':
        cell['source'] = [
            "# ====== DATA LOADING (PaySim — Filter TRANSFER + CASH_OUT) ======\n",
            "import os, pandas as pd, numpy as np, zipfile, glob, gc\n",
            "\n",
            "ZIP_PATH = '/content/drive/MyDrive/MVS_XAI_Data/paysim1.zip'\n",
            "EXTRACT_DIR = '/content/paysim-dataset'\n",
            "\n",
            "print('--- PaySim Data Loading ---')\n",
            "df_raw = None\n",
            "\n",
            "if os.path.exists(ZIP_PATH):\n",
            "    print('\\u2705 Found ZIP! Extracting...')\n",
            "    os.makedirs(EXTRACT_DIR, exist_ok=True)\n",
            "    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:\n",
            "        zf.extractall(EXTRACT_DIR)\n",
            "    csv_files = glob.glob(f'{EXTRACT_DIR}/**/*.csv', recursive=True)\n",
            "    if csv_files:\n",
            "        df_full = pd.read_csv(csv_files[0])\n",
            "        print(f'  Full dataset: {df_full.shape}')\n",
            "        print(f'  Types: {df_full[\"type\"].value_counts().to_dict()}')\n",
            "        print(f'  Fraud by type:')\n",
            "        print(df_full.groupby('type')['isFraud'].sum())\n",
            "        # Filter: fraud only in TRANSFER + CASH_OUT (Lopez-Rojas 2016)\n",
            "        df_raw = df_full[df_full['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()\n",
            "        df_raw = df_raw.reset_index(drop=True)\n",
            "        del df_full; gc.collect()\n",
            "        for col in df_raw.select_dtypes(include=['float64']).columns:\n",
            "            df_raw[col] = df_raw[col].astype(np.float32)\n",
            "        print(f'\\n  Filtered (TRANSFER+CASH_OUT): {df_raw.shape}')\n",
            "        print(f'  Fraud: {df_raw[\"isFraud\"].sum()} ({df_raw[\"isFraud\"].mean():.4%})')\n",
            "        print(f'  Memory: {df_raw.memory_usage(deep=True).sum()/1e9:.2f} GB')\n",
            "else:\n",
            "    print('\\u274c ZIP not found')\n",
            "\n",
            "if df_raw is None:\n",
            "    print('Falling back to synthetic data...')\n",
            "    np.random.seed(42); N = 20000\n",
            "    df_raw = pd.DataFrame({\n",
            "        'step': np.sort(np.random.randint(1, 500, N)),\n",
            "        'type': np.random.choice(['TRANSFER', 'CASH_OUT'], N),\n",
            "        'amount': np.random.uniform(10, 50000, N).astype(np.float32),\n",
            "        'nameOrig': ['C'+str(i) for i in np.random.randint(0, 2000, N)],\n",
            "        'oldbalanceOrg': np.random.uniform(0, 100000, N).astype(np.float32),\n",
            "        'newbalanceOrig': np.random.uniform(0, 100000, N).astype(np.float32),\n",
            "        'nameDest': ['C'+str(i) for i in np.random.randint(2000, 4000, N)],\n",
            "        'oldbalanceDest': np.random.uniform(0, 100000, N).astype(np.float32),\n",
            "        'newbalanceDest': np.random.uniform(0, 100000, N).astype(np.float32),\n",
            "        'isFraud': np.random.choice([0, 1], N, p=[0.95, 0.05])})\n",
            "print(df_raw.head())\n",
        ]
        print('[DONE] Data loading: full 6.3M -> filter TRANSFER+CASH_OUT -> ~2.77M')

    # 2. OOF: Restore SMOTE-compatible signature (same as IEEE)
    if cell_id == 'oof_gen':
        cell['source'] = [
            "def generate_oof(X_tab_tr, y_smote, X_tab_vl,\n",
            "                 X_seq_tr, X_seq_vl, y_raw,\n",
            "                 X_grp_tr, X_grp_vl):\n",
            "    n = X_tab_vl.shape[0]\n",
            "    oof = np.zeros((n, 5))\n",
            "    print('    [RF] Training...')\n",
            "    rf = RandomForestClassifier(100, max_depth=15, n_jobs=-1, random_state=42)\n",
            "    rf.fit(X_tab_tr, y_smote); oof[:,0] = rf.predict_proba(X_tab_vl)[:,1]\n",
            "    print('    [XGB] Training...')\n",
            "    xc = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,\n",
            "        tree_method='hist', device='cuda',\n",
            "        colsample_bytree=0.7, subsample=0.8, random_state=42, n_jobs=-1)\n",
            "    xc.fit(X_tab_tr, y_smote); oof[:,1] = xc.predict_proba(X_tab_vl)[:,1]\n",
            "    print('    [LGB] Training...')\n",
            "    lc = lgb.LGBMClassifier(n_estimators=500, max_depth=12, learning_rate=0.05,\n",
            "        colsample_bytree=0.7, subsample=0.8,\n",
            "        random_state=42, n_jobs=-1, verbose=-1)\n",
            "    lc.fit(X_tab_tr, y_smote); oof[:,2] = lc.predict_proba(X_tab_vl)[:,1]\n",
            "    oof[:,3] = train_lstm_oof(X_seq_tr, y_raw, X_seq_vl)\n",
            "    oof[:,4] = train_gat_oof(X_grp_tr, y_raw, X_grp_vl)\n",
            "    print(f'    OOF shape: {oof.shape}')\n",
            "    return oof, lc\n",
        ]
        print('[DONE] OOF: SMOTE-compatible signature (same as IEEE)')

    # 3. PIPELINE: Restore SMOTE (same as IEEE)
    if cell_id == 'run_pipeline':
        cell['source'] = [
            "# ====== STAGE 1 ======\n",
            "df_proc, scaler = stage1_preprocessing(df_raw)\n",
            "del df_raw; import gc; gc.collect()\n",
            "\n",
            "# ====== STAGE 3 ======\n",
            "print('\\n[Stage 3] Multi-View Feature Engineering...')\n",
            "X_tab = extract_tabular_view(df_proc)\n",
            "X_seq = extract_sequential_view(df_proc)\n",
            "X_grp, gcols = extract_graph_view(df_proc)\n",
            "y = df_proc['isFraud'].values\n",
            "df_meta = df_proc[['amount']].copy() if 'amount' in df_proc.columns else None\n",
            "del df_proc; gc.collect()\n",
            "print('  Data arrays ready. Memory freed.')\n",
            "\n",
            "# ====== STAGE 2: 5-Block Walk-Forward CV ======\n",
            "print('\\n[Stage 2] 5-Block Walk-Forward CV...')\n",
            "tscv = TimeSeriesSplit(n_splits=5)\n",
            "fold_metrics = []\n",
            "\n",
            "for fold, (tr_i, vl_i) in enumerate(tscv.split(X_tab)):\n",
            "    print(f'\\n{\"=\"*60}\\nFOLD {fold+1}/5\\n{\"=\"*60}')\n",
            "    Xt_tr, Xt_vl = X_tab[tr_i], X_tab[vl_i]\n",
            "    Xs_tr, Xs_vl = X_seq[tr_i], X_seq[vl_i]\n",
            "    Xg_tr, Xg_vl = X_grp[tr_i], X_grp[vl_i]\n",
            "    y_tr, y_vl = y[tr_i], y[vl_i]\n",
            "\n",
            "    # K-Means SMOTE on Tabular ONLY (consistent with IEEE pipeline)\n",
            "    print('  [SMOTE] K-Means SMOTE on Tabular branch...')\n",
            "    try:\n",
            "        sm = KMeansSMOTE(cluster_balance_threshold=0.1, random_state=42)\n",
            "        Xt_sm, y_sm = sm.fit_resample(Xt_tr, y_tr)\n",
            "    except:\n",
            "        Xt_sm, y_sm = SMOTE(random_state=42).fit_resample(Xt_tr, y_tr)\n",
            "    print(f'    After SMOTE: {Xt_sm.shape[0]} (was {Xt_tr.shape[0]})')\n",
            "\n",
            "    # Stage 4: OOF\n",
            "    print('  [Stage 4] OOF [Nx5]...')\n",
            "    oof_vl, best_lgb = generate_oof(Xt_sm, y_sm, Xt_vl, Xs_tr, Xs_vl, y_tr, Xg_tr, Xg_vl)\n",
            "    oof_tr, _ = generate_oof(Xt_sm, y_sm, Xt_tr, Xs_tr, Xs_tr, y_tr, Xg_tr, Xg_tr)\n",
            "\n",
            "    # Stage 5: Meta\n",
            "    meta, meta_p = stage5_meta(oof_tr, y_tr, oof_vl)\n",
            "    auprc = average_precision_score(y_vl, meta_p)\n",
            "    roc = roc_auc_score(y_vl, meta_p)\n",
            "    fold_metrics.append({'Fold': fold+1, 'AUPRC': auprc, 'ROC-AUC': roc})\n",
            "    print(f'  FOLD {fold+1}: AUPRC={auprc:.4f}, ROC-AUC={roc:.4f}')\n",
            "    del Xt_sm, y_sm; gc.collect()\n",
            "\n",
            "    if fold == 4:\n",
            "        print(f'\\n{\"=\"*60}\\nFOLD 5 (FINAL): XAI + Dual-Output + HITL\\n{\"=\"*60}')\n",
            "        Xs = Xt_vl[:200]\n",
            "        top3 = stage6_xai(best_lgb, Xt_tr[:2000], Xs, TABULAR_FEATURES)\n",
            "        stage7_dual_output(meta_p, top3, y_vl)\n",
            "        if df_meta is not None:\n",
            "            stage8_hitl(meta_p, df_meta.iloc[vl_i])\n",
            "\n",
            "print(f'\\n{\"=\"*60}\\nMVS-XAI PaySim Pipeline Complete\\n{\"=\"*60}')\n",
            "mdf = pd.DataFrame(fold_metrics)\n",
            "print(mdf.to_string(index=False))\n",
            "print(f'\\nMean AUPRC: {mdf[\"AUPRC\"].mean():.4f} +/- {mdf[\"AUPRC\"].std():.4f}')\n",
            "print(f'Mean ROC-AUC: {mdf[\"ROC-AUC\"].mean():.4f} +/- {mdf[\"ROC-AUC\"].std():.4f}')\n",
        ]
        print('[DONE] Pipeline: SMOTE restored (consistent with IEEE)')

with open('MVS_XAI_Colab_DataPrep_Phase1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('\n=== PaySim: Filter + SMOTE (consistent with IEEE) applied! ===')

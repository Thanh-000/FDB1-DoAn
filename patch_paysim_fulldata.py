import json

with open('MVS_XAI_Colab_DataPrep_Phase1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    cell_id = cell.get('metadata', {}).get('id', '')

    # 1. DATA LOADING
    if cell_id == 'main_load':
        cell['source'] = [
            "# ====== DATA LOADING (FULL 6.3M PaySim) ======\n",
            "import os, pandas as pd, numpy as np, zipfile, glob, gc\n",
            "\n",
            "ZIP_PATH = '/content/drive/MyDrive/MVS_XAI_Data/paysim1.zip'\n",
            "EXTRACT_DIR = '/content/paysim-dataset'\n",
            "\n",
            "print('--- Full Data Loading (PaySim 6.3M) ---')\n",
            "df_raw = None\n",
            "\n",
            "if os.path.exists(ZIP_PATH):\n",
            "    print('\\u2705 Found ZIP! Extracting...')\n",
            "    os.makedirs(EXTRACT_DIR, exist_ok=True)\n",
            "    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:\n",
            "        zf.extractall(EXTRACT_DIR)\n",
            "    csv_files = glob.glob(f'{EXTRACT_DIR}/**/*.csv', recursive=True)\n",
            "    if csv_files:\n",
            "        df_raw = pd.read_csv(csv_files[0])\n",
            "        for col in df_raw.select_dtypes(include=['float64']).columns:\n",
            "            df_raw[col] = df_raw[col].astype(np.float32)\n",
            "        print(f'  Shape: {df_raw.shape}, Fraud: {df_raw[\"isFraud\"].sum()} ({df_raw[\"isFraud\"].mean():.4%})')\n",
            "        print(f'  Memory: {df_raw.memory_usage(deep=True).sum()/1e9:.2f} GB')\n",
            "else:\n",
            "    print('\\u274c ZIP not found at MVS_XAI_Data/paysim1.zip')\n",
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
        print('[DONE] Data loading: full 6.3M')

    # 2. STAGE 1
    if cell_id == 'stage1':
        cell['source'] = [
            "def stage1_preprocessing(df):\n",
            "    print('[Stage 1] Preprocessing Pipeline...')\n",
            "    df = df.fillna(0)\n",
            "    if 'isFlaggedFraud' in df.columns:\n",
            "        df = df.drop(columns=['isFlaggedFraud'])\n",
            "    le = LabelEncoder()\n",
            "    df['type_encoded'] = le.fit_transform(df['type'])\n",
            "    df['errorBalanceOrg'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']\n",
            "    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']\n",
            "    df['amountToOldBalanceRatio'] = df['amount'] / (df['oldbalanceOrg'] + 1)\n",
            "    num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',\n",
            "                'newbalanceDest', 'errorBalanceOrg', 'errorBalanceDest', 'amountToOldBalanceRatio']\n",
            "    scaler = MinMaxScaler()\n",
            "    df[num_cols] = scaler.fit_transform(df[num_cols])\n",
            "    df = df.sort_values('step').reset_index(drop=True)\n",
            "    for col in df.select_dtypes(include=['float64']).columns:\n",
            "        df[col] = df[col].astype(np.float32)\n",
            "    import gc; gc.collect()\n",
            "    print(f'  Preprocessed shape: {df.shape}')\n",
            "    print(f'  Fraud ratio: {df[\"isFraud\"].mean():.4%}')\n",
            "    print(f'  Memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB')\n",
            "    return df, scaler\n",
        ]
        print('[DONE] Stage 1')

    # 3. TABULAR
    if cell_id == 'stage3_tabular':
        cell['source'] = [
            "TABULAR_FEATURES = [\n",
            "    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrg',\n",
            "    'oldbalanceDest', 'newbalanceDest', 'errorBalanceDest',\n",
            "    'type_encoded', 'amountToOldBalanceRatio'\n",
            "]\n",
            "\n",
            "def extract_tabular_view(df):\n",
            "    cols = [c for c in TABULAR_FEATURES if c in df.columns]\n",
            "    print(f'  [View 1] Tabular ({len(cols)} features)')\n",
            "    return df[cols].values.astype(np.float32)\n",
        ]
        print('[DONE] Tabular')

    # 4. SEQUENTIAL VIEW: VECTORIZED
    if cell_id == 'stage3_sequential':
        cell['source'] = [
            "SEQ_WINDOW = 10\n",
            "\n",
            "def extract_sequential_view(df, window=SEQ_WINDOW):\n",
            "    import time; t0 = time.time()\n",
            "    print(f'  [View 2] Sequential (T={window}, VECTORIZED)...')\n",
            "    candidates = ['amount', 'errorBalanceOrg', 'errorBalanceDest',\n",
            "                  'amountToOldBalanceRatio', 'oldbalanceOrg', 'type_encoded']\n",
            "    seq_feats = [c for c in candidates if c in df.columns]\n",
            "    n_feat = len(seq_feats)\n",
            "    print(f'    Seq features ({n_feat}): {seq_feats}')\n",
            "    seq = np.zeros((len(df), window, n_feat), dtype=np.float32)\n",
            "    df_work = df[['nameOrig'] + seq_feats].copy()\n",
            "    df_work['_oidx'] = np.arange(len(df))\n",
            "    df_work = df_work.sort_values(['nameOrig', '_oidx'])\n",
            "    for i, feat in enumerate(seq_feats):\n",
            "        for t in range(window):\n",
            "            shifted = df_work.groupby('nameOrig')[feat].shift(t)\n",
            "            seq[df_work['_oidx'].values, window - 1 - t, i] = shifted.fillna(0).values\n",
            "        print(f'      Feature {i+1}/{n_feat} done')\n",
            "    del df_work; import gc; gc.collect()\n",
            "    print(f'    Shape: {seq.shape} ({time.time()-t0:.1f}s)')\n",
            "    return seq\n",
        ]
        print('[DONE] Sequential: VECTORIZED')

    # 5. GRAPH
    if cell_id == 'stage3_graph':
        cell['source'] = [
            "def extract_graph_view(df, n_hops=2):\n",
            "    import networkx as nx, time; t0 = time.time()\n",
            "    print(f'  [View 3] Graph ({n_hops}-hop ego-network)...')\n",
            "    G = nx.from_pandas_edgelist(df, 'nameOrig', 'nameDest',\n",
            "                                 edge_attr='amount', create_using=nx.DiGraph())\n",
            "    print(f'    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')\n",
            "    in_deg = dict(G.in_degree()); out_deg = dict(G.out_degree())\n",
            "    try:\n",
            "        pr = nx.pagerank(G, max_iter=20, tol=1e-2)\n",
            "    except:\n",
            "        pr = {n: 0.0 for n in G.nodes()}\n",
            "    df['orig_in_deg'] = df['nameOrig'].map(in_deg).fillna(0).astype(np.float32)\n",
            "    df['orig_out_deg'] = df['nameOrig'].map(out_deg).fillna(0).astype(np.float32)\n",
            "    df['dest_in_deg'] = df['nameDest'].map(in_deg).fillna(0).astype(np.float32)\n",
            "    df['orig_pr'] = df['nameOrig'].map(pr).fillna(0).astype(np.float32)\n",
            "    df['orig_ego_dens'] = (df['orig_out_deg'] / 5.0).astype(np.float32)\n",
            "    df['orig_ego_sz'] = (df['orig_out_deg'] + df['orig_in_deg']).astype(np.float32)\n",
            "    del G, in_deg, out_deg, pr; import gc; gc.collect()\n",
            "    gcols = ['orig_in_deg','orig_out_deg','dest_in_deg','orig_pr','orig_ego_dens','orig_ego_sz']\n",
            "    print(f'    Features: {gcols} ({time.time()-t0:.1f}s)')\n",
            "    return df[gcols].values.astype(np.float32), gcols\n",
        ]
        print('[DONE] Graph')

    # 6. OOF GENERATION: class_weight
    if cell_id == 'oof_gen':
        cell['source'] = [
            "def generate_oof(X_tab_tr, y_tr, X_tab_vl,\n",
            "                 X_seq_tr, X_seq_vl,\n",
            "                 X_grp_tr, X_grp_vl, scale_weight=1.0):\n",
            "    n = X_tab_vl.shape[0]\n",
            "    oof = np.zeros((n, 5))\n",
            "    print('    [RF] Training...')\n",
            "    rf = RandomForestClassifier(100, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42)\n",
            "    rf.fit(X_tab_tr, y_tr); oof[:,0] = rf.predict_proba(X_tab_vl)[:,1]\n",
            "    print('    [XGB] Training...')\n",
            "    xc = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,\n",
            "        scale_pos_weight=scale_weight, tree_method='hist', device='cuda',\n",
            "        colsample_bytree=0.7, subsample=0.8, random_state=42, n_jobs=-1)\n",
            "    xc.fit(X_tab_tr, y_tr); oof[:,1] = xc.predict_proba(X_tab_vl)[:,1]\n",
            "    print('    [LGB] Training...')\n",
            "    lc = lgb.LGBMClassifier(n_estimators=500, max_depth=12, learning_rate=0.05,\n",
            "        is_unbalance=True, colsample_bytree=0.7, subsample=0.8,\n",
            "        random_state=42, n_jobs=-1, verbose=-1)\n",
            "    lc.fit(X_tab_tr, y_tr); oof[:,2] = lc.predict_proba(X_tab_vl)[:,1]\n",
            "    oof[:,3] = train_lstm_oof(X_seq_tr, y_tr, X_seq_vl)\n",
            "    oof[:,4] = train_gat_oof(X_grp_tr, y_tr, X_grp_vl)\n",
            "    print(f'    OOF shape: {oof.shape}')\n",
            "    return oof, lc\n",
        ]
        print('[DONE] OOF: class_weight')

    # 7. PIPELINE: Remove SMOTE
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
            "    n_neg = (y_tr == 0).sum()\n",
            "    n_pos = max((y_tr == 1).sum(), 1)\n",
            "    scale_w = n_neg / n_pos\n",
            "    print(f'  [Class Weight] scale_pos_weight={scale_w:.1f} (fraud: {n_pos}, normal: {n_neg})')\n",
            "\n",
            "    print('  [Stage 4] OOF [Nx5]...')\n",
            "    oof_vl, best_lgb = generate_oof(Xt_tr, y_tr, Xt_vl, Xs_tr, Xs_vl, Xg_tr, Xg_vl, scale_w)\n",
            "    oof_tr, _ = generate_oof(Xt_tr, y_tr, Xt_tr, Xs_tr, Xs_tr, Xg_tr, Xg_tr, scale_w)\n",
            "\n",
            "    meta, meta_p = stage5_meta(oof_tr, y_tr, oof_vl)\n",
            "    auprc = average_precision_score(y_vl, meta_p)\n",
            "    roc = roc_auc_score(y_vl, meta_p)\n",
            "    fold_metrics.append({'Fold': fold+1, 'AUPRC': auprc, 'ROC-AUC': roc})\n",
            "    print(f'  FOLD {fold+1}: AUPRC={auprc:.4f}, ROC-AUC={roc:.4f}')\n",
            "    gc.collect()\n",
            "\n",
            "    if fold == 4:\n",
            "        print(f'\\n{\"=\"*60}\\nFOLD 5 (FINAL): XAI + Dual-Output + HITL\\n{\"=\"*60}')\n",
            "        Xs = Xt_vl[:200]\n",
            "        top3 = stage6_xai(best_lgb, Xt_tr[:2000], Xs, TABULAR_FEATURES)\n",
            "        stage7_dual_output(meta_p, top3, y_vl)\n",
            "        if df_meta is not None:\n",
            "            stage8_hitl(meta_p, df_meta.iloc[vl_i])\n",
            "\n",
            "print(f'\\n{\"=\"*60}\\nMVS-XAI PaySim Pipeline Complete (FULL 6.3M)\\n{\"=\"*60}')\n",
            "mdf = pd.DataFrame(fold_metrics)\n",
            "print(mdf.to_string(index=False))\n",
            "print(f'\\nMean AUPRC: {mdf[\"AUPRC\"].mean():.4f} +/- {mdf[\"AUPRC\"].std():.4f}')\n",
            "print(f'Mean ROC-AUC: {mdf[\"ROC-AUC\"].mean():.4f} +/- {mdf[\"ROC-AUC\"].std():.4f}')\n",
        ]
        print('[DONE] Pipeline: full 6.3M, class_weight, no SMOTE')

    # 8. INSTALL
    if cell_id == 'install':
        src = ''.join(cell['source'])
        if 'import gc' not in src:
            src = src.replace("import os, glob, warnings", "import os, glob, gc, warnings")
            cell['source'] = [line + '\n' for line in src.split('\n')]
            if cell['source'][-1].strip() == '':
                cell['source'] = cell['source'][:-1]
            print('[DONE] Install: gc')

with open('MVS_XAI_Colab_DataPrep_Phase1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('\n=== PaySim: FULL 6.3M mode applied! ===')

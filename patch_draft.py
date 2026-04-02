import json

for file in ['MVS_XAI_Colab_DataPrep_Phase1.ipynb', 'MVS_XAI_Colab_IEEE_CIS.ipynb']:
    is_paysim = 'Phase1' in file
    ds = 'PaySim' if is_paysim else 'IEEE-CIS'

    with open(file, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for c in nb['cells']:
        if c['cell_type'] != 'code': continue
        cid = c.get('metadata', {}).get('id', '')

        # ============================================================
        # 1. RESTORE ORIGINAL generate_oof (train+predict in 1 call)
        # ============================================================
        if cid == 'oof_gen':
            c['source'] = [
                "def generate_oof(X_tab_sm, y_sm, X_tab_pred, X_seq_pred, X_grp_pred):\n",
                "    \"\"\"Train 5 models on SMOTE data, predict on target set.\"\"\"\n",
                "    n = X_tab_pred.shape[0]\n",
                "    oof = np.zeros((n, 5))\n",
                "\n",
                "    print('    [RF] Training...')\n",
                "    rf = RandomForestClassifier(100, max_depth=15, n_jobs=-1, random_state=42)\n",
                "    rf.fit(X_tab_sm, y_sm)\n",
                "    oof[:,0] = rf.predict_proba(X_tab_pred)[:,1]\n",
                "\n",
                "    print('    [XGB] Training...')\n",
                "    xc = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,\n",
                "        tree_method='hist', device='cuda',\n",
                "        colsample_bytree=0.7, subsample=0.8, random_state=42, n_jobs=-1)\n",
                "    xc.fit(X_tab_sm, y_sm)\n",
                "    oof[:,1] = xc.predict_proba(X_tab_pred)[:,1]\n",
                "\n",
                "    print('    [LGB] Training...')\n",
                "    lc = lgb.LGBMClassifier(n_estimators=500, max_depth=12, learning_rate=0.05,\n",
                "        colsample_bytree=0.7, subsample=0.8,\n",
                "        random_state=42, n_jobs=-1, verbose=-1)\n",
                "    lc.fit(X_tab_sm, y_sm)\n",
                "    oof[:,2] = lc.predict_proba(X_tab_pred)[:,1]\n",
                "\n",
                "    print('    [LSTM] Focal Loss training...')\n",
                "    lstm_m = build_lstm(X_seq_pred.shape[1], X_seq_pred.shape[2])\n",
                "    lstm_m.fit(X_seq_pred, np.zeros(n), epochs=15, batch_size=2048, verbose=0)\n",
                "    oof[:,3] = lstm_m.predict(X_seq_pred, batch_size=4096, verbose=0).ravel()\n",
                "\n",
                "    print('    [GAT] Focal Loss training...')\n",
                "    oof[:,4] = train_gat_oof(X_grp_pred, np.zeros(n), X_grp_pred)\n",
                "\n",
                "    print(f'    OOF shape: {oof.shape}')\n",
                "    return oof, lc\n",
            ]
            # WAIT - this won't work because LSTM/GAT need proper training data
            # Let me think about this more carefully

    # Don't save
    pass

# The problem is that the ORIGINAL generate_oof function's exact structure varied
# between PaySim and IEEE. Let me read what ACTUALLY worked by checking the 
# conversation logs for the original version that got 0.85
print('Need to reconstruct original from conversation history')

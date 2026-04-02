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
        # 1. Restore OOF with PROPER hyperparameters (not reduced)
        # ============================================================
        if cid == 'oof_gen':
            if is_paysim:
                tab_features_line = "TABULAR_FEATURES"
            else:
                tab_features_line = "selected"
            
            c['source'] = [
                "def generate_oof_train(X_tab_tr, y_smote, X_seq_tr, y_raw, X_grp_tr):\n",
                "    \"\"\"Train 5 models ONCE. Return trained models.\"\"\"\n",
                "    print('    [RF] Training...')\n",
                "    rf = RandomForestClassifier(100, max_depth=15, n_jobs=-1, random_state=42)\n",
                "    rf.fit(X_tab_tr, y_smote)\n",
                "    print('    [XGB] Training...')\n",
                "    xc = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,\n",
                "        tree_method='hist', device='cuda',\n",
                "        colsample_bytree=0.7, subsample=0.8, random_state=42, n_jobs=-1)\n",
                "    xc.fit(X_tab_tr, y_smote)\n",
                "    print('    [LGB] Training...')\n",
                "    lc = lgb.LGBMClassifier(n_estimators=500, max_depth=12, learning_rate=0.05,\n",
                "        colsample_bytree=0.7, subsample=0.8,\n",
                "        random_state=42, n_jobs=-1, verbose=-1)\n",
                "    lc.fit(X_tab_tr, y_smote)\n",
                "    print('    [LSTM] Training...')\n",
                "    lstm_model = build_lstm(X_seq_tr.shape[1], X_seq_tr.shape[2])\n",
                "    lstm_model.fit(X_seq_tr, y_raw, epochs=15, batch_size=2048, verbose=0,\n",
                "                  class_weight={0:1, 1:max(1, int((y_raw==0).sum()/(y_raw==1).sum()))})\n",
                "    print('    [GAT] Training...')\n",
                "    gat_model = train_gat_oof_model(X_grp_tr, y_raw, epochs=50, lr=1e-3)\n",
                "    return rf, xc, lc, lstm_model, gat_model\n",
                "\n",
                "def predict_oof(models, X_tab, X_seq, X_grp):\n",
                "    \"\"\"Predict with trained models. No re-training!\"\"\"\n",
                "    rf, xc, lc, lstm_m, gat_m = models\n",
                "    n = X_tab.shape[0]\n",
                "    oof = np.zeros((n, 5))\n",
                "    oof[:,0] = rf.predict_proba(X_tab)[:,1]\n",
                "    oof[:,1] = xc.predict_proba(X_tab)[:,1]\n",
                "    oof[:,2] = lc.predict_proba(X_tab)[:,1]\n",
                "    oof[:,3] = lstm_m.predict(X_seq, batch_size=4096, verbose=0).ravel()\n",
                "    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "    with torch.no_grad():\n",
                "        x_g = torch.tensor(X_grp, dtype=torch.float32).to(dev)\n",
                "        oof[:,4] = gat_m(x_g).cpu().numpy().flatten()\n",
                "    print(f'    OOF shape: {oof.shape}')\n",
                "    return oof, lc\n",
            ]
            print(f'  [{ds}] oof_gen: restored full hyperparameters')

        # ============================================================
        # 2. Pipeline with META-HOLDOUT (no data leakage)
        # ============================================================
        if cid == 'run_pipeline':
            if is_paysim:
                stage1_call = "df_proc, scaler = stage1_preprocessing(df_raw)\ndel df_raw; import gc; gc.collect()"
                stage3_block = """print('\\n[Stage 3] Multi-View Feature Engineering...')
X_tab = extract_tabular_view(df_proc)
X_seq = extract_sequential_view(df_proc)
X_grp, gcols = extract_graph_view(df_proc)
y = df_proc['isFraud'].values
df_meta = df_proc[['amount']].copy() if 'amount' in df_proc.columns else None
del df_proc; gc.collect()
print('  Data arrays ready. Memory freed.')"""
                xai_features = "TABULAR_FEATURES"
                pipeline_name = "PaySim"
                meta_col = "'amount'"
            else:
                stage1_call = "df_proc, scaler = stage1_preprocessing(df_raw)\ndel df_raw; import gc; gc.collect()"
                stage3_block = """print('\\n[Stage 3] Multi-View Feature Engineering...')
selected = select_tabular_cols(df_proc)
X_tab = extract_tabular_view(df_proc, selected)
X_seq = extract_sequential_view(df_proc)
X_grp, gcols = extract_graph_view(df_proc)
y = df_proc['isFraud'].values
df_meta = df_proc[['TransactionAmt']].copy() if 'TransactionAmt' in df_proc.columns else None
del df_proc; gc.collect()
print('  Data arrays ready. Memory freed.')"""
                xai_features = "selected"
                pipeline_name = "IEEE-CIS"
                meta_col = "'TransactionAmt'"

            pipeline_code = f"""from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import time as _time

# ====== STAGE 1 ======
t_start = _time.time()
{stage1_call}

# ====== STAGE 3 ======
{stage3_block}

# ====== STAGE 2: 5-Block Walk-Forward CV ======
print('\\n[Stage 2] 5-Block Walk-Forward CV...')
tscv = TimeSeriesSplit(n_splits=5)
fold_metrics = []

for fold, (tr_i, vl_i) in enumerate(tscv.split(X_tab)):
    t_fold = _time.time()
    print(f'\\n{{"="*60}}\\nFOLD {{fold+1}}/5\\n{{"="*60}}')
    Xt_tr, Xt_vl = X_tab[tr_i], X_tab[vl_i]
    Xs_tr, Xs_vl = X_seq[tr_i], X_seq[vl_i]
    Xg_tr, Xg_vl = X_grp[tr_i], X_grp[vl_i]
    y_tr, y_vl = y[tr_i], y[vl_i]

    # === META-HOLDOUT SPLIT (80/20) ===
    # Prevents data leakage in stacking (Wolpert, 1992)
    cut = int(len(y_tr) * 0.8)
    print(f'  [Meta-Holdout] Base train: {{cut}}, Meta holdout: {{len(y_tr)-cut}}')

    # SMOTE on 80% base subset only
    print('  [SMOTE] K-Means SMOTE on base subset...')
    try:
        sm = KMeansSMOTE(cluster_balance_threshold=0.1, random_state=42)
        Xt_sm, y_sm = sm.fit_resample(Xt_tr[:cut], y_tr[:cut])
    except:
        Xt_sm, y_sm = SMOTE(random_state=42).fit_resample(Xt_tr[:cut], y_tr[:cut])
    print(f'    After SMOTE: {{Xt_sm.shape[0]}} (was {{cut}})')

    # Stage 4: Train ONCE on 80%
    print('  [Stage 4] Training models on base subset...')
    models = generate_oof_train(Xt_sm, y_sm, Xs_tr[:cut], y_tr[:cut], Xg_tr[:cut])

    # Predict on 20% HOLDOUT → HONEST predictions (no data leakage!)
    print('  [Stage 4] Predicting OOF on meta-holdout (HONEST)...')
    oof_hold, _ = predict_oof(models, Xt_tr[cut:], Xs_tr[cut:], Xg_tr[cut:])

    # Predict on VALIDATION
    print('  [Stage 4] Predicting OOF on validation...')
    oof_vl, best_lgb = predict_oof(models, Xt_vl, Xs_vl, Xg_vl)

    # Stage 5: Meta-learner trained on HONEST holdout predictions
    meta, meta_p = stage5_meta(oof_hold, y_tr[cut:], oof_vl)
    auprc = average_precision_score(y_vl, meta_p)
    roc = roc_auc_score(y_vl, meta_p)

    # Detailed metrics
    y_pred = (meta_p >= 0.5).astype(int)
    f1 = f1_score(y_vl, y_pred, zero_division=0)
    prec = precision_score(y_vl, y_pred, zero_division=0)
    rec = recall_score(y_vl, y_pred, zero_division=0)

    fold_time = _time.time() - t_fold
    fold_metrics.append({{'Fold': fold+1, 'AUPRC': auprc, 'ROC-AUC': roc,
                         'F1': f1, 'Precision': prec, 'Recall': rec,
                         'Time_min': fold_time/60}})
    print(f'  FOLD {{fold+1}}: AUPRC={{auprc:.4f}}, ROC-AUC={{roc:.4f}}, F1={{f1:.4f}}, P={{prec:.4f}}, R={{rec:.4f}} ({{fold_time/60:.1f}}min)')
    del Xt_sm, y_sm; gc.collect()

    if fold == 4:
        print(f'\\n{{"="*60}}\\nFOLD 5 (FINAL): Detailed Metrics + XAI\\n{{"="*60}}')
        print('\\n--- Classification Report ---')
        print(classification_report(y_vl, y_pred, target_names=['Normal', 'Fraud']))
        print('--- Confusion Matrix ---')
        cm = confusion_matrix(y_vl, y_pred)
        print(f'  TN={{cm[0,0]:,}}  FP={{cm[0,1]:,}}')
        print(f'  FN={{cm[1,0]:,}}  TP={{cm[1,1]:,}}')
        print(f'\\n--- Threshold Analysis ---')
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            yt = (meta_p >= t).astype(int)
            print(f'  t={{t:.1f}}: F1={{f1_score(y_vl, yt, zero_division=0):.4f}}, '
                  f'P={{precision_score(y_vl, yt, zero_division=0):.4f}}, '
                  f'R={{recall_score(y_vl, yt, zero_division=0):.4f}}')

        # XAI
        Xs = Xt_vl[:200]
        top3 = stage6_xai(best_lgb, Xt_tr[:2000], Xs, {xai_features})
        stage7_dual_output(meta_p, top3, y_vl)
        if df_meta is not None:
            stage8_hitl(meta_p, df_meta.iloc[vl_i])

# ====== SUMMARY ======
total_time = (_time.time() - t_start) / 60
print(f'\\n{{"="*60}}\\nMVS-XAI {pipeline_name} Pipeline Complete\\n{{"="*60}}')
mdf = pd.DataFrame(fold_metrics)
print(mdf.to_string(index=False))
print(f'\\nMean AUPRC:   {{mdf["AUPRC"].mean():.4f}} +/- {{mdf["AUPRC"].std():.4f}}')
print(f'Mean ROC-AUC: {{mdf["ROC-AUC"].mean():.4f}} +/- {{mdf["ROC-AUC"].std():.4f}}')
print(f'Mean F1:      {{mdf["F1"].mean():.4f}} +/- {{mdf["F1"].std():.4f}}')
print(f'Mean Prec:    {{mdf["Precision"].mean():.4f}} +/- {{mdf["Precision"].std():.4f}}')
print(f'Mean Recall:  {{mdf["Recall"].mean():.4f}} +/- {{mdf["Recall"].std():.4f}}')
print(f'Total runtime: {{total_time:.1f}} min')
"""
            c['source'] = [line + '\n' for line in pipeline_code.split('\n')]
            print(f'  [{ds}] run_pipeline: META-HOLDOUT applied')

    with open(file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f'[SAVED] {file}')

print('\n=== DONE: Both notebooks now use Meta-Holdout Stacking ===')

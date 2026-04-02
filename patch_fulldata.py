import json

with open('MVS_XAI_Colab_IEEE_CIS.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    cell_id = cell.get('metadata', {}).get('id', '')

    # ============================================================
    # 1. DATA LOADING: Full rows + float32
    # ============================================================
    if cell_id == 'main_load':
        new_source = """# ====== DATA LOADING & EXTRACTION (ZIP MODE — FULL DATA) ======
import os
import pandas as pd
import numpy as np
import zipfile
import glob
import gc

ZIP_PATH = '/content/drive/MyDrive/MVS_XAI_Data/ieee-fraud-detection.zip'
EXTRACT_DIR = '/content/ieee-dataset'

print('--- Full Data Loading (IEEE-CIS 590K) ---')
print(f'Checking for ZIP file: {ZIP_PATH}')

df_txn = None
df_id = None

if os.path.exists(ZIP_PATH):
    print('\\u2705 Found ZIP file! Extracting to local Colab storage...')
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print('\\u2705 Extraction complete!')

    txn_files = glob.glob(f'{EXTRACT_DIR}/**/train_transaction.csv', recursive=True)
    id_files  = glob.glob(f'{EXTRACT_DIR}/**/train_identity.csv', recursive=True)

    if txn_files:
        print(f'\\u2705 Loading FULL train_transaction (no row limit)...')
        df_txn = pd.read_csv(txn_files[0])
        # Convert to float32 immediately to save ~50% RAM
        for col in df_txn.select_dtypes(include=['float64']).columns:
            df_txn[col] = df_txn[col].astype(np.float32)
        print(f'    Shape: {df_txn.shape}, Memory: {df_txn.memory_usage(deep=True).sum()/1e9:.2f} GB')
    else:
        print('\\u274c train_transaction.csv not found in ZIP!')

    if id_files:
        print(f'\\u2705 Loading train_identity...')
        df_id = pd.read_csv(id_files[0])
        for col in df_id.select_dtypes(include=['float64']).columns:
            df_id[col] = df_id[col].astype(np.float32)
    else:
        print('\\u274c train_identity.csv not found in ZIP!')
else:
    print('\\u274c ZIP file MISSING at MVS_XAI_Data/ieee-fraud-detection.zip')

print('\\n============================================================')
if df_txn is not None:
    if df_id is not None:
        df_raw = df_txn.merge(df_id, on='TransactionID', how='left')
        del df_txn, df_id
        gc.collect()
        print(f'Merged data shape: {df_raw.shape}')
    else:
        df_raw = df_txn
        del df_txn
        gc.collect()
        print(f'Transaction-only shape: {df_raw.shape}')
    print(f'Total Memory: {df_raw.memory_usage(deep=True).sum()/1e9:.2f} GB')
else:
    print('IEEE-CIS dataset NOT FOUND! Falling back to synthetic data...')
    print('============================================================\\n')
    np.random.seed(42)
    N = 20000
    df_raw = pd.DataFrame({
        'TransactionID': range(N),
        'TransactionDT': np.sort(np.random.randint(100000, 16000000, N)),
        'TransactionAmt': np.random.uniform(1, 5000, N).astype(np.float32),
        'ProductCD': np.random.choice(['W','H','C','S','R'], N),
        'card1': np.random.randint(1000, 9999, N),
        'card2': np.random.randint(100, 600, N).astype(np.float32),
        'card3': np.random.randint(100, 250, N).astype(np.float32),
        'card4': np.random.choice(['visa','mastercard','discover'], N),
        'card5': np.random.randint(100, 300, N).astype(np.float32),
        'card6': np.random.choice(['debit','credit'], N),
        'addr1': np.random.randint(100, 500, N).astype(np.float32),
        'addr2': np.random.randint(1, 100, N).astype(np.float32),
        'P_emaildomain': np.random.choice(['gmail.com','yahoo.com','outlook.com', np.nan], N),
        'R_emaildomain': np.random.choice(['gmail.com','yahoo.com', np.nan], N),
        'isFraud': np.random.choice([0, 1], N, p=[0.965, 0.035])
    })
    print(f'Synthetic Dataset shape: {df_raw.shape}\\n')

print(df_raw.head())
"""
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        print('[DONE] Data loading: full rows + float32')

    # ============================================================
    # 2. TABULAR FEATURE SELECTION: 53 -> ~120 (add V-columns)
    # ============================================================
    if cell_id == 'stage3_setup':
        new_source = """def select_tabular_cols(df):
    \"\"\"Select top ~120 features: base + C/D/M + top V-columns (missing < 50%).\"\"\"
    base = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
    C_cols = [c for c in df.columns if c.startswith('C') and c[1:].isdigit()]
    D_cols = [c for c in df.columns if c.startswith('D') and c[1:].isdigit()]
    M_cols = [c for c in df.columns if c.startswith('M') and c[1:].isdigit()]
    extra = ['LogAmt', 'Hour', 'DayOfWeek']

    # Add top V-columns (missing rate < 50%) — these are the most powerful fraud features
    V_cols = [c for c in df.columns if c.startswith('V') and c[1:].isdigit()]
    V_good = [c for c in V_cols if df[c].isna().mean() < 0.50]
    # Sort by missing rate (least missing first) and take top 70
    V_good = sorted(V_good, key=lambda c: df[c].isna().mean())[:70]

    all_cols = [c for c in base + C_cols + D_cols + M_cols + extra + V_good if c in df.columns]
    print(f'    Selected {len(all_cols)} features ({len(V_good)} V-columns included)')
    return all_cols

def extract_tabular_view(df, cols):
    print(f'  [View 1] Tabular ({len(cols)} features)')
    return df[cols].values.astype(np.float32)
"""
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        print('[DONE] Tabular features: +70 V-columns')

    # ============================================================
    # 3. SEQUENTIAL VIEW: Add more features for LSTM
    # ============================================================
    if cell_id == 'stage3_seq':
        new_source = """SEQ_WINDOW = 10

def extract_sequential_view(df, window=SEQ_WINDOW):
    \"\"\"T=10 sliding window per AccountID. Right-padded zeros for cuDNN.\"\"\"
    print(f'  [View 2] Sequential (T={window} per AccountID)...')
    # Expanded features for stronger LSTM signal
    candidates = ['TransactionAmt', 'LogAmt', 'Hour', 'C1', 'C2', 'D1', 'D2']
    seq_feats = [c for c in candidates if c in df.columns]
    n_feat = len(seq_feats)
    print(f'    Seq features ({n_feat}): {seq_feats}')
    seq = np.zeros((len(df), window, n_feat), dtype=np.float32)
    for _, grp in df.groupby('AccountID'):
        vals = grp[seq_feats].values.astype(np.float32)
        idxs = grp.index.values
        for pos, idx in enumerate(idxs):
            start = max(0, pos - window + 1)
            s = vals[start:pos+1]
            seq[idx, :len(s), :] = s
    print(f'    Shape: {seq.shape}')
    return seq
"""
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        print('[DONE] Sequential view: 3->7 features')

    # ============================================================
    # 4. OOF GENERATION: XGB+LGB tuning for more features
    # ============================================================
    if cell_id == 'oof_gen':
        new_source = """def generate_oof(X_tab_tr, y_smote, X_tab_vl,
                 X_seq_tr, X_seq_vl, y_raw,
                 X_grp_tr, X_grp_vl):
    \"\"\"OOF matrix [N_val x 5]. Trees on SMOTE, DL on raw + Focal Loss.\"\"\"
    n = X_tab_vl.shape[0]
    oof = np.zeros((n, 5))

    print('    [RF] Training...'); rf = RandomForestClassifier(100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_tab_tr, y_smote); oof[:,0] = rf.predict_proba(X_tab_vl)[:,1]

    print('    [XGB] Training...'); xc = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        tree_method='hist', device='cuda',
        colsample_bytree=0.7, subsample=0.8,
        random_state=42, n_jobs=-1)
    xc.fit(X_tab_tr, y_smote); oof[:,1] = xc.predict_proba(X_tab_vl)[:,1]

    print('    [LGB] Training...'); lc = lgb.LGBMClassifier(
        n_estimators=500, max_depth=12, learning_rate=0.05,
        colsample_bytree=0.7, subsample=0.8,
        random_state=42, n_jobs=-1, verbose=-1)
    lc.fit(X_tab_tr, y_smote); oof[:,2] = lc.predict_proba(X_tab_vl)[:,1]

    oof[:,3] = train_lstm_oof(X_seq_tr, y_raw, X_seq_vl)
    oof[:,4] = train_gat_oof(X_grp_tr, y_raw, X_grp_vl)

    print(f'    OOF shape: {oof.shape}')
    return oof, lc
"""
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        print('[DONE] XGB/LGB: colsample=0.7, subsample=0.8')

    # ============================================================
    # 5. STAGE 1: Add gc.collect() + float32 enforcement
    # ============================================================
    if cell_id == 'stage1':
        src = ''.join(cell['source'])
        # Add float32 cast and gc at end of function
        old_return = "    return df, scaler"
        new_return = """    # Cast all numerics to float32 to save RAM
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    import gc; gc.collect()
    print(f'  Memory after preprocessing: {df.memory_usage(deep=True).sum()/1e9:.2f} GB')
    return df, scaler"""
        src = src.replace(old_return, new_return)
        cell['source'] = [line + '\n' for line in src.split('\n')]
        if cell['source'][-1].strip() == '':
            cell['source'] = cell['source'][:-1]
        print('[DONE] Stage 1: float32 cast + gc.collect()')

    # ============================================================
    # 6. PIPELINE: Add gc.collect() between folds
    # ============================================================
    if cell_id == 'run_pipeline':
        src = ''.join(cell['source'])
        old_fold_end = "    print(f'  FOLD {fold+1}: AUPRC={auprc:.4f}, ROC-AUC={roc:.4f}')"
        new_fold_end = """    print(f'  FOLD {fold+1}: AUPRC={auprc:.4f}, ROC-AUC={roc:.4f}')
    # Free memory between folds
    import gc; gc.collect()"""
        src = src.replace(old_fold_end, new_fold_end)
        cell['source'] = [line + '\n' for line in src.split('\n')]
        if cell['source'][-1].strip() == '':
            cell['source'] = cell['source'][:-1]
        print('[DONE] Pipeline: gc.collect() between folds')

with open('MVS_XAI_Colab_IEEE_CIS.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('\n=== All full-data optimizations applied to IEEE-CIS notebook! ===')

import json

with open('MVS_XAI_Colab_DataPrep_Phase1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    cell_id = cell.get('metadata', {}).get('id', '')

    # ============================================================
    # 1. DATA LOADING: float32 + smart sampling for 6.3M rows
    # ============================================================
    if cell_id == 'main_load':
        new_source = """# ====== DATA LOADING & EXTRACTION (ZIP MODE — OPTIMIZED) ======
import os
import pandas as pd
import numpy as np
import zipfile
import glob
import gc

ZIP_PATH = '/content/drive/MyDrive/MVS_XAI_Data/paysim1.zip'
EXTRACT_DIR = '/content/paysim-dataset'

# PaySim has 6.3M rows — use stratified sample for Colab RAM
NROWS = 1000000  # 1M rows (balance between coverage & RAM)

print('--- Optimized Data Loading (PaySim) ---')
print(f'Checking for ZIP file: {ZIP_PATH}')

df_raw = None

if os.path.exists(ZIP_PATH):
    print('\\u2705 Found ZIP file! Extracting to local Colab storage...')
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print('\\u2705 Extraction complete!')

    csv_files = glob.glob(f'{EXTRACT_DIR}/**/*.csv', recursive=True)

    if csv_files:
        print(f'\\u2705 Found PaySim CSV at: {csv_files[0]}')
        df_full = pd.read_csv(csv_files[0])
        print(f'    Full dataset shape: {df_full.shape}')
        print(f'    Fraud count: {df_full["isFraud"].sum()} ({df_full["isFraud"].mean():.4%})')

        # Stratified sampling: keep ALL fraud + random sample of normal
        if len(df_full) > NROWS:
            fraud_df = df_full[df_full['isFraud'] == 1]
            normal_df = df_full[df_full['isFraud'] == 0].sample(
                n=min(NROWS - len(fraud_df), len(df_full[df_full['isFraud'] == 0])),
                random_state=42
            )
            df_raw = pd.concat([fraud_df, normal_df]).sort_values('step').reset_index(drop=True)
            del df_full, fraud_df, normal_df
            gc.collect()
            print(f'    Stratified sample: {df_raw.shape} (ALL fraud kept)')
        else:
            df_raw = df_full
            del df_full
            gc.collect()

        # Convert to float32
        for col in df_raw.select_dtypes(include=['float64']).columns:
            df_raw[col] = df_raw[col].astype(np.float32)
        print(f'    Memory: {df_raw.memory_usage(deep=True).sum()/1e9:.2f} GB')
    else:
        print('\\u274c No CSV files found inside the ZIP!')
else:
    print('\\u274c ZIP file MISSING at MVS_XAI_Data/paysim1.zip')

if df_raw is None:
    print('PaySim dataset NOT FOUND! Falling back to synthetic data...')
    print('============================================================\\n')
    np.random.seed(42)
    N = 20000
    df_raw = pd.DataFrame({
        'step': np.sort(np.random.randint(1, 500, N)),
        'type': np.random.choice(['TRANSFER', 'CASH_OUT'], N),
        'amount': np.random.uniform(10, 50000, N).astype(np.float32),
        'nameOrig': ['C'+str(i) for i in np.random.randint(0, 2000, N)],
        'oldbalanceOrg': np.random.uniform(0, 100000, N).astype(np.float32),
        'newbalanceOrig': np.random.uniform(0, 100000, N).astype(np.float32),
        'nameDest': ['C'+str(i) for i in np.random.randint(2000, 4000, N)],
        'oldbalanceDest': np.random.uniform(0, 100000, N).astype(np.float32),
        'newbalanceDest': np.random.uniform(0, 100000, N).astype(np.float32),
        'isFraud': np.random.choice([0, 1], N, p=[0.95, 0.05])
    })
    print(f'Synthetic Dataset shape: {df_raw.shape}\\n')

print(f'\\nFinal dataset: {df_raw.shape}')
print(f'Fraud ratio: {df_raw["isFraud"].mean():.4%}')
print(df_raw.head())
"""
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        print('[DONE] Data loading: stratified sample 1M + float32')

    # ============================================================
    # 2. STAGE 1: Add float32 cast + gc
    # ============================================================
    if cell_id == 'stage1':
        src = ''.join(cell['source'])
        old_return = "    return df, scaler"
        new_return = """    # Cast to float32 to save RAM
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    import gc; gc.collect()
    print(f'  Memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB')
    return df, scaler"""
        src = src.replace(old_return, new_return)
        cell['source'] = [line + '\n' for line in src.split('\n')]
        if cell['source'][-1].strip() == '':
            cell['source'] = cell['source'][:-1]
        print('[DONE] Stage 1: float32 + gc.collect()')

    # ============================================================
    # 3. TABULAR VIEW: adapt for PaySim columns
    # ============================================================
    if cell_id == 'stage3_tabular':
        new_source = """TABULAR_FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrg',
    'oldbalanceDest', 'newbalanceDest', 'errorBalanceDest',
    'type_encoded', 'amountToOldBalanceRatio'
]

def extract_tabular_view(df):
    cols = [c for c in TABULAR_FEATURES if c in df.columns]
    print(f'  [View 1] Tabular ({len(cols)} features)')
    return df[cols].values.astype(np.float32)
"""
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        print('[DONE] Tabular view: float32, safe column check')

    # ============================================================
    # 4. SEQUENTIAL VIEW: Add more features
    # ============================================================
    if cell_id == 'stage3_sequential':
        new_source = """SEQ_WINDOW = 10

def extract_sequential_view(df, window=SEQ_WINDOW):
    \"\"\"T=10 sliding window per account. RIGHT-padded zeros for cuDNN.\"\"\"
    print(f'  [View 2] Sequential (T={window} sliding window)...')
    candidates = ['amount', 'errorBalanceOrg', 'errorBalanceDest', 
                  'amountToOldBalanceRatio', 'oldbalanceOrg', 'type_encoded']
    seq_feats = [c for c in candidates if c in df.columns]
    n_feat = len(seq_feats)
    print(f'    Seq features ({n_feat}): {seq_feats}')
    seq = np.zeros((len(df), window, n_feat), dtype=np.float32)
    for _, grp in df.groupby('nameOrig'):
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
        print('[DONE] Sequential view: 3->6 features')

    # ============================================================
    # 5. OOF GENERATION: Add colsample/subsample
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
        print('[DONE] OOF: XGB/LGB colsample=0.7, subsample=0.8')

    # ============================================================
    # 6. PIPELINE: Add gc.collect() between folds
    # ============================================================
    if cell_id == 'run_pipeline':
        src = ''.join(cell['source'])
        old_fold_end = "    print(f'  FOLD {fold+1}: AUPRC={auprc:.4f}, ROC-AUC={roc:.4f}')"
        new_fold_end = """    print(f'  FOLD {fold+1}: AUPRC={auprc:.4f}, ROC-AUC={roc:.4f}')
    import gc; gc.collect()"""
        src = src.replace(old_fold_end, new_fold_end)
        cell['source'] = [line + '\n' for line in src.split('\n')]
        if cell['source'][-1].strip() == '':
            cell['source'] = cell['source'][:-1]
        print('[DONE] Pipeline: gc.collect() between folds')

with open('MVS_XAI_Colab_DataPrep_Phase1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('\n=== PaySim notebook fully optimized! ===')

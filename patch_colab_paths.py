import json

def patch_notebook(fp, dataset_type):
    with open(fp, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and cell.get('metadata', {}).get('id') == 'main_load':
            if dataset_type == 'ieee':
                new_source = f"""# ====== DATA LOADING & EXTRACTION (ZIP MODE) ======
import os
import pandas as pd
import numpy as np
import zipfile
import glob

NROWS = 200000  # Memory-safe for Colab free tier

ZIP_PATH = '/content/drive/MyDrive/MVS_XAI_Data/ieee-fraud-detection.zip'
EXTRACT_DIR = '/content/ieee-dataset'

print('--- Fast ZIP Data Loading (IEEE-CIS) ---')
print(f'Checking for ZIP file: {{ZIP_PATH}}')

df_txn = None
df_id = None

if os.path.exists(ZIP_PATH):
    print('\\u2705 Found ZIP file! Extracting to local Colab storage...')
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print('\\u2705 Extraction complete!')
    
    # Locate the CSVs (sometimes they extract directly, sometimes inside a folder)
    txn_files = glob.glob(f'{{EXTRACT_DIR}}/**/train_transaction.csv', recursive=True)
    id_files  = glob.glob(f'{{EXTRACT_DIR}}/**/train_identity.csv', recursive=True)
    
    if txn_files:
        print(f'\\u2705 Found train_transaction at: {{txn_files[0]}}')
        df_txn = pd.read_csv(txn_files[0], nrows=NROWS)
    else:
        print('\\u274c train_transaction.csv not found in ZIP!')
        
    if id_files:
        print(f'\\u2705 Found train_identity at: {{id_files[0]}}')
        df_id = pd.read_csv(id_files[0])
    else:
        print('\\u274c train_identity.csv not found in ZIP!')
else:
    print('\\u274c ZIP file MISSING at MVS_XAI_Data/ieee-fraud-detection.zip')

print('\\n============================================================')
if df_txn is not None:
    if df_id is not None:
        df_raw = df_txn.merge(df_id, on='TransactionID', how='left')
        print(f'Merged data shape: {{df_raw.shape}}')
    else:
        df_raw = df_txn
        print(f'Transaction-only shape: {{df_raw.shape}} (identity NOT found)')
else:
    print('IEEE-CIS dataset NOT FOUND! Falling back to synthetic data...')
    print('============================================================\\n')
    np.random.seed(42)
    N = 20000
    df_raw = pd.DataFrame({{
        'TransactionID': range(N),
        'TransactionDT': np.sort(np.random.randint(100000, 16000000, N)),
        'TransactionAmt': np.random.uniform(1, 5000, N),
        'ProductCD': np.random.choice(['W','H','C','S','R'], N),
        'card1': np.random.randint(1000, 9999, N),
        'card2': np.random.randint(100, 600, N).astype(float),
        'card3': np.random.randint(100, 250, N).astype(float),
        'card4': np.random.choice(['visa','mastercard','discover'], N),
        'card5': np.random.randint(100, 300, N).astype(float),
        'card6': np.random.choice(['debit','credit'], N),
        'addr1': np.random.randint(100, 500, N).astype(float),
        'addr2': np.random.randint(1, 100, N).astype(float),
        'P_emaildomain': np.random.choice(['gmail.com','yahoo.com','outlook.com', np.nan], N),
        'R_emaildomain': np.random.choice(['gmail.com','yahoo.com', np.nan], N),
        'isFraud': np.random.choice([0, 1], N, p=[0.965, 0.035])
    }})
    print(f'Synthetic Dataset shape: {{df_raw.shape}}\\n')

print(df_raw.head())
"""
            else:
                new_source = f"""# ====== DATA LOADING & EXTRACTION (ZIP MODE) ======
import os
import pandas as pd
import numpy as np
import zipfile
import glob

ZIP_PATH = '/content/drive/MyDrive/MVS_XAI_Data/paysim1.zip'
EXTRACT_DIR = '/content/paysim-dataset'

print('--- Fast ZIP Data Loading (PaySim) ---')
print(f'Checking for ZIP file: {{ZIP_PATH}}')

df_raw = None

if os.path.exists(ZIP_PATH):
    print('\\u2705 Found ZIP file! Extracting to local Colab storage...')
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print('\\u2705 Extraction complete!')
    
    # Locate the CSV (usually PS_20174392719_1491204439457_log.csv or similar)
    csv_files = glob.glob(f'{{EXTRACT_DIR}}/**/*.csv', recursive=True)
    
    if csv_files:
        print(f'\\u2705 Found PaySim CSV at: {{csv_files[0]}}')
        df_raw = pd.read_csv(csv_files[0])
        print(f'\\nDataset shape: {{df_raw.shape}}')
    else:
        print('\\u274c No CSV files found inside the ZIP!')
else:
    print('\\u274c ZIP file MISSING at MVS_XAI_Data/paysim1.zip')

if df_raw is None:
    print('PaySim dataset NOT FOUND! Falling back to synthetic data...')
    print('============================================================\\n')
    np.random.seed(42)
    N = 20000
    df_raw = pd.DataFrame({{
        'step': np.sort(np.random.randint(1, 500, N)),
        'type': np.random.choice(['TRANSFER', 'CASH_OUT'], N),
        'amount': np.random.uniform(10, 50000, N),
        'nameOrig': ['C'+str(i) for i in np.random.randint(0, 2000, N)],
        'oldbalanceOrg': np.random.uniform(0, 100000, N),
        'newbalanceOrig': np.random.uniform(0, 100000, N),
        'nameDest': ['C'+str(i) for i in np.random.randint(2000, 4000, N)],
        'oldbalanceDest': np.random.uniform(0, 100000, N),
        'newbalanceDest': np.random.uniform(0, 100000, N),
        'isFraud': np.random.choice([0, 1], N, p=[0.95, 0.05])
    }})
    print(f'Synthetic Dataset shape: {{df_raw.shape}}\\n')

print(df_raw.head())
"""
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

patch_notebook('MVS_XAI_Colab_IEEE_CIS.ipynb', 'ieee')
patch_notebook('MVS_XAI_Colab_DataPrep_Phase1.ipynb', 'paysim')
print("Notebooks patched heavily successfully.")

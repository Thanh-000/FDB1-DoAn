import json

# ===================== FIX PaySim Notebook =====================
with open('MVS_XAI_Colab_DataPrep_Phase1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and cell.get('metadata', {}).get('id') == 'main':
        new_source = """# ====== DATA LOADING & EXTRACTION (ZIP MODE) ======
import os
import pandas as pd
import numpy as np
import zipfile
import glob

ZIP_PATH = '/content/drive/MyDrive/MVS_XAI_Data/paysim1.zip'
EXTRACT_DIR = '/content/paysim-dataset'

print('--- Fast ZIP Data Loading (PaySim) ---')
print(f'Checking for ZIP file: {ZIP_PATH}')

df_raw = None

if os.path.exists(ZIP_PATH):
    print('\\u2705 Found ZIP file! Extracting to local Colab storage...')
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print('\\u2705 Extraction complete!')
    
    # Locate the CSV
    csv_files = glob.glob(f'{EXTRACT_DIR}/**/*.csv', recursive=True)
    
    if csv_files:
        print(f'\\u2705 Found PaySim CSV at: {csv_files[0]}')
        df_raw = pd.read_csv(csv_files[0])
        print(f'\\nDataset shape: {df_raw.shape}')
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
        'amount': np.random.uniform(10, 50000, N),
        'nameOrig': ['C'+str(i) for i in np.random.randint(0, 2000, N)],
        'oldbalanceOrg': np.random.uniform(0, 100000, N),
        'newbalanceOrig': np.random.uniform(0, 100000, N),
        'nameDest': ['C'+str(i) for i in np.random.randint(2000, 4000, N)],
        'oldbalanceDest': np.random.uniform(0, 100000, N),
        'newbalanceDest': np.random.uniform(0, 100000, N),
        'isFraud': np.random.choice([0, 1], N, p=[0.95, 0.05])
    })
    print(f'Synthetic Dataset shape: {df_raw.shape}\\n')

print(df_raw.head())
"""
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        # Also rename the cell id to main_load for consistency
        cell['metadata']['id'] = 'main_load'
        print("PaySim data cell FIXED!")

with open('MVS_XAI_Colab_DataPrep_Phase1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("PaySim notebook saved successfully.")

import json

with open('MVS_XAI_Colab_DataPrep_Phase1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    cell_id = cell.get('metadata', {}).get('id', '')

    # FIX 1: Sequential view — RIGHT-padded for cuDNN
    if cell_id == 'stage3_sequential':
        cell['source'] = [
            "SEQ_WINDOW = 10\n",
            "\n",
            "def extract_sequential_view(df, window=SEQ_WINDOW):\n",
            "    import time; t0 = time.time()\n",
            "    print(f'  [View 2] Sequential (T={window}, VECTORIZED, RIGHT-padded)...')\n",
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
            "            # RIGHT-padded: data at positions 0..len-1, zeros at len..window-1\n",
            "            seq[df_work['_oidx'].values, t, i] = shifted.fillna(0).values\n",
            "        print(f'      Feature {i+1}/{n_feat} done')\n",
            "    del df_work; import gc; gc.collect()\n",
            "    print(f'    Shape: {seq.shape} ({time.time()-t0:.1f}s)')\n",
            "    return seq\n",
        ]
        print('[DONE] Sequential: RIGHT-padded (cuDNN compatible)')

with open('MVS_XAI_Colab_DataPrep_Phase1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('=== Fix applied! ===')

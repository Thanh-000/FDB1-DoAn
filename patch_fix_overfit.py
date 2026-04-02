import json

for file in ['MVS_XAI_Colab_DataPrep_Phase1.ipynb', 'MVS_XAI_Colab_IEEE_CIS.ipynb']:
    with open(file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for c in nb['cells']:
        if c['cell_type'] != 'code': continue
        cid = c.get('metadata', {}).get('id', '')
        if cid == 'oof_gen':
            src = ''.join(c['source'])
            
            # Revert to stable hyperparameters to prevent Meta-Learner overfitting
            src = src.replace('RandomForestClassifier(100, max_depth=15', 'RandomForestClassifier(100, max_depth=10')
            
            src = src.replace('xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05', 
                              'xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1')
            
            src = src.replace('lgb.LGBMClassifier(n_estimators=500, max_depth=12, learning_rate=0.05',
                              'lgb.LGBMClassifier(n_estimators=150, max_depth=8, learning_rate=0.1')
            
            src = src.replace('lstm_model.fit(X_seq_tr, y_raw, epochs=15', 
                              'lstm_model.fit(X_seq_tr, y_raw, epochs=10')
            
            src = src.replace('gat_model = train_gat_oof_model(X_grp_tr, y_raw, epochs=50, lr=1e-3)',
                              'gat_model = train_gat_oof_model(X_grp_tr, y_raw, epochs=20, lr=5e-3)')
                              
            c['source'] = [line + '\n' for line in src.split('\n')]
            if c['source'][-1].strip() == '':
                c['source'] = c['source'][:-1]
                
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f'[SAVED] {file}')

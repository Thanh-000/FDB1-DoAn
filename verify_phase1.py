import json

files = ['MVS_XAI_Colab_IEEE_CIS.ipynb', 'MVS_XAI_Colab_DataPrep_Phase1.ipynb']

for fname in files:
    print(f"\n=== {fname} ===")
    nb = json.load(open(fname, encoding='utf-8'))
    full_code = ''
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            full_code += ''.join(cell.get('source', []))
    
    checks = {
        'generate_oof_train(X_tab_tr, y_tr)': 'def generate_oof_train(X_tab_tr, y_tr)' in full_code,
        'predict_oof(models, X_tab) [2 args]': 'def predict_oof(models, X_tab)' in full_code,
        'OOF [Nx4]': 'np.zeros((n, 4))' in full_code,
        '4 model names (no BiLSTM)': "['RF','XGB','LGB','CatBoost']" in full_code,
        'XGB Meta-Learner': 'XGBClassifier(n_estimators=50' in full_code,
        'No BiLSTM in training': 'BiLSTM' not in full_code.split('def predict_oof')[0].split('def generate_oof_train')[1] if 'def predict_oof' in full_code else False,
        'No LogisticRegression Meta': 'LogisticRegression' not in full_code,
        'Call: generate_oof_train(Xt_tr[:cut], y_tr[:cut])': 'generate_oof_train(Xt_tr[:cut], y_tr[:cut])' in full_code,
        'Call: predict_oof(models, Xt_tr[cut:])': 'predict_oof(models, Xt_tr[cut:])' in full_code,
        'Call: predict_oof(models, Xt_vl)': 'predict_oof(models, Xt_vl)' in full_code,
        'scale_pos_weight in XGB': 'scale_pos_weight=ratio,' in full_code,
        'scale_pos_weight in Meta': 'scale_pos_weight=ratio_meta' in full_code,
        'CatBoost present': 'train_catboost' in full_code,
        'drive.mount present': 'drive.mount' in full_code,
        'No SMOTE/imblearn': 'imblearn' not in full_code,
        'feature_importances_ in viz': 'meta.feature_importances_' in full_code,
    }
    
    issues = []
    for name, passed in checks.items():
        status = '✅' if passed else '❌'
        print(f"  {status} {name}")
        if not passed:
            issues.append(name)
    
    if issues:
        print(f"\n  ⚠️ ISSUES ({len(issues)}): {issues}")
    else:
        print(f"\n  🎉 ALL {len(checks)} CHECKS PASSED!")

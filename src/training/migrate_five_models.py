"""One-time migration helper: replace the old Decision Tree comparison with SVM.
Uses the existing trained pipelines, fits/tunes SVM, and refreshes comparison artifacts.
Run from the project root: python -m src.training.migrate_five_models
"""
import json
import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data.load_data import load_raw_data
from src.data.preprocess_data import split_data, create_preprocessor_pipeline
from src.features.build_features import FEATURE_NAMES
from src.training.evaluate import evaluate_classification_model, compute_permutation_feature_importance
from src.training.train import calculate_clinical_selection_score, extract_native_feature_importance

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'


def main():
    raw_df, data_hash = load_raw_data(auto_download=False)
    X_train, X_test, y_train, y_test = split_data(raw_df, test_size=0.20, random_state=42)
    all_models_path = MODELS / 'all_trained_models.joblib'
    old_models = joblib.load(all_models_path)
    # Keep only the requested algorithms, preserving the already-trained pipelines.
    models = {name: old_models[name] for name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost'] if name in old_models}
    if len(models) != 4:
        raise RuntimeError(f'Expected four existing pipelines, found {list(models)}')

    pipe = Pipeline([
        ('preprocessor', create_preprocessor_pipeline(with_scaling=True)),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    grid = GridSearchCV(
        pipe,
        {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__class_weight': [None, 'balanced'],
        },
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    models['SVM'] = grid.best_estimator_
    models = {name: models[name] for name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'XGBoost']}
    svm_tuning = {'best_params': grid.best_params_, 'best_cv_score': round(float(grid.best_score_), 4)}

    old_meta = json.loads((MODELS / 'model_metadata.json').read_text(encoding='utf-8'))
    old_eval = json.loads((REPORTS / 'evaluation_summary.json').read_text(encoding='utf-8'))
    old_cal = json.loads((REPORTS / 'calibration_summary.json').read_text(encoding='utf-8'))
    old_fi = json.loads((REPORTS / 'feature_importance.json').read_text(encoding='utf-8'))
    old_tuning = {name: old_meta.get('models_evaluated', {}).get(name, {}).get('tuning_parameters', {}) for name in models if name != 'SVM'}
    old_tuning['SVM'] = svm_tuning

    new_meta = dict(old_meta)
    new_meta['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    new_meta['dataset_hash'] = data_hash
    new_meta['models_evaluated'] = {}
    # Existing serialized models were produced by the project's recorded sklearn version.
    # Preserve their recorded evaluation metrics rather than re-fitting cross-version artifacts.
    new_eval = {name: old_eval[name] for name in models if name != 'SVM' and name in old_eval}
    new_cal = {name: old_cal[name] for name in models if name != 'SVM' and name in old_cal}
    new_fi = {name: old_fi[name] for name in models if name != 'SVM' and name in old_fi}
    records = []
    scores = {}
    for name, model in models.items():
        if name == 'SVM':
            test_res = evaluate_classification_model(model, X_test, y_test, model_name=name)
            new_eval[name] = test_res
            prob = model.predict_proba(X_test)[:, 1]
            from sklearn.calibration import calibration_curve
            frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy='uniform')
            new_cal[name] = {'fraction_of_positives': frac_pos.tolist(), 'mean_predicted_value': mean_pred.tolist(), 'brier_score': test_res['brier_score']}
            new_fi[name] = {
                'permutation_importance': compute_permutation_feature_importance(model, X_test, y_test, FEATURE_NAMES, n_repeats=5, random_state=42),
                'native_importance': extract_native_feature_importance(model, FEATURE_NAMES),
            }
            test_metrics = {k: test_res[k] for k in ['accuracy','balanced_accuracy','precision','recall','specificity','f1_score','roc_auc','pr_auc','brier_score']}
            cv_mean = float(grid.best_score_)
            cv_std = float(grid.cv_results_['std_test_score'][grid.best_index_])
            cv_summary = {'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'roc_auc': cv_mean}
            score = calculate_clinical_selection_score(test_res)
            record = {
                'Model': name, 'Accuracy': test_res['accuracy'], 'Balanced Acc': test_res['balanced_accuracy'],
                'Precision': test_res['precision'], 'Recall (Sensitivity)': test_res['recall'],
                'Specificity': test_res['specificity'], 'F1-Score': test_res['f1_score'],
                'ROC-AUC': test_res['roc_auc'], 'PR-AUC': test_res['pr_auc'], 'Brier Score': test_res['brier_score'],
                'CV ROC-AUC (Mean±Std)': f"{round(cv_mean, 4)} ± {round(cv_std, 4)}", 'Selection Score': score,
            }
        else:
            old_entry = old_meta.get('models_evaluated', {}).get(name, {})
            test_metrics = old_entry.get('test_metrics', {})
            cv_summary = old_entry.get('cv_metrics_mean', {})
            score = float(old_entry.get('selection_score', 0.0))
            old_record = next((r for r in pd.read_csv(REPORTS / 'evaluation_summary.csv').to_dict(orient='records') if r.get('Model') == name), None)
            if old_record is None:
                raise RuntimeError(f'Missing saved metrics row for {name}')
            record = old_record
        scores[name] = score
        new_meta['models_evaluated'][name] = {
            'tuning_parameters': old_tuning.get(name, {}),
            'cv_metrics_mean': cv_summary,
            'test_metrics': test_metrics,
            'selection_score': score,
            'is_best': False,
        }
        records.append(record)

    # Keep the existing default model unless SVM has a better recorded selection score.
    best_name = max(scores, key=scores.get)
    new_meta['selected_model_name'] = best_name
    new_meta['clinical_selection_score'] = scores[best_name]
    for name in models:
        new_meta['models_evaluated'][name]['is_best'] = name == best_name
    joblib.dump(models, all_models_path)
    joblib.dump(models[best_name], MODELS / 'medipredict_best_model.joblib')
    (MODELS / 'model_metadata.json').write_text(json.dumps(new_meta, indent=2), encoding='utf-8')
    (REPORTS / 'evaluation_summary.json').write_text(json.dumps(new_eval, indent=2), encoding='utf-8')
    (REPORTS / 'calibration_summary.json').write_text(json.dumps(new_cal, indent=2), encoding='utf-8')
    (REPORTS / 'feature_importance.json').write_text(json.dumps(new_fi, indent=2), encoding='utf-8')
    pd.DataFrame(records).to_csv(REPORTS / 'evaluation_summary.csv', index=False)
    print('Five-model migration complete. Models:', ', '.join(models))
    print('Selected default model:', best_name)
    print(pd.DataFrame(records)[['Model','Accuracy','Precision','Recall (Sensitivity)','F1-Score','ROC-AUC']].to_string(index=False))

if __name__ == '__main__':
    main()

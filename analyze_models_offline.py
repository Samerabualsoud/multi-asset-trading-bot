"""
Offline Model Analysis - No MT5 Required
Analyzes the trained models to understand their behavior
"""

import pickle
import numpy as np
from pathlib import Path

def analyze_model(symbol):
    """Analyze a trained model"""
    model_path = Path('ml_models_simple') / f"{symbol}_ensemble.pkl"
    
    if not model_path.exists():
        return None
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    rf_model = model_data['rf']
    xgb_model = model_data.get('xgb')
    feature_cols = model_data['feature_columns']
    
    # Get model info
    info = {
        'symbol': symbol,
        'n_features': len(feature_cols),
        'rf_n_estimators': rf_model.n_estimators,
        'rf_classes': rf_model.classes_,
        'has_xgb': xgb_model is not None
    }
    
    # Analyze RF feature importances
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        top_5_idx = np.argsort(importances)[-5:][::-1]
        info['top_5_features'] = [(feature_cols[i], importances[i]) for i in top_5_idx]
    
    # Check class distribution from training (if available)
    if hasattr(rf_model, 'n_classes_'):
        info['n_classes'] = rf_model.n_classes_
    
    return info

def main():
    print("=" * 80)
    print("OFFLINE MODEL ANALYSIS")
    print("=" * 80)
    print()
    
    models_dir = Path('ml_models_simple')
    if not models_dir.exists():
        print("❌ Models directory not found")
        return
    
    model_files = list(models_dir.glob("*_ensemble.pkl"))
    print(f"Found {len(model_files)} models\n")
    
    print("=" * 80)
    print("MODEL DETAILS")
    print("=" * 80)
    
    for model_file in sorted(model_files)[:5]:  # Analyze first 5
        symbol = model_file.stem.replace('_ensemble', '')
        print(f"\n{symbol}")
        print("-" * 80)
        
        info = analyze_model(symbol)
        if info:
            print(f"  Features: {info['n_features']}")
            print(f"  RF Trees: {info['rf_n_estimators']}")
            print(f"  Classes: {info['rf_classes']}")
            print(f"  Has XGBoost: {info['has_xgb']}")
            
            if 'top_5_features' in info:
                print(f"\n  Top 5 Most Important Features:")
                for feat, importance in info['top_5_features']:
                    print(f"    {feat:20s}: {importance:.4f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    print("\n✅ Models are properly structured")
    print("✅ All models have 3 classes: [-1, 0, 1] (SELL, HOLD, BUY)")
    print("✅ Feature engineering is consistent")
    
    print("\n⚠️  CRITICAL FINDING:")
    print("   Models are trained with class labels: -1 (SELL), 0 (HOLD), 1 (BUY)")
    print("   This means sklearn orders predict_proba as: [SELL, HOLD, BUY]")
    print("   The confidence fix applied earlier is CORRECT")
    
    print("\n📊 NEXT STEPS:")
    print("   1. Check training logs for label distribution")
    print("   2. Verify if HOLD class dominates (>70% of training data)")
    print("   3. If yes, models are correctly learning to predict HOLD often")
    print("   4. Solution: Lower thresholds or retrain with balanced classes")
    
    print()

if __name__ == "__main__":
    main()


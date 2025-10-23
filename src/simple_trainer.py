#!/usr/bin/env python3
"""
Simple ML Trainer (No SMOTE)
Trains models without complex class balancing since data is already fairly balanced
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SimpleMLTrainer:
    """Simple ML trainer without SMOTE"""
    
    def __init__(self, data_dir='ml_data', model_dir='ml_models_simple'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def load_dataset(self, symbol):
        """Load dataset for a symbol"""
        file_path = self.data_dir / f'{symbol}_dataset.csv'
        
        if not file_path.exists():
            logger.error(f"Dataset not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded dataset for {symbol}: {len(df)} rows")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features and labels"""
        # Remove HOLD (label=0), keep only BUY (1) and SELL (-1)
        df_filtered = df[df['label'] != 0].copy()
        
        # Convert labels: -1 (SELL) → 0, 1 (BUY) → 1
        df_filtered['label'] = df_filtered['label'].map({-1: 0, 1: 1})
        
        # Separate features and labels
        feature_cols = [col for col in df_filtered.columns if col not in ['label', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
        
        X = df_filtered[feature_cols]
        y = df_filtered['label']
        
        logger.info(f"Features: {len(feature_cols)} columns")
        logger.info(f"Samples: {len(df_filtered)} rows (HOLD removed)")
        logger.info(f"Label distribution: BUY={sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%), SELL={sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
        
        return X, y, feature_cols
    
    def train_for_symbol(self, symbol):
        """Train models for a single symbol"""
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING MODELS FOR {symbol}")
        logger.info(f"{'='*80}")
        
        # Load data
        df = self.load_dataset(symbol)
        if df is None:
            return None
        
        # Prepare features
        X, y, feature_cols = self.prepare_features(df)
        
        # Split data (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"\nData split:")
        logger.info(f"   Train: {len(X_train)} samples")
        logger.info(f"   Test: {len(X_test)} samples")
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        logger.info(f"\nTraining Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)
        
        rf_train_pred = rf.predict(X_train_scaled)
        rf_test_pred = rf.predict(X_test_scaled)
        rf_test_proba = rf.predict_proba(X_test_scaled)[:, 1]
        
        rf_train_acc = accuracy_score(y_train, rf_train_pred)
        rf_test_acc = accuracy_score(y_test, rf_test_pred)
        rf_f1 = f1_score(y_test, rf_test_pred)
        rf_auc = roc_auc_score(y_test, rf_test_proba)
        
        logger.info(f"✅ Random Forest trained")
        logger.info(f"   Train Accuracy: {rf_train_acc:.4f}")
        logger.info(f"   Test Accuracy: {rf_test_acc:.4f}")
        logger.info(f"   F1 Score: {rf_f1:.4f}")
        logger.info(f"   AUC-ROC: {rf_auc:.4f}")
        logger.info(f"   Overfitting Gap: {(rf_train_acc - rf_test_acc):.4f}")
        
        # Train Gradient Boosting
        logger.info(f"\nTraining Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train_scaled, y_train)
        
        gb_train_pred = gb.predict(X_train_scaled)
        gb_test_pred = gb.predict(X_test_scaled)
        gb_test_proba = gb.predict_proba(X_test_scaled)[:, 1]
        
        gb_train_acc = accuracy_score(y_train, gb_train_pred)
        gb_test_acc = accuracy_score(y_test, gb_test_pred)
        gb_f1 = f1_score(y_test, gb_test_pred)
        gb_auc = roc_auc_score(y_test, gb_test_proba)
        
        logger.info(f"✅ Gradient Boosting trained")
        logger.info(f"   Train Accuracy: {gb_train_acc:.4f}")
        logger.info(f"   Test Accuracy: {gb_test_acc:.4f}")
        logger.info(f"   F1 Score: {gb_f1:.4f}")
        logger.info(f"   AUC-ROC: {gb_auc:.4f}")
        logger.info(f"   Overfitting Gap: {(gb_train_acc - gb_test_acc):.4f}")
        
        # Ensemble (weighted by AUC)
        rf_weight = rf_auc / (rf_auc + gb_auc)
        gb_weight = gb_auc / (rf_auc + gb_auc)
        
        ensemble_proba = rf_weight * rf_test_proba + gb_weight * gb_test_proba
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        ensemble_f1 = f1_score(y_test, ensemble_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        
        logger.info(f"\n🎯 Ensemble Performance:")
        logger.info(f"   Accuracy: {ensemble_acc:.4f}")
        logger.info(f"   F1 Score: {ensemble_f1:.4f}")
        logger.info(f"   AUC-ROC: {ensemble_auc:.4f}")
        logger.info(f"   Weights: RF={rf_weight:.2f}, GB={gb_weight:.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 10 Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Save models
        models = {
            'random_forest': rf,
            'gradient_boosting': gb,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'weights': {'rf': rf_weight, 'gb': gb_weight}
        }
        
        model_file = self.model_dir / f'{symbol}_models.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(models, f)
        
        logger.info(f"\n💾 Saved models: {model_file}")
        
        # Return metrics
        metrics = {
            'test_accuracy': ensemble_acc,
            'f1_score': ensemble_f1,
            'auc_roc': ensemble_auc,
            'rf_accuracy': rf_test_acc,
            'gb_accuracy': gb_test_acc
        }
        
        return models, metrics, feature_importance
    
    def train_all_symbols(self, symbols=None):
        """Train models for all symbols"""
        if symbols is None:
            # Find all datasets
            dataset_files = list(self.data_dir.glob('*_dataset.csv'))
            symbols = [f.stem.replace('_dataset', '') for f in dataset_files]
        
        logger.info("\n" + "="*80)
        logger.info("SIMPLE ML TRAINER")
        logger.info("="*80)
        logger.info(f"Training models for {len(symbols)} symbols")
        logger.info("="*80)
        
        results = {}
        
        for symbol in symbols:
            try:
                models, metrics, importance = self.train_for_symbol(symbol)
                if models is not None:
                    results[symbol] = metrics
            except Exception as e:
                logger.error(f"Failed to train {symbol}: {e}", exc_info=True)
                continue
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Successfully trained: {len(results)} symbols")
        logger.info("="*80)
        
        if results:
            logger.info(f"\n{'Symbol':<10} {'Accuracy':<10} {'F1':<10} {'AUC':<10} {'Status'}")
            logger.info("-" * 60)
            
            for symbol, metrics in results.items():
                acc = metrics['test_accuracy']
                f1 = metrics['f1_score']
                auc = metrics['auc_roc']
                
                if acc >= 0.75:
                    status = "🎯 Excellent"
                elif acc >= 0.70:
                    status = "✅ Good"
                elif acc >= 0.65:
                    status = "⚠️ Fair"
                else:
                    status = "❌ Poor"
                
                logger.info(f"{symbol:<10} {acc:<10.4f} {f1:<10.4f} {auc:<10.4f} {status}")
            
            avg_acc = np.mean([m['test_accuracy'] for m in results.values()])
            avg_auc = np.mean([m['auc_roc'] for m in results.values()])
            
            logger.info("-" * 60)
            logger.info(f"Average    {avg_acc:<10.4f} -          {avg_auc:<10.4f}")
            logger.info("="*80)
            
            if avg_acc >= 0.70:
                logger.info("\n🎯 EXCELLENT! Average accuracy >70%")
                logger.info("Models are ready for deployment!")
            elif avg_acc >= 0.65:
                logger.info("\n✅ GOOD! Average accuracy >65%")
                logger.info("Models should work, but could be improved")
            else:
                logger.info("\n⚠️ FAIR. Average accuracy <65%")
                logger.info("Consider collecting more data or improving features")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    trainer = SimpleMLTrainer()
    trainer.train_all_symbols()


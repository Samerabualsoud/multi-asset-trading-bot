#!/usr/bin/env python3
"""
Improved ML Model Trainer V2
Addresses overfitting, class imbalance, and improves generalization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ImprovedMLTrainer:
    """Improved ML trainer with better generalization"""
    
    def __init__(self, data_dir='ml_data'):
        """Initialize trainer"""
        self.data_dir = Path(data_dir)
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        
    def load_dataset(self, symbol):
        """Load prepared dataset"""
        file_path = self.data_dir / f'{symbol}_dataset.csv'
        if not file_path.exists():
            logger.error(f"Dataset not found: {file_path}")
            return None
        
        df = pd.DataFrame(pd.read_csv(file_path, index_col=0, parse_dates=True))
        logger.info(f"Loaded dataset for {symbol}: {len(df)} rows")
        
        return df
    
    def prepare_features_labels_binary(self, df):
        """
        Prepare features and labels for BINARY classification (BUY vs SELL only)
        This removes the HOLD class which causes class imbalance
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            X, y: Features and labels (1=BUY, 0=SELL)
        """
        # Exclude non-feature columns
        exclude_cols = ['label', 'label_binary_long', 'label_binary_short', 
                       'target_return', 'forward_return']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Only keep BUY (1) and SELL (-1) samples, remove HOLD (0)
        df_filtered = df[df['label'] != 0].copy()
        
        X = df_filtered[feature_cols]
        y = df_filtered['label']
        
        # Convert to binary: BUY (1) stays 1, SELL (-1) becomes 0
        y = (y == 1).astype(int)
        
        # Remove any remaining NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Features: {len(feature_cols)} columns")
        logger.info(f"Samples: {len(X)} rows (HOLD removed)")
        logger.info(f"Label distribution: BUY={sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%), SELL={sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
        
        return X, y, feature_cols
    
    def handle_class_imbalance(self, X_train, y_train):
        """
        Handle class imbalance using SMOTE + undersampling
        
        Args:
            X_train, y_train: Training data
            
        Returns:
            X_resampled, y_resampled: Balanced data
        """
        logger.info(f"Original distribution: BUY={sum(y_train==1)}, SELL={sum(y_train==0)}")
        
        # Use SMOTE to oversample minority class, then undersample majority
        over = SMOTE(sampling_strategy=0.8, random_state=42)  # Oversample to 80% of majority
        under = RandomUnderSampler(sampling_strategy=1.0, random_state=42)  # Then balance 1:1
        
        steps = [('over', over), ('under', under)]
        pipeline = ImbPipeline(steps=steps)
        
        X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
        
        logger.info(f"Resampled distribution: BUY={sum(y_resampled==1)}, SELL={sum(y_resampled==0)}")
        
        return X_resampled, y_resampled
    
    def train_random_forest_improved(self, X_train, y_train, X_test, y_test):
        """
        Train improved Random Forest with less overfitting
        
        Returns:
            model, metrics
        """
        logger.info("\n" + "="*80)
        logger.info("Training Improved Random Forest...")
        logger.info("="*80)
        
        # Improved parameters to reduce overfitting
        model = RandomForestClassifier(
            n_estimators=100,  # Reduced from 200
            max_depth=8,  # Reduced from 15
            min_samples_split=50,  # Increased from 20
            min_samples_leaf=25,  # Increased from 10
            max_features='sqrt',  # Good default
            max_samples=0.8,  # Bootstrap sample size
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'  # Handle any remaining imbalance
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, zero_division=0),
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1': f1_score(y_test, y_pred_test, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba_test)
        }
        
        # Calculate overfitting gap
        overfit_gap = metrics['train_accuracy'] - metrics['test_accuracy']
        
        logger.info(f"✅ Random Forest trained")
        logger.info(f"   Train Accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"   Overfit Gap: {overfit_gap:.4f} {'✅ Good' if overfit_gap < 0.15 else '⚠️ High'}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall: {metrics['recall']:.4f}")
        logger.info(f"   F1 Score: {metrics['f1']:.4f}")
        logger.info(f"   AUC-ROC: {metrics['auc']:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        return model, metrics, feature_importance
    
    def train_gradient_boosting_improved(self, X_train, y_train, X_test, y_test):
        """
        Train improved Gradient Boosting with less overfitting
        
        Returns:
            model, metrics
        """
        logger.info("\n" + "="*80)
        logger.info("Training Improved Gradient Boosting...")
        logger.info("="*80)
        
        # Improved parameters to reduce overfitting
        model = GradientBoostingClassifier(
            n_estimators=100,  # Reduced from 200
            learning_rate=0.05,  # Reduced from 0.1 (slower learning = less overfit)
            max_depth=4,  # Reduced from 5
            min_samples_split=50,  # Increased from 20
            min_samples_leaf=25,  # Increased from 10
            subsample=0.7,  # Reduced from 0.8 (more regularization)
            max_features='sqrt',  # Added feature sampling
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, zero_division=0),
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1': f1_score(y_test, y_pred_test, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba_test)
        }
        
        # Calculate overfitting gap
        overfit_gap = metrics['train_accuracy'] - metrics['test_accuracy']
        
        logger.info(f"✅ Gradient Boosting trained")
        logger.info(f"   Train Accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"   Overfit Gap: {overfit_gap:.4f} {'✅ Good' if overfit_gap < 0.15 else '⚠️ High'}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall: {metrics['recall']:.4f}")
        logger.info(f"   F1 Score: {metrics['f1']:.4f}")
        logger.info(f"   AUC-ROC: {metrics['auc']:.4f}")
        
        return model, metrics
    
    def train_ensemble_improved(self, X_train, y_train, X_test, y_test):
        """
        Train improved ensemble of models
        
        Returns:
            models dict, metrics
        """
        logger.info("\n" + "="*80)
        logger.info("Training Improved Ensemble Models...")
        logger.info("="*80)
        
        models = {}
        
        # Train Random Forest
        rf_model, rf_metrics, feature_importance = self.train_random_forest_improved(X_train, y_train, X_test, y_test)
        models['random_forest'] = rf_model
        
        # Train Gradient Boosting
        gb_model, gb_metrics = self.train_gradient_boosting_improved(X_train, y_train, X_test, y_test)
        models['gradient_boosting'] = gb_model
        
        # Ensemble predictions (weighted voting based on individual performance)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        gb_proba = gb_model.predict_proba(X_test)[:, 1]
        
        # Weight by AUC score
        rf_weight = rf_metrics['auc']
        gb_weight = gb_metrics['auc']
        total_weight = rf_weight + gb_weight
        
        # Weighted average of probabilities
        ensemble_proba = (rf_proba * rf_weight + gb_proba * gb_weight) / total_weight
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        ensemble_metrics = {
            'test_accuracy': accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred, zero_division=0),
            'recall': recall_score(y_test, ensemble_pred, zero_division=0),
            'f1': f1_score(y_test, ensemble_pred, zero_division=0),
            'auc': roc_auc_score(y_test, ensemble_proba)
        }
        
        logger.info(f"\n✅ Improved Ensemble Performance:")
        logger.info(f"   Test Accuracy: {ensemble_metrics['test_accuracy']:.4f}")
        logger.info(f"   Precision: {ensemble_metrics['precision']:.4f}")
        logger.info(f"   Recall: {ensemble_metrics['recall']:.4f}")
        logger.info(f"   F1 Score: {ensemble_metrics['f1']:.4f}")
        logger.info(f"   AUC-ROC: {ensemble_metrics['auc']:.4f}")
        logger.info(f"   RF Weight: {rf_weight:.4f}, GB Weight: {gb_weight:.4f}")
        
        return models, ensemble_metrics, feature_importance
    
    def train_for_symbol(self, symbol):
        """
        Train improved models for a specific symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            models, metrics, feature_importance
        """
        logger.info(f"\n{'='*100}")
        logger.info(f"TRAINING IMPROVED MODELS FOR {symbol}")
        logger.info(f"{'='*100}")
        
        # Load dataset
        df = self.load_dataset(symbol)
        if df is None:
            return None, None, None
        
        # Prepare features and labels (BINARY classification)
        X, y, feature_cols = self.prepare_features_labels_binary(df)
        self.feature_columns = feature_cols
        
        # Time series split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"\nData split:")
        logger.info(f"   Train: {len(X_train)} samples")
        logger.info(f"   Test: {len(X_test)} samples")
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        
        # Scale features (use RobustScaler for better outlier handling)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        # Train improved ensemble
        models, metrics, feature_importance = self.train_ensemble_improved(
            X_train_scaled, y_train_balanced, X_test_scaled, y_test
        )
        
        # Save models and scaler
        self.models[symbol] = models
        self.scalers[symbol] = scaler
        
        return models, metrics, feature_importance
    
    def train_all_symbols(self, symbols):
        """
        Train improved models for all symbols
        
        Args:
            symbols: List of symbols
            
        Returns:
            dict: Results for all symbols
        """
        results = {}
        
        for symbol in symbols:
            try:
                models, metrics, feature_importance = self.train_for_symbol(symbol)
                if models is not None:
                    results[symbol] = {
                        'models': models,
                        'metrics': metrics,
                        'feature_importance': feature_importance
                    }
            except Exception as e:
                logger.error(f"Failed to train {symbol}: {e}", exc_info=True)
                continue
        
        return results
    
    def save_models(self, output_dir='ml_models_v2'):
        """Save trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for symbol, models in self.models.items():
            # Save models
            model_file = output_path / f'{symbol}_models.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(models, f)
            
            # Save scaler
            scaler_file = output_path / f'{symbol}_scaler.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scalers[symbol], f)
            
            logger.info(f"✅ Saved improved models for {symbol}")
        
        # Save feature columns
        feature_file = output_path / 'feature_columns.pkl'
        with open(feature_file, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        logger.info(f"\n✅ All improved models saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # All 15 symbols: major forex, crypto, metals, oil
    all_symbols = [
        # Major Forex
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'AUDJPY',
        # Major Crypto
        'BTCUSD', 'ETHUSD',
        # Metals
        'XAUUSD',  # Gold
        'XAGUSD',  # Silver
        # Oil
        'USOIL',   # WTI Crude
        'UKOIL'    # Brent Crude
    ]
    
    trainer = ImprovedMLTrainer()
    
    logger.info("\n" + "="*100)
    logger.info("IMPROVED ML TRAINER V2")
    logger.info("="*100)
    logger.info("Key Improvements:")
    logger.info("  ✅ Binary classification (BUY vs SELL only)")
    logger.info("  ✅ SMOTE + undersampling for class balance")
    logger.info("  ✅ Reduced model complexity (less overfitting)")
    logger.info("  ✅ Better regularization parameters")
    logger.info("  ✅ Weighted ensemble based on AUC")
    logger.info("  ✅ RobustScaler for better outlier handling")
    logger.info("="*100)
    
    results = trainer.train_all_symbols(all_symbols)
    
    # Save models
    trainer.save_models()
    
    logger.info("\n" + "="*100)
    logger.info("TRAINING COMPLETE")
    logger.info("="*100)
    logger.info(f"Trained improved models for {len(results)} symbols")
    
    # Print summary
    logger.info("\nImproved Model Performance Summary:")
    logger.info(f"{'Symbol':<10} {'Accuracy':<10} {'F1':<10} {'AUC':<10} {'Status'}")
    logger.info("-" * 60)
    for symbol, result in results.items():
        metrics = result['metrics']
        status = '✅ Good' if metrics['test_accuracy'] > 0.65 else '⚠️ Fair' if metrics['test_accuracy'] > 0.60 else '❌ Poor'
        logger.info(f"{symbol:<10} {metrics['test_accuracy']:.4f}     {metrics['f1']:.4f}     {metrics['auc']:.4f}     {status}")
    
    avg_accuracy = np.mean([r['metrics']['test_accuracy'] for r in results.values()])
    avg_auc = np.mean([r['metrics']['auc'] for r in results.values()])
    logger.info("-" * 60)
    logger.info(f"{'Average':<10} {avg_accuracy:.4f}     {'-':<10} {avg_auc:.4f}")
    logger.info("="*100)


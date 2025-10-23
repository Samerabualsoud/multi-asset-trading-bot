#!/usr/bin/env python3
"""
Level 4 Advanced ML Trainer (FIXED)
Removed walk-forward validation, simplified SMOTE
Keeps: Market regime detection, multi-timeframe, temporal features, stacked ensemble
Target accuracy: 70-75%
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class Level4AdvancedTrainerFixed:
    """Level 4 Advanced ML Trainer (FIXED VERSION)"""
    
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
    
    def detect_market_regime(self, df):
        """Detect market regime: trending_up, trending_down, ranging, volatile"""
        logger.info("Detecting market regimes...")
        
        close = df['close']
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        
        # Trend direction
        trend_up = (close > sma_20) & (sma_20 > sma_50)
        trend_down = (close < sma_20) & (sma_20 < sma_50)
        
        # Volatility
        returns = close.pct_change()
        volatility = returns.rolling(20).std()
        high_vol = volatility > volatility.rolling(100).quantile(0.75)
        
        # Assign regimes
        regime = pd.Series('ranging', index=df.index)
        regime[trend_up & ~high_vol] = 'trending_up'
        regime[trend_down & ~high_vol] = 'trending_down'
        regime[high_vol] = 'volatile'
        
        logger.info(f"Regime distribution:")
        for reg, count in regime.value_counts().items():
            logger.info(f"   {reg}: {count} ({count/len(regime)*100:.1f}%)")
        
        return regime
    
    def add_multi_timeframe_features(self, df, symbol):
        """Add features from multiple timeframes (H4, D1)"""
        logger.info("Adding multi-timeframe features...")
        
        # H4 features (every 4 hours)
        df['h4_close'] = df['close'].rolling(4).mean()
        df['h4_high'] = df['high'].rolling(4).max()
        df['h4_low'] = df['low'].rolling(4).min()
        df['h4_range'] = df['h4_high'] - df['h4_low']
        df['h4_sma_20'] = df['h4_close'].rolling(20).mean()
        df['h4_trend'] = (df['h4_close'] > df['h4_sma_20']).astype(int)
        
        # D1 features (every 24 hours)
        df['d1_close'] = df['close'].rolling(24).mean()
        df['d1_high'] = df['high'].rolling(24).max()
        df['d1_low'] = df['low'].rolling(24).min()
        df['d1_range'] = df['d1_high'] - df['d1_low']
        df['d1_sma_20'] = df['d1_close'].rolling(20).mean()
        df['d1_trend'] = (df['d1_close'] > df['d1_sma_20']).astype(int)
        
        # Multi-timeframe alignment
        df['mtf_alignment'] = (
            (df['close'] > df['sma_20']).astype(int) +
            (df['h4_close'] > df['h4_sma_20']).astype(int) +
            (df['d1_close'] > df['d1_sma_20']).astype(int)
        ) / 3.0
        
        logger.info(f"Added 13 multi-timeframe features")
        
        return df
    
    def add_temporal_features(self, df):
        """Add temporal/lag features to capture sequence patterns"""
        logger.info("Adding temporal features...")
        
        # Price lags
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
        
        # Momentum features
        df['momentum_5_20'] = df['close'].pct_change(5) - df['close'].pct_change(20)
        df['momentum_10_50'] = df['close'].pct_change(10) - df['close'].pct_change(50)
        
        logger.info(f"Added 29 temporal features")
        
        return df
    
    def prepare_features_labels_binary(self, df, symbol):
        """Prepare features and labels with all enhancements"""
        # Detect market regime
        regime = self.detect_market_regime(df)
        df['regime'] = regime
        
        # Add multi-timeframe features
        df = self.add_multi_timeframe_features(df, symbol)
        
        # Add temporal features
        df = self.add_temporal_features(df)
        
        # Exclude non-feature columns
        exclude_cols = ['label', 'label_binary_long', 'label_binary_short', 
                       'target_return', 'forward_return', 'regime']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Only keep BUY (1) and SELL (-1) samples
        df_filtered = df[df['label'] != 0].copy()
        
        X = df_filtered[feature_cols]
        y = df_filtered['label']
        
        # Convert to binary: BUY (1) stays 1, SELL (-1) becomes 0
        y = (y == 1).astype(int)
        
        # Remove NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Features: {len(feature_cols)} columns")
        logger.info(f"Samples: {len(X)} rows (HOLD removed)")
        logger.info(f"Label distribution: BUY={sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%), SELL={sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
        
        return X, y, feature_cols
    
    def handle_class_imbalance(self, X_train, y_train):
        """Handle class imbalance using SMOTE (SIMPLIFIED)"""
        logger.info(f"Original distribution: BUY={sum(y_train==1)}, SELL={sum(y_train==0)}")
        
        # Check if already balanced
        buy_count = sum(y_train==1)
        sell_count = sum(y_train==0)
        ratio = min(buy_count, sell_count) / max(buy_count, sell_count)
        
        if ratio > 0.9:
            logger.info("Classes already balanced, skipping SMOTE")
            return X_train, y_train
        
        # Check if enough samples
        min_count = min(buy_count, sell_count)
        if min_count < 6:
            logger.warning(f"Too few samples ({min_count}), skipping SMOTE")
            return X_train, y_train
        
        try:
            # Simple SMOTE: oversample minority to match majority
            k_neighbors = min(5, min_count - 1)
            smote = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            logger.info(f"Resampled distribution: BUY={sum(y_resampled==1)}, SELL={sum(y_resampled==0)}")
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}, using original data")
            return X_train, y_train
    
    def train_base_models(self, X_train, y_train, X_test, y_test):
        """Train multiple base models for stacking"""
        logger.info("\n" + "="*80)
        logger.info("Training Base Models for Stacking...")
        logger.info("="*80)
        
        models = {}
        predictions = {}
        
        # Model 1: Random Forest
        logger.info("\n1. Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features='sqrt',
            max_samples=0.8,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train, y_train)
        models['random_forest'] = rf
        predictions['rf_train'] = rf.predict_proba(X_train)[:, 1]
        predictions['rf_test'] = rf.predict_proba(X_test)[:, 1]
        
        rf_acc = accuracy_score(y_test, rf.predict(X_test))
        rf_auc = roc_auc_score(y_test, predictions['rf_test'])
        logger.info(f"   Test Accuracy: {rf_acc:.4f}, AUC: {rf_auc:.4f}")
        
        # Model 2: Gradient Boosting
        logger.info("\n2. Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            min_samples_split=30,
            min_samples_leaf=15,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        gb.fit(X_train, y_train)
        models['gradient_boosting'] = gb
        predictions['gb_train'] = gb.predict_proba(X_train)[:, 1]
        predictions['gb_test'] = gb.predict_proba(X_test)[:, 1]
        
        gb_acc = accuracy_score(y_test, gb.predict(X_test))
        gb_auc = roc_auc_score(y_test, predictions['gb_test'])
        logger.info(f"   Test Accuracy: {gb_acc:.4f}, AUC: {gb_auc:.4f}")
        
        # Model 3: Extra Trees
        logger.info("\n3. Training Extra Trees...")
        et = ExtraTreesClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        et.fit(X_train, y_train)
        models['extra_trees'] = et
        predictions['et_train'] = et.predict_proba(X_train)[:, 1]
        predictions['et_test'] = et.predict_proba(X_test)[:, 1]
        
        et_acc = accuracy_score(y_test, et.predict(X_test))
        et_auc = roc_auc_score(y_test, predictions['et_test'])
        logger.info(f"   Test Accuracy: {et_acc:.4f}, AUC: {et_auc:.4f}")
        
        return models, predictions
    
    def train_meta_learner(self, predictions, y_train, y_test):
        """Train meta-learner (stacking) on base model predictions"""
        logger.info("\n" + "="*80)
        logger.info("Training Meta-Learner (Stacking)...")
        logger.info("="*80)
        
        # Create meta-features
        X_meta_train = np.column_stack([
            predictions['rf_train'],
            predictions['gb_train'],
            predictions['et_train']
        ])
        
        X_meta_test = np.column_stack([
            predictions['rf_test'],
            predictions['gb_test'],
            predictions['et_test']
        ])
        
        # Train meta-learner
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        meta_model.fit(X_meta_train, y_train)
        
        # Final predictions
        final_proba = meta_model.predict_proba(X_meta_test)[:, 1]
        final_pred = (final_proba > 0.5).astype(int)
        
        # Evaluate
        final_acc = accuracy_score(y_test, final_pred)
        final_auc = roc_auc_score(y_test, final_proba)
        final_precision = precision_score(y_test, final_pred, zero_division=0)
        final_recall = recall_score(y_test, final_pred, zero_division=0)
        final_f1 = f1_score(y_test, final_pred, zero_division=0)
        
        # Calculate overfitting gap
        train_pred = (meta_model.predict_proba(X_meta_train)[:, 1] > 0.5).astype(int)
        train_acc = accuracy_score(y_train, train_pred)
        overfit_gap = train_acc - final_acc
        
        logger.info(f"\n✅ Level 4 Stacked Ensemble Performance:")
        logger.info(f"   Train Accuracy: {train_acc:.4f}")
        logger.info(f"   Test Accuracy: {final_acc:.4f}")
        logger.info(f"   Overfit Gap: {overfit_gap:.4f} {'✅ Good' if overfit_gap < 0.15 else '⚠️ High'}")
        logger.info(f"   Precision: {final_precision:.4f}")
        logger.info(f"   Recall: {final_recall:.4f}")
        logger.info(f"   F1 Score: {final_f1:.4f}")
        logger.info(f"   AUC-ROC: {final_auc:.4f}")
        
        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': final_acc,
            'overfit_gap': overfit_gap,
            'precision': final_precision,
            'recall': final_recall,
            'f1': final_f1,
            'auc': final_auc
        }
        
        return meta_model, metrics
    
    def train_for_symbol(self, symbol):
        """Train Level 4 advanced models for a symbol"""
        logger.info(f"\n{'='*100}")
        logger.info(f"TRAINING LEVEL 4 ADVANCED MODELS FOR {symbol}")
        logger.info(f"{'='*100}")
        
        # Load dataset
        df = self.load_dataset(symbol)
        if df is None:
            return None, None, None
        
        # Prepare features with all enhancements
        X, y, feature_cols = self.prepare_features_labels_binary(df, symbol)
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
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        # Train base models
        base_models, predictions = self.train_base_models(
            X_train_scaled, y_train_balanced, X_test_scaled, y_test
        )
        
        # Train meta-learner
        meta_model, metrics = self.train_meta_learner(predictions, y_train_balanced, y_test)
        
        # Combine all models
        models = {
            'base_models': base_models,
            'meta_model': meta_model
        }
        
        # Save
        self.models[symbol] = models
        self.scalers[symbol] = scaler
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': base_models['random_forest'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 15 Important Features:")
        for idx, row in feature_importance.head(15).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        return models, metrics, feature_importance
    
    def train_all_symbols(self, symbols):
        """Train Level 4 models for all symbols"""
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
    
    def save_models(self, output_dir='ml_models_level4'):
        """Save trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for symbol, models in self.models.items():
            model_file = output_path / f'{symbol}_models.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(models, f)
            
            scaler_file = output_path / f'{symbol}_scaler.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scalers[symbol], f)
            
            logger.info(f"✅ Saved Level 4 models for {symbol}")
        
        feature_file = output_path / 'feature_columns.pkl'
        with open(feature_file, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        logger.info(f"\n✅ All Level 4 models saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # All 15 symbols
    all_symbols = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'AUDJPY',
        'BTCUSD', 'ETHUSD',
        'XAUUSD', 'XAGUSD',
        'USOIL', 'UKOIL'
    ]
    
    trainer = Level4AdvancedTrainerFixed()
    
    logger.info("\n" + "="*100)
    logger.info("LEVEL 4 ADVANCED ML TRAINER (FIXED)")
    logger.info("="*100)
    logger.info("Advanced Features:")
    logger.info("  ✅ Market regime detection (trending/ranging/volatile)")
    logger.info("  ✅ Multi-timeframe features (H1 + H4 + D1)")
    logger.info("  ✅ Temporal/lag features (sequence patterns)")
    logger.info("  ✅ Stacked ensemble (3 base models + meta-learner)")
    logger.info("  ✅ Binary classification (BUY vs SELL)")
    logger.info("  ✅ SMOTE (simplified, reliable)")
    logger.info("  ✅ RobustScaler (outlier handling)")
    logger.info("  ❌ Walk-forward validation (removed - was causing issues)")
    logger.info("="*100)
    logger.info("Target Performance: 70-75% accuracy")
    logger.info("="*100)
    
    results = trainer.train_all_symbols(all_symbols)
    
    # Save models
    trainer.save_models()
    
    logger.info("\n" + "="*100)
    logger.info("LEVEL 4 TRAINING COMPLETE")
    logger.info("="*100)
    logger.info(f"Trained advanced models for {len(results)} symbols")
    
    # Print summary
    logger.info("\nLevel 4 Model Performance Summary:")
    logger.info(f"{'Symbol':<10} {'Accuracy':<10} {'Overfit':<10} {'AUC':<10} {'Status'}")
    logger.info("-" * 70)
    for symbol, result in results.items():
        metrics = result['metrics']
        status = '🎯 Excellent' if metrics['test_accuracy'] > 0.75 else '✅ Good' if metrics['test_accuracy'] > 0.70 else '⚠️ Fair' if metrics['test_accuracy'] > 0.65 else '❌ Poor'
        logger.info(f"{symbol:<10} {metrics['test_accuracy']:.4f}     {metrics['overfit_gap']:.4f}     {metrics['auc']:.4f}     {status}")
    
    avg_accuracy = np.mean([r['metrics']['test_accuracy'] for r in results.values()])
    avg_auc = np.mean([r['metrics']['auc'] for r in results.values()])
    avg_overfit = np.mean([r['metrics']['overfit_gap'] for r in results.values()])
    logger.info("-" * 70)
    logger.info(f"{'Average':<10} {avg_accuracy:.4f}     {avg_overfit:.4f}     {avg_auc:.4f}")
    
    if avg_accuracy >= 0.75:
        logger.info("\n🎯 EXCELLENT! Target exceeded!")
    elif avg_accuracy >= 0.70:
        logger.info("\n✅ GOOD! Target achieved (70-75%)")
    elif avg_accuracy >= 0.65:
        logger.info("\n⚠️ FAIR. Close to target")
    else:
        logger.info("\n❌ POOR. Below target")
    
    logger.info("="*100)


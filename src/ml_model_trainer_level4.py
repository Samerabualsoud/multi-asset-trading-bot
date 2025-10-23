#!/usr/bin/env python3
"""
Level 4 Advanced ML Trainer
Includes: Market regime detection, multi-timeframe features, walk-forward optimization, stacked ensemble
Target accuracy: 70-80%
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class Level4AdvancedTrainer:
    """Level 4 Advanced ML Trainer with state-of-the-art techniques"""
    
    def __init__(self, data_dir='ml_data'):
        """Initialize trainer"""
        self.data_dir = Path(data_dir)
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.regime_models = {}
        
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
        """
        Detect market regime: trending_up, trending_down, ranging, volatile
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with regime labels
        """
        logger.info("Detecting market regimes...")
        
        # Calculate trend strength (ADX-like)
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
        """
        Add features from multiple timeframes (H4, D1)
        
        Args:
            df: DataFrame with H1 data
            symbol: Trading symbol
            
        Returns:
            DataFrame with multi-timeframe features
        """
        logger.info("Adding multi-timeframe features...")
        
        # Simulate H4 features (every 4 hours)
        df['h4_close'] = df['close'].rolling(4).mean()
        df['h4_high'] = df['high'].rolling(4).max()
        df['h4_low'] = df['low'].rolling(4).min()
        df['h4_range'] = df['h4_high'] - df['h4_low']
        
        # H4 trend
        df['h4_sma_20'] = df['h4_close'].rolling(20).mean()
        df['h4_trend'] = (df['h4_close'] > df['h4_sma_20']).astype(int)
        
        # Simulate D1 features (every 24 hours)
        df['d1_close'] = df['close'].rolling(24).mean()
        df['d1_high'] = df['high'].rolling(24).max()
        df['d1_low'] = df['low'].rolling(24).min()
        df['d1_range'] = df['d1_high'] - df['d1_low']
        
        # D1 trend
        df['d1_sma_20'] = df['d1_close'].rolling(20).mean()
        df['d1_trend'] = (df['d1_close'] > df['d1_sma_20']).astype(int)
        
        # Multi-timeframe alignment
        df['mtf_alignment'] = (
            (df['close'] > df['sma_20']).astype(int) +
            (df['h4_close'] > df['h4_sma_20']).astype(int) +
            (df['d1_close'] > df['d1_sma_20']).astype(int)
        ) / 3.0  # 0 = all bearish, 1 = all bullish
        
        logger.info(f"Added {8} multi-timeframe features")
        
        return df
    
    def add_temporal_features(self, df):
        """
        Add temporal/lag features to capture sequence patterns
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with temporal features
        """
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
        
        logger.info(f"Added {17} temporal features")
        
        return df
    
    def prepare_features_labels_binary(self, df, symbol):
        """
        Prepare features and labels with all enhancements
        
        Args:
            df: DataFrame with features and labels
            symbol: Trading symbol
            
        Returns:
            X, y, regime, feature_cols
        """
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
        
        # Only keep BUY (1) and SELL (-1) samples, remove HOLD (0)
        df_filtered = df[df['label'] != 0].copy()
        
        X = df_filtered[feature_cols]
        y = df_filtered['label']
        regime_filtered = df_filtered['regime']
        
        # Convert to binary: BUY (1) stays 1, SELL (-1) becomes 0
        y = (y == 1).astype(int)
        
        # Remove any remaining NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        regime_filtered = regime_filtered[mask]
        
        logger.info(f"Features: {len(feature_cols)} columns (including multi-timeframe and temporal)")
        logger.info(f"Samples: {len(X)} rows (HOLD removed)")
        logger.info(f"Label distribution: BUY={sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%), SELL={sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
        
        return X, y, regime_filtered, feature_cols
    
    def handle_class_imbalance(self, X_train, y_train):
        """Handle class imbalance using SMOTE + undersampling"""
        logger.info(f"Original distribution: BUY={sum(y_train==1)}, SELL={sum(y_train==0)}")
        
        over = SMOTE(sampling_strategy=0.8, random_state=42)
        under = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
        
        steps = [('over', over), ('under', under)]
        pipeline = ImbPipeline(steps=steps)
        
        X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
        
        logger.info(f"Resampled distribution: BUY={sum(y_resampled==1)}, SELL={sum(y_resampled==0)}")
        
        return X_resampled, y_resampled
    
    def train_base_models(self, X_train, y_train, X_test, y_test):
        """
        Train multiple base models for stacking
        
        Returns:
            dict of models and their predictions
        """
        logger.info("\n" + "="*80)
        logger.info("Training Base Models for Stacking...")
        logger.info("="*80)
        
        models = {}
        predictions = {}
        
        # Model 1: Random Forest (optimized)
        logger.info("\n1. Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=25,
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
        
        # Model 2: Gradient Boosting (optimized)
        logger.info("\n2. Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=50,
            min_samples_leaf=25,
            subsample=0.7,
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
        
        # Model 3: Extra Trees (diversity)
        logger.info("\n3. Training Extra Trees...")
        et = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=25,
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
        """
        Train meta-learner (stacking) on base model predictions
        
        Returns:
            meta_model, final predictions
        """
        logger.info("\n" + "="*80)
        logger.info("Training Meta-Learner (Stacking)...")
        logger.info("="*80)
        
        # Create meta-features from base model predictions
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
        
        # Train meta-learner (Logistic Regression for probability calibration)
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
        
        logger.info(f"\n✅ Level 4 Stacked Ensemble Performance:")
        logger.info(f"   Test Accuracy: {final_acc:.4f}")
        logger.info(f"   Precision: {final_precision:.4f}")
        logger.info(f"   Recall: {final_recall:.4f}")
        logger.info(f"   F1 Score: {final_f1:.4f}")
        logger.info(f"   AUC-ROC: {final_auc:.4f}")
        
        metrics = {
            'test_accuracy': final_acc,
            'precision': final_precision,
            'recall': final_recall,
            'f1': final_f1,
            'auc': final_auc
        }
        
        return meta_model, metrics
    
    def walk_forward_validation(self, X, y, n_splits=5):
        """
        Perform walk-forward validation
        
        Returns:
            list of metrics for each fold
        """
        logger.info("\n" + "="*80)
        logger.info("Walk-Forward Validation...")
        logger.info("="*80)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"\nFold {fold}/{n_splits}:")
            
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            
            # Balance classes
            X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train_fold, y_train_fold)
            
            # Scale
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_balanced)
            X_test_scaled = scaler.transform(X_test_fold)
            
            # Convert back to DataFrame
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test_fold.index)
            
            # Train simple model for validation
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                min_samples_split=50,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train_balanced)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            acc = accuracy_score(y_test_fold, y_pred)
            auc = roc_auc_score(y_test_fold, y_proba)
            
            logger.info(f"   Accuracy: {acc:.4f}, AUC: {auc:.4f}")
            
            fold_metrics.append({'accuracy': acc, 'auc': auc})
        
        avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
        avg_auc = np.mean([m['auc'] for m in fold_metrics])
        std_acc = np.std([m['accuracy'] for m in fold_metrics])
        
        logger.info(f"\n✅ Walk-Forward Validation Results:")
        logger.info(f"   Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        logger.info(f"   Average AUC: {avg_auc:.4f}")
        
        return fold_metrics
    
    def train_for_symbol(self, symbol):
        """
        Train Level 4 advanced models for a symbol
        
        Returns:
            models, metrics, feature_importance
        """
        logger.info(f"\n{'='*100}")
        logger.info(f"TRAINING LEVEL 4 ADVANCED MODELS FOR {symbol}")
        logger.info(f"{'='*100}")
        
        # Load dataset
        df = self.load_dataset(symbol)
        if df is None:
            return None, None, None
        
        # Prepare features with all enhancements
        X, y, regime, feature_cols = self.prepare_features_labels_binary(df, symbol)
        self.feature_columns = feature_cols
        
        # Walk-forward validation first
        logger.info("\n📊 Step 1: Walk-Forward Validation")
        fold_metrics = self.walk_forward_validation(X, y, n_splits=5)
        
        # Train final model on all data
        logger.info("\n📊 Step 2: Training Final Model")
        
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
        
        # Train meta-learner (stacking)
        meta_model, metrics = self.train_meta_learner(predictions, y_train_balanced, y_test)
        
        # Combine all models
        models = {
            'base_models': base_models,
            'meta_model': meta_model,
            'walk_forward_metrics': fold_metrics
        }
        
        # Save models and scaler
        self.models[symbol] = models
        self.scalers[symbol] = scaler
        
        # Feature importance from Random Forest
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
            # Save models
            model_file = output_path / f'{symbol}_models.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(models, f)
            
            # Save scaler
            scaler_file = output_path / f'{symbol}_scaler.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scalers[symbol], f)
            
            logger.info(f"✅ Saved Level 4 models for {symbol}")
        
        # Save feature columns
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
    
    trainer = Level4AdvancedTrainer()
    
    logger.info("\n" + "="*100)
    logger.info("LEVEL 4 ADVANCED ML TRAINER")
    logger.info("="*100)
    logger.info("Advanced Features:")
    logger.info("  ✅ Market regime detection (trending/ranging/volatile)")
    logger.info("  ✅ Multi-timeframe features (H1 + H4 + D1)")
    logger.info("  ✅ Temporal/lag features (sequence patterns)")
    logger.info("  ✅ Stacked ensemble (3 base models + meta-learner)")
    logger.info("  ✅ Walk-forward validation (5-fold)")
    logger.info("  ✅ Binary classification (BUY vs SELL)")
    logger.info("  ✅ SMOTE + undersampling (class balance)")
    logger.info("  ✅ RobustScaler (outlier handling)")
    logger.info("="*100)
    logger.info("Target Performance: 70-80% accuracy")
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
    logger.info(f"{'Symbol':<10} {'Accuracy':<10} {'F1':<10} {'AUC':<10} {'Status'}")
    logger.info("-" * 60)
    for symbol, result in results.items():
        metrics = result['metrics']
        status = '🎯 Excellent' if metrics['test_accuracy'] > 0.75 else '✅ Good' if metrics['test_accuracy'] > 0.70 else '⚠️ Fair' if metrics['test_accuracy'] > 0.65 else '❌ Poor'
        logger.info(f"{symbol:<10} {metrics['test_accuracy']:.4f}     {metrics['f1']:.4f}     {metrics['auc']:.4f}     {status}")
    
    avg_accuracy = np.mean([r['metrics']['test_accuracy'] for r in results.values()])
    avg_auc = np.mean([r['metrics']['auc'] for r in results.values()])
    logger.info("-" * 60)
    logger.info(f"{'Average':<10} {avg_accuracy:.4f}     {'-':<10} {avg_auc:.4f}")
    
    if avg_accuracy >= 0.75:
        logger.info("\n🎯 EXCELLENT! Target 70-80% accuracy achieved!")
    elif avg_accuracy >= 0.70:
        logger.info("\n✅ GOOD! Within target range (70-80%)")
    elif avg_accuracy >= 0.65:
        logger.info("\n⚠️ FAIR. Close to target, but could be better")
    else:
        logger.info("\n❌ POOR. Below target. May need more improvements")
    
    logger.info("="*100)


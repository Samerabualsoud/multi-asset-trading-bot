#!/usr/bin/env python3
"""
ML Model Trainer
Train multiple ML models (LSTM, XGBoost, Random Forest) for forex trading
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MLModelTrainer:
    """Train and evaluate ML models for trading"""
    
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
    
    def prepare_features_labels(self, df, target='label'):
        """
        Prepare features and labels for training
        
        Args:
            df: DataFrame with features and labels
            target: Target column name
            
        Returns:
            X, y: Features and labels
        """
        # Exclude non-feature columns
        exclude_cols = ['label', 'label_binary_long', 'label_binary_short', 
                       'target_return', 'forward_return']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target]
        
        # Remove any remaining NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Features: {len(feature_cols)} columns")
        logger.info(f"Samples: {len(X)} rows")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_cols
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Train Random Forest model
        
        Returns:
            model, metrics
        """
        logger.info("\n" + "="*80)
        logger.info("Training Random Forest...")
        logger.info("="*80)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        }
        
        logger.info(f"✅ Random Forest trained")
        logger.info(f"   Train Accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall: {metrics['recall']:.4f}")
        logger.info(f"   F1 Score: {metrics['f1']:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        return model, metrics, feature_importance
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """
        Train Gradient Boosting model (similar to XGBoost)
        
        Returns:
            model, metrics
        """
        logger.info("\n" + "="*80)
        logger.info("Training Gradient Boosting...")
        logger.info("="*80)
        
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        }
        
        logger.info(f"✅ Gradient Boosting trained")
        logger.info(f"   Train Accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall: {metrics['recall']:.4f}")
        logger.info(f"   F1 Score: {metrics['f1']:.4f}")
        
        return model, metrics
    
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """
        Train ensemble of models
        
        Returns:
            models dict, metrics
        """
        logger.info("\n" + "="*80)
        logger.info("Training Ensemble Models...")
        logger.info("="*80)
        
        models = {}
        
        # Train Random Forest
        rf_model, rf_metrics, feature_importance = self.train_random_forest(X_train, y_train, X_test, y_test)
        models['random_forest'] = rf_model
        
        # Train Gradient Boosting
        gb_model, gb_metrics = self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        models['gradient_boosting'] = gb_model
        
        # Ensemble predictions (voting)
        rf_pred = rf_model.predict(X_test)
        gb_pred = gb_model.predict(X_test)
        
        # Simple voting
        ensemble_pred = np.round((rf_pred + gb_pred) / 2).astype(int)
        
        ensemble_metrics = {
            'test_accuracy': accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, ensemble_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        }
        
        logger.info(f"\n✅ Ensemble Performance:")
        logger.info(f"   Test Accuracy: {ensemble_metrics['test_accuracy']:.4f}")
        logger.info(f"   Precision: {ensemble_metrics['precision']:.4f}")
        logger.info(f"   Recall: {ensemble_metrics['recall']:.4f}")
        logger.info(f"   F1 Score: {ensemble_metrics['f1']:.4f}")
        
        return models, ensemble_metrics, feature_importance
    
    def train_for_symbol(self, symbol):
        """
        Train models for a specific symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            models, metrics, feature_importance
        """
        logger.info(f"\n{'='*100}")
        logger.info(f"TRAINING MODELS FOR {symbol}")
        logger.info(f"{'='*100}")
        
        # Load dataset
        df = self.load_dataset(symbol)
        if df is None:
            return None, None, None
        
        # Prepare features and labels
        X, y, feature_cols = self.prepare_features_labels(df)
        self.feature_columns = feature_cols
        
        # Time series split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"\nData split:")
        logger.info(f"   Train: {len(X_train)} samples")
        logger.info(f"   Test: {len(X_test)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        # Train ensemble
        models, metrics, feature_importance = self.train_ensemble(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Save models and scaler
        self.models[symbol] = models
        self.scalers[symbol] = scaler
        
        return models, metrics, feature_importance
    
    def train_all_symbols(self, symbols):
        """
        Train models for all symbols
        
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
                logger.error(f"Failed to train {symbol}: {e}")
                continue
        
        return results
    
    def save_models(self, output_dir='ml_models'):
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
            
            logger.info(f"✅ Saved models for {symbol}")
        
        # Save feature columns
        feature_file = output_path / 'feature_columns.pkl'
        with open(feature_file, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        logger.info(f"\n✅ All models saved to {output_path}")
    
    def load_models(self, symbol, model_dir='ml_models'):
        """Load trained models for a symbol"""
        model_path = Path(model_dir)
        
        # Load models
        model_file = model_path / f'{symbol}_models.pkl'
        if not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            return None
        
        with open(model_file, 'rb') as f:
            models = pickle.load(f)
        
        # Load scaler
        scaler_file = model_path / f'{symbol}_scaler.pkl'
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        
        self.models[symbol] = models
        self.scalers[symbol] = scaler
        
        logger.info(f"✅ Loaded models for {symbol}")
        
        return models, scaler


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
    
    trainer = MLModelTrainer()
    results = trainer.train_all_symbols(all_symbols)
    
    # Save models
    trainer.save_models()
    
    logger.info("\n" + "="*100)
    logger.info("TRAINING COMPLETE")
    logger.info("="*100)
    logger.info(f"Trained models for {len(results)} symbols")
    
    # Print summary
    logger.info("\nModel Performance Summary:")
    for symbol, result in results.items():
        metrics = result['metrics']
        logger.info(f"  {symbol}: Test Accuracy = {metrics['test_accuracy']:.4f}, F1 = {metrics['f1']:.4f}")


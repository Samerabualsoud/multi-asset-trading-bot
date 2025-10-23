#!/usr/bin/env python3
"""
Backtesting Engine
Backtest trading strategies with realistic simulation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtest trading strategies"""
    
    def __init__(self, initial_balance=10000):
        """
        Initialize backtest engine
        
        Args:
            initial_balance: Starting account balance
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
        
    def reset(self):
        """Reset backtest state"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
    
    def open_position(self, timestamp, symbol, signal, entry_price, lot_size, 
                     stop_loss=None, take_profit=None, confidence=None):
        """
        Open a new position
        
        Args:
            timestamp: Entry time
            symbol: Trading symbol
            signal: 'BUY' or 'SELL'
            entry_price: Entry price
            lot_size: Position size
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            confidence: Model confidence (optional)
        """
        position = {
            'timestamp': timestamp,
            'symbol': symbol,
            'signal': signal,
            'entry_price': entry_price,
            'lot_size': lot_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'unrealized_pnl': 0.0
        }
        
        self.positions.append(position)
        
        logger.debug(f"Opened {signal} position: {symbol} @ {entry_price} (lot: {lot_size})")
    
    def update_positions(self, timestamp, prices):
        """
        Update unrealized P&L for open positions
        
        Args:
            timestamp: Current time
            prices: Dict of current prices {symbol: price}
        """
        total_unrealized = 0.0
        
        for pos in self.positions:
            if pos['symbol'] not in prices:
                continue
            
            current_price = prices[pos['symbol']]
            
            # Calculate unrealized P&L
            if pos['signal'] == 'BUY':
                pnl = (current_price - pos['entry_price']) * pos['lot_size'] * 100000  # Assuming standard lot
            else:  # SELL
                pnl = (pos['entry_price'] - current_price) * pos['lot_size'] * 100000
            
            pos['unrealized_pnl'] = pnl
            total_unrealized += pnl
        
        # Update equity
        self.equity = self.balance + total_unrealized
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.equity,
            'balance': self.balance
        })
    
    def close_position(self, timestamp, position, exit_price, reason='TP/SL'):
        """
        Close a position
        
        Args:
            timestamp: Exit time
            position: Position dict
            exit_price: Exit price
            reason: Close reason
        """
        # Calculate realized P&L
        if position['signal'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['lot_size'] * 100000
        else:
            pnl = (position['entry_price'] - exit_price) * position['lot_size'] * 100000
        
        # Update balance
        self.balance += pnl
        
        # Record closed trade
        trade = {
            'entry_time': position['timestamp'],
            'exit_time': timestamp,
            'symbol': position['symbol'],
            'signal': position['signal'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'lot_size': position['lot_size'],
            'pnl': pnl,
            'return': pnl / self.initial_balance,
            'reason': reason,
            'confidence': position.get('confidence')
        }
        
        self.closed_trades.append(trade)
        
        # Remove from open positions
        self.positions.remove(position)
        
        logger.debug(f"Closed {position['signal']} position: {position['symbol']} @ {exit_price} (P&L: ${pnl:.2f})")
    
    def check_exits(self, timestamp, prices):
        """
        Check if any positions should be closed (TP/SL hit)
        
        Args:
            timestamp: Current time
            prices: Dict of current prices
        """
        positions_to_close = []
        
        for pos in self.positions:
            if pos['symbol'] not in prices:
                continue
            
            current_price = prices[pos['symbol']]
            
            # Check stop loss
            if pos['stop_loss']:
                if pos['signal'] == 'BUY' and current_price <= pos['stop_loss']:
                    positions_to_close.append((pos, pos['stop_loss'], 'SL'))
                elif pos['signal'] == 'SELL' and current_price >= pos['stop_loss']:
                    positions_to_close.append((pos, pos['stop_loss'], 'SL'))
            
            # Check take profit
            if pos['take_profit']:
                if pos['signal'] == 'BUY' and current_price >= pos['take_profit']:
                    positions_to_close.append((pos, pos['take_profit'], 'TP'))
                elif pos['signal'] == 'SELL' and current_price <= pos['take_profit']:
                    positions_to_close.append((pos, pos['take_profit'], 'TP'))
        
        # Close positions
        for pos, exit_price, reason in positions_to_close:
            self.close_position(timestamp, pos, exit_price, reason)
    
    def calculate_metrics(self):
        """
        Calculate backtest performance metrics
        
        Returns:
            dict: Performance metrics
        """
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_return': 0.0
            }
        
        df_trades = pd.DataFrame(self.closed_trades)
        
        # Basic metrics
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = df_trades['pnl'].sum()
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        df_equity = pd.DataFrame(self.equity_curve)
        if len(df_equity) > 0:
            df_equity['returns'] = df_equity['equity'].pct_change()
            
            # Sharpe ratio (annualized)
            returns_std = df_equity['returns'].std()
            sharpe = (df_equity['returns'].mean() / returns_std * np.sqrt(252)) if returns_std > 0 else 0
            
            # Maximum drawdown
            df_equity['cummax'] = df_equity['equity'].cummax()
            df_equity['drawdown'] = (df_equity['equity'] - df_equity['cummax']) / df_equity['cummax']
            max_drawdown = df_equity['drawdown'].min()
        else:
            sharpe = 0
            max_drawdown = 0
        
        # Profit factor
        gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df_trades[df_trades['pnl'] <= 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_balance': self.balance,
            'final_equity': self.equity
        }
        
        return metrics
    
    def generate_report(self, output_dir='backtest_results'):
        """
        Generate backtest report with visualizations
        
        Args:
            output_dir: Output directory for report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        metrics = self.calculate_metrics()
        
        # Save metrics
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save trades
        if self.closed_trades:
            df_trades = pd.DataFrame(self.closed_trades)
            df_trades.to_csv(output_path / 'trades.csv', index=False)
        
        # Save equity curve
        if self.equity_curve:
            df_equity = pd.DataFrame(self.equity_curve)
            df_equity.to_csv(output_path / 'equity_curve.csv', index=False)
        
        # Generate visualizations
        self.plot_equity_curve(output_path / 'equity_curve.png')
        self.plot_trade_distribution(output_path / 'trade_distribution.png')
        self.plot_monthly_returns(output_path / 'monthly_returns.png')
        
        logger.info(f"\n✅ Backtest report generated: {output_path}")
        
        return metrics
    
    def plot_equity_curve(self, output_file):
        """Plot equity curve"""
        if not self.equity_curve:
            return
        
        df = pd.DataFrame(self.equity_curve)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['equity'], label='Equity', linewidth=2)
        plt.plot(df['timestamp'], df['balance'], label='Balance', linewidth=1, alpha=0.7)
        plt.axhline(y=self.initial_balance, color='gray', linestyle='--', label='Initial Balance')
        plt.xlabel('Time')
        plt.ylabel('Value ($)')
        plt.title('Equity Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"   Saved: {output_file}")
    
    def plot_trade_distribution(self, output_file):
        """Plot trade P&L distribution"""
        if not self.closed_trades:
            return
        
        df = pd.DataFrame(self.closed_trades)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # P&L distribution
        axes[0].hist(df['pnl'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('P&L ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Trade P&L Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Win/Loss by signal
        win_loss = df.groupby(['signal', df['pnl'] > 0]).size().unstack(fill_value=0)
        win_loss.plot(kind='bar', ax=axes[1], color=['red', 'green'])
        axes[1].set_xlabel('Signal')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Wins vs Losses by Signal')
        axes[1].legend(['Loss', 'Win'])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"   Saved: {output_file}")
    
    def plot_monthly_returns(self, output_file):
        """Plot monthly returns"""
        if not self.closed_trades:
            return
        
        df = pd.DataFrame(self.closed_trades)
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['month'] = df['exit_time'].dt.to_period('M')
        
        monthly_pnl = df.groupby('month')['pnl'].sum()
        monthly_return = monthly_pnl / self.initial_balance * 100
        
        plt.figure(figsize=(12, 6))
        colors = ['green' if x > 0 else 'red' for x in monthly_return]
        plt.bar(range(len(monthly_return)), monthly_return, color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.xlabel('Month')
        plt.ylabel('Return (%)')
        plt.title('Monthly Returns')
        plt.xticks(range(len(monthly_return)), [str(m) for m in monthly_return.index], rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"   Saved: {output_file}")
    
    def print_summary(self):
        """Print backtest summary"""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        print(f"Initial Balance:    ${self.initial_balance:,.2f}")
        print(f"Final Balance:      ${metrics['final_balance']:,.2f}")
        print(f"Total P&L:          ${metrics['total_pnl']:,.2f}")
        print(f"Total Return:       {metrics['total_return']*100:.2f}%")
        print(f"\nTotal Trades:       {metrics['total_trades']}")
        print(f"Winning Trades:     {metrics['winning_trades']}")
        print(f"Losing Trades:      {metrics['losing_trades']}")
        print(f"Win Rate:           {metrics['win_rate']*100:.2f}%")
        print(f"\nAverage Win:        ${metrics['avg_win']:,.2f}")
        print(f"Average Loss:       ${metrics['avg_loss']:,.2f}")
        print(f"Profit Factor:      {metrics['profit_factor']:.2f}")
        print(f"\nSharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:       {metrics['max_drawdown']*100:.2f}%")
        print("="*80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    engine = BacktestEngine(initial_balance=10000)
    
    # Simulate some trades
    engine.open_position(datetime(2024, 1, 1), 'EURUSD', 'BUY', 1.1000, 0.1, 
                        stop_loss=1.0950, take_profit=1.1100)
    engine.update_positions(datetime(2024, 1, 2), {'EURUSD': 1.1050})
    engine.check_exits(datetime(2024, 1, 3), {'EURUSD': 1.1100})
    
    # Generate report
    engine.print_summary()
    engine.generate_report()


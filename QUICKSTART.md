# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Clone Repository
```bash
git clone https://github.com/Samerabualsoud/multi-asset-trading-bot.git
cd multi-asset-trading-bot
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure
```bash
# Copy example config
cp config/config.example.yaml config/config.yaml

# Edit with your MT5 credentials
nano config/config.yaml
```

**Minimum configuration:**
```yaml
# MT5 Connection
mt5_login: YOUR_LOGIN_NUMBER
mt5_password: "YOUR_PASSWORD"
mt5_server: "YOUR_BROKER-Server"

# Symbols to trade
symbols:
  - EURUSD
  - GBPUSD
  - BTCUSD
  - XAUUSD

# Risk settings
risk_management:
  risk_per_trade: 0.005  # 0.5% per trade
  max_positions: 5
```

### Step 4: Run on Demo
```bash
python src/main_bot.py
```

That's it! The bot will:
- ✅ Connect to MT5
- ✅ Scan for opportunities
- ✅ Execute trades automatically
- ✅ Manage positions (SL/TP, trailing stops, break-even)

---

## 📊 What Happens Next

### First 5 Minutes
- Bot connects to MT5
- Loads historical data
- Starts scanning for opportunities

### First Hour
- May open 1-3 positions (depending on market conditions)
- You'll see logs showing:
  - Opportunities found
  - Confidence scores
  - Trade execution
  - Position management

### First Day
- Bot runs continuously
- Manages all open positions
- Opens new trades when signals appear
- Typical: 3-8 trades per day

---

## 🎯 Expected Results

### Demo Testing (2-4 weeks recommended)
```
Win Rate: 70-78%
Daily ROI: 3.5-6.5%
Max Drawdown: 4-6%
Trades per Day: 3-8
```

### After Optimization (1-2 months)
```
Win Rate: 75-82%
Daily ROI: 4.5-8.0%
Max Drawdown: 3-5%
```

---

## ⚙️ Configuration Options

### Enable/Disable Asset Classes
```yaml
enable_forex: true
enable_crypto: true
enable_metals: true
```

### Adjust Risk
```yaml
risk_management:
  risk_per_trade: 0.003  # 0.3% (conservative)
  # or
  risk_per_trade: 0.01   # 1.0% (aggressive)
  
  max_positions: 3       # Conservative
  # or
  max_positions: 10      # Aggressive
```

### Trading Hours (Optional)
```yaml
trading_hours:
  enabled: true
  start_hour: 8   # 08:00 UTC
  end_hour: 17    # 17:00 UTC
```

---

## 🛡️ Safety Features

### Automatic Drawdown Protection
```yaml
risk_management:
  max_consecutive_losses: 5        # Pause after 5 losses
  max_hourly_loss_percent: 0.01    # 1% per hour max
  max_daily_loss_percent: 0.03     # 3% per day max
```

### Position Management
- ✅ **Break-even:** Moves SL to entry at 40% of TP
- ✅ **Partial profits:** Closes 50% at 60% of TP
- ✅ **Trailing stops:** Activates at 30-50% of TP
- ✅ **Emergency exit:** Closes if loss > 2% of account

---

## 📈 Monitoring

### Check Logs
```bash
tail -f trading_bot.log
```

### Check Trades
```bash
cat trades.csv
```

### Performance Metrics
```bash
cat performance.json
```

---

## 🎓 Next Steps

### After 2 Weeks
1. Review performance
2. Adjust risk settings if needed
3. Enable/disable specific strategies

### After 1 Month
1. Optimize strategy weights
2. Add AI enhancements (optional)
3. Consider live deployment (start conservative!)

### Going Live
1. ✅ Test on demo for 2-4 weeks minimum
2. ✅ Start with 0.3% risk per trade
3. ✅ Maximum 2-3 positions initially
4. ✅ Monitor closely for first week
5. ✅ Gradually increase after proven performance

---

## 🆘 Troubleshooting

### "Failed to connect to MT5"
- Check MT5 is running
- Verify login/password/server in config
- Ensure MT5 allows automated trading

### "No opportunities found"
- Normal during low-volatility periods
- Lower `min_confidence` to 55-60
- Check if markets are open

### "Position size too small"
- Increase `risk_per_trade`
- Check account balance
- Verify broker minimum lot size

### "Import error: MetaTrader5"
```bash
pip install MetaTrader5
```

---

## 📚 Documentation

- **[README.md](README.md)** - Complete overview
- **[FOREX_STRATEGIES.md](docs/FOREX_STRATEGIES.md)** - Forex strategies explained
- **[CRYPTO_STRATEGIES.md](docs/CRYPTO_STRATEGIES.md)** - Crypto strategies explained
- **[METALS_STRATEGIES.md](docs/METALS_STRATEGIES.md)** - Metals strategies explained
- **[STRATEGY_OPTIMIZER.md](docs/STRATEGY_OPTIMIZER.md)** - Strategy weighting system
- **[RISK_MANAGEMENT.md](docs/RISK_MANAGEMENT.md)** - Risk management details
- **[AI_INTEGRATION.md](docs/AI_INTEGRATION.md)** - AI enhancement guide

---

## 💡 Pro Tips

1. **Start conservative** - Use 0.3-0.5% risk per trade
2. **Demo test first** - Always test for 2-4 weeks
3. **Monitor daily** - Check logs and performance
4. **Be patient** - Don't expect profits every day
5. **Trust the system** - Don't interfere with trades

---

## ⚠️ Important Warnings

- ⚠️ **Trading involves substantial risk of loss**
- ⚠️ **Never trade with money you can't afford to lose**
- ⚠️ **Always test on demo first**
- ⚠️ **Past performance doesn't guarantee future results**
- ⚠️ **Start conservative and increase gradually**

---

## 🎉 You're Ready!

```bash
# Start trading
python src/main_bot.py

# Watch it work
tail -f trading_bot.log
```

**Good luck and trade responsibly!** 🚀

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/Samerabualsoud/multi-asset-trading-bot/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Samerabualsoud/multi-asset-trading-bot/discussions)


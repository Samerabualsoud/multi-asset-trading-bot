# Position Sizing Fix

## 🔴 Problem

The bot was using **fixed 0.01 lot** regardless of account size:
- Account: **$865,000**
- Position: **0.01 lot** 
- Risk: **~$0.30 per pip** = **$9 on 30 pip SL**
- **That's only 0.001% risk!** Way too small!

---

## ✅ Solution

Now calculates position size based on:
1. **Account balance**
2. **Risk percentage** (default 0.5%)
3. **Stop loss in pips**

---

## 📊 Calculation Formula

```
Risk Amount = Account Balance × Risk %
Lot Size = Risk Amount / (SL in pips × Pip Value per Lot)
```

### Example: $865,000 Account

**Settings:**
- Account: $865,000
- Risk: 0.5% = $4,325
- SL: 30 pips
- Symbol: AUDUSD

**Calculation:**
```
Pip value per lot (AUDUSD) = 100,000 × 0.0001 = $10
Lot size = $4,325 / (30 pips × $10) = 14.42 lots
```

**Result:** **14.42 lots** instead of 0.01 lot!

---

## 💰 Before vs After

### Before (WRONG)
```
Account: $865,000
Lot size: 0.01 (fixed)
Risk per trade: $9 (0.001%)
```

**Problem:** Massively under-utilizing capital!

### After (CORRECT)
```
Account: $865,000
Risk: 0.5% = $4,325
SL: 30 pips
Lot size: 14.42 lots
Actual risk: $4,326 (0.5%)
```

**Perfect!** ✅

---

## ⚙️ Configuration

Edit `config/config.yaml`:

### Conservative (0.3%)
```yaml
risk_management:
  risk_per_trade: 0.003  # 0.3%
```

**Result:** ~8.65 lots on $865K account

### Moderate (0.5% - Default)
```yaml
risk_management:
  risk_per_trade: 0.005  # 0.5%
```

**Result:** ~14.42 lots on $865K account

### Aggressive (1.0%)
```yaml
risk_management:
  risk_per_trade: 0.01  # 1.0%
```

**Result:** ~28.83 lots on $865K account

---

## 📊 Position Sizing Examples

### Different Account Sizes

**$10,000 account, 0.5% risk, 30 pip SL:**
```
Risk: $50
Lot size: 0.17 lots
```

**$100,000 account, 0.5% risk, 30 pip SL:**
```
Risk: $500
Lot size: 1.67 lots
```

**$865,000 account, 0.5% risk, 30 pip SL:**
```
Risk: $4,325
Lot size: 14.42 lots
```

**$1,000,000 account, 0.5% risk, 30 pip SL:**
```
Risk: $5,000
Lot size: 16.67 lots
```

---

## 🎯 What You'll See in Logs

```
📊 Position sizing:
   Account balance: $865,000.00
   Risk per trade: 0.50% = $4,325.00
   SL: 30 pips
   Calculated lot size: 14.42

✅ Order executed: SELL 14.42 AUDUSD
   Entry: 0.65105
   SL: 0.65405 (30 pips)
   TP: 0.64505 (60 pips)
   Risk:Reward = 1:2.0
```

---

## 🛡️ Safety Features

### 1. Broker Limits
```python
min_lot = symbol_info.volume_min  # e.g., 0.01
max_lot = symbol_info.volume_max  # e.g., 100.0
lot = max(min_lot, min(lot, max_lot))
```

Bot respects broker's min/max lot sizes.

### 2. Lot Step Rounding
```python
lot_step = symbol_info.volume_step  # e.g., 0.01
lot = round(lot / lot_step) * lot_step
```

Rounds to broker's allowed increments (usually 0.01).

### 3. Maximum Positions
```yaml
risk_management:
  max_positions: 5
```

Limits total open positions.

---

## 💡 Risk Management Tips

### 1. Start Conservative
```yaml
risk_per_trade: 0.003  # 0.3%
```

Test for 2-4 weeks before increasing.

### 2. Account for Correlation
If trading multiple correlated pairs:
```yaml
risk_per_trade: 0.003  # Lower risk
max_positions: 3       # Fewer positions
```

### 3. Adjust for Volatility
High volatility periods:
```yaml
risk_per_trade: 0.003  # Reduce risk
```

Low volatility periods:
```yaml
risk_per_trade: 0.005  # Normal risk
```

### 4. Never Risk More Than 2% Total
```yaml
risk_per_trade: 0.005  # 0.5%
max_positions: 3       # Max 1.5% total risk
```

---

## 🔧 How to Update

### Windows:
```cmd
cd C:\Users\aa\multi-asset-trading-bot
git pull
python src\main_bot.py
```

---

## 📈 Expected Results

### $865,000 Account Example

**Per Trade (0.5% risk):**
```
Risk: $4,325
Lot size: ~14 lots
Potential profit (60 pips): $8,400
Potential loss (30 pips): $4,200
```

**Monthly (20 trades, 70% win rate):**
```
Wins: 14 × $8,400 = $117,600
Losses: 6 × $4,200 = $25,200
Net profit: $92,400 (10.7% monthly)
```

**Much better than 0.01 lot!** 🚀

---

## ⚠️ Important Notes

### 1. Demo Test First
Always test position sizing on demo before live!

### 2. Check Broker Limits
Some brokers limit:
- Maximum lot size per order
- Maximum total exposure
- Maximum positions

### 3. Margin Requirements
Ensure sufficient margin:
```
Required margin = Lot size × Contract size × Price / Leverage
```

For 14 lots AUDUSD with 1:100 leverage:
```
Margin = 14 × 100,000 × 0.65 / 100 = $9,100
```

### 4. Start Small
Even with large account, start with 0.3% risk to test the system.

---

## ✅ Summary

**Fixed:**
- ✅ Dynamic position sizing based on account balance
- ✅ Proper risk percentage calculation
- ✅ Respects broker limits
- ✅ Clear logging of position size calculation

**Before:** 0.01 lot (0.001% risk)  
**After:** 14.42 lots (0.5% risk)

**Update now for proper position sizing!** 🚀


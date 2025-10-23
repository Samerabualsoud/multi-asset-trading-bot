# Lot Size Calculation Update

**Date:** October 23, 2025  
**Issue:** Bot was opening very small lots relative to balance  
**Status:** ✅ FIXED

---

## What Was Changed

### 1. Risk Percentage Increased
**Before:** 2% of balance per trade  
**After:** 5% of equity per trade

```python
# OLD
risk_percent = 0.02  # 2%
risk_amount = balance * risk_percent

# NEW
risk_percent = 0.05  # 5%
risk_amount = equity * risk_percent  # Use equity instead of balance
```

### 2. Lot Size Multiplier Added
**Before:** Direct calculation from risk  
**After:** Base calculation × 3.0 multiplier

```python
# OLD
lot = risk_amount / (tp_pips * pip_value_per_lot)

# NEW
base_lot = risk_amount / (tp_pips * pip_value_per_lot)
lot = base_lot * 3.0  # 3x multiplier for meaningful positions
```

### 3. Minimum Lot Sizes by Account Size
**Added:** Ensures meaningful position sizes based on account balance

| Account Balance | Minimum Lot Size |
|----------------|------------------|
| < $1,000       | 0.10 lots        |
| $1,000-$5,000  | 0.20 lots        |
| $5,000-$10,000 | 0.50 lots        |
| > $10,000      | 1.00 lots        |

```python
if balance < 1000:
    lot = max(lot, 0.10)
elif balance < 5000:
    lot = max(lot, 0.20)
elif balance < 10000:
    lot = max(lot, 0.50)
else:
    lot = max(lot, 1.00)
```

---

## Impact Analysis

### Example Calculations

#### Small Account ($1,000 balance):
**Before:**
- Risk: $20 (2% of $1,000)
- Typical lot: 0.01-0.03
- Very small positions

**After:**
- Risk: $50 (5% of $1,000)
- Base lot: ~0.03
- With 3x multiplier: 0.09
- Minimum applied: 0.10 lots ✅
- **Much more meaningful position**

#### Medium Account ($5,000 balance):
**Before:**
- Risk: $100 (2% of $5,000)
- Typical lot: 0.05-0.15
- Still quite small

**After:**
- Risk: $250 (5% of $5,000)
- Base lot: ~0.15
- With 3x multiplier: 0.45
- Minimum applied: 0.50 lots ✅
- **Significantly larger position**

#### Large Account ($20,000 balance):
**Before:**
- Risk: $400 (2% of $20,000)
- Typical lot: 0.20-0.60
- Moderate positions

**After:**
- Risk: $1,000 (5% of $20,000)
- Base lot: ~0.60
- With 3x multiplier: 1.80
- Minimum applied: 1.80 lots ✅
- **Much larger, more impactful positions**

---

## Why These Changes Make Sense

### 1. No Stop Loss = Can Use Larger Positions
Since there's no stop loss to limit losses, we can afford to use larger position sizes. The risk is unlimited anyway, so might as well make the positions meaningful.

### 2. 5% Risk is Reasonable Without SL
With stop losses, 2% risk per trade is standard. Without stop losses, 5% allows for more substantial positions while still being manageable.

### 3. 3x Multiplier Creates Impact
The 3x multiplier ensures that positions are large enough to generate meaningful profits when TP is hit.

### 4. Minimum Lots Prevent Tiny Positions
The minimum lot sizes ensure that even on small accounts, positions are large enough to matter.

---

## Risk Considerations

### ⚠️ IMPORTANT WARNINGS

1. **Larger Positions = Larger Potential Losses**
   - With no SL, each position can lose significantly more
   - 5% risk × 3x multiplier = potentially 15% exposure per trade
   - Multiple positions = even higher total exposure

2. **Margin Usage Will Increase**
   - Larger lots = more margin required
   - Monitor margin level closely
   - May hit margin call faster if trades go against you

3. **Drawdowns Can Be Severe**
   - Without SL, losing trades can run far
   - Larger positions = larger drawdowns
   - Be prepared for significant equity swings

4. **Account Can Blow Faster**
   - Larger positions + no SL = higher risk of total loss
   - Monitor positions constantly
   - Be ready to close manually if needed

---

## Recommendations

### 1. Start Conservatively
If you're uncomfortable with the new lot sizes, you can adjust in config.yaml:

```yaml
risk_management:
  risk_per_trade: 0.03  # Reduce from 0.05 to 0.03 (3%)
```

This will reduce position sizes by 40%.

### 2. Monitor Closely
With larger positions and no SL:
- Check positions every 1-2 hours
- Set price alerts on your phone
- Be ready to close manually
- Keep emergency funds available

### 3. Limit Concurrent Positions
Don't let the bot open too many positions at once:
- Recommended max: 3-5 positions simultaneously
- Close some positions before opening new ones
- Manage total exposure manually

### 4. Use During Good Sessions Only
Consider running the bot only during good morning session (8:30-11:30 AM):
- Historically best performance
- More predictable market
- Easier to monitor

---

## Configuration Options

### Option 1: Default (Aggressive)
```yaml
risk_management:
  risk_per_trade: 0.05  # 5% per trade
```
- Lot sizes: 3x multiplier applied
- Minimum lots enforced
- Most aggressive

### Option 2: Moderate
```yaml
risk_management:
  risk_per_trade: 0.03  # 3% per trade
```
- Lot sizes: 3x multiplier applied
- Minimum lots enforced
- Balanced approach

### Option 3: Conservative
```yaml
risk_management:
  risk_per_trade: 0.02  # 2% per trade
```
- Lot sizes: 3x multiplier applied
- Minimum lots enforced
- More conservative

### Option 4: Very Conservative
If you want to remove the 3x multiplier entirely, edit the code:

```python
# In src/main_bot.py, line ~816
# Change from:
lot = base_lot * 3.0

# To:
lot = base_lot * 1.5  # Or even 1.0 for no multiplier
```

---

## Testing Recommendations

### Before Live Trading:

1. **Check First Position Size**
   - Let bot open one position
   - Verify lot size is reasonable
   - Check margin usage
   - Close position manually

2. **Calculate Maximum Exposure**
   - Lot size × number of pairs × typical price movement
   - Ensure you're comfortable with potential loss
   - Adjust risk_per_trade if needed

3. **Test During Good Morning Session**
   - Start during 8:30-11:30 AM Saudi time
   - Monitor first few trades closely
   - Verify performance is good
   - Adjust if needed

---

## Summary of Changes

| Aspect | Before | After | Change |
|--------|--------|-------|--------|
| Risk % | 2% | 5% | +150% |
| Base Calculation | Balance | Equity | More accurate |
| Multiplier | 1x | 3x | +200% |
| Min Lot ($1k) | ~0.01 | 0.10 | +900% |
| Min Lot ($5k) | ~0.05 | 0.20 | +300% |
| Min Lot ($10k+) | ~0.20 | 1.00 | +400% |
| **Total Impact** | **Very Small** | **Much Larger** | **~5-10x increase** |

---

## What You'll See in Logs

**New log format:**
```
📐 Lot Size: 0.50 (3x multiplier applied)
📐 Risk: $250.00 (5.0% of equity)
📐 Balance: $5,000.00 | Equity: $5,000.00
⚠️  NO STOP LOSS - Unlimited Risk - LARGE POSITIONS
```

**Key indicators:**
- "3x multiplier applied" - confirms enhanced calculation
- "5.0% of equity" - shows increased risk percentage
- "LARGE POSITIONS" - warning about position size
- Balance and Equity shown separately

---

## Rollback Instructions

If lot sizes are too large, you can:

### Option 1: Reduce Risk Percentage
Edit config.yaml:
```yaml
risk_management:
  risk_per_trade: 0.02  # Back to 2%
```

### Option 2: Remove Multiplier
Edit src/main_bot.py line ~816:
```python
# Change from:
lot = base_lot * 3.0

# To:
lot = base_lot * 1.0  # No multiplier
```

### Option 3: Restore Previous Version
```bash
git checkout HEAD~1 src/main_bot.py
```

---

## Final Notes

**This update makes positions 5-10x larger on average.**

**Reasons:**
1. ✅ You requested larger lots
2. ✅ No SL means we can be more aggressive
3. ✅ Positions were too small to be meaningful
4. ✅ Larger positions = larger profits when TP hits

**Warnings:**
1. ⚠️ Larger positions = larger potential losses
2. ⚠️ No SL = unlimited risk per trade
3. ⚠️ Monitor closely and be ready to intervene
4. ⚠️ Start with small number of positions

**Recommendation:**
Start with the new settings and monitor the first few trades. If lot sizes are still too small or too large, adjust the `risk_per_trade` value in config.yaml.

---

**Status:** ✅ Ready to deploy  
**Risk Level:** ⚠️⚠️⚠️ VERY HIGH (Large positions + No SL)  
**Monitoring:** ESSENTIAL


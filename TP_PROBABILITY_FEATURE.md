# TP Probability Analysis Feature

## Overview

The bot now **intelligently analyzes each open position** to predict the probability of hitting TP (Take Profit) vs SL (Stop Loss).

## New Table Columns

Your position monitoring table now includes:

| Column | Description |
|--------|-------------|
| **TP Prob** | Probability (0-100%) that position will hit TP |
| **Analysis** | Why the probability is high/low (trend, momentum, RSI) |
| **Action** | Recommended action based on analysis |

## Example Output

```
+----------+--------+------+------+----------+----------+-----------+--------+---------+--------------------------------+-------------------------+
| Ticket   | Symbol | Type | Lot  | Open     | Current  | P/L       | Status | TP Prob | Analysis                       | Action                  |
+==========+========+======+======+==========+==========+===========+========+=========+================================+=========================+
| 77587380 | USDJPY | SELL | 19.75| 151.101  | 151.779  | $-8822.37 | ❌ Loss | 25%     | Against trend, Momentum +1.2%  | ❌ CLOSE (Very low TP)  |
+----------+--------+------+------+----------+----------+-----------+--------+---------+--------------------------------+-------------------------+
| 77590890 | USDJPY | SELL | 20.09| 151.190  | 151.779  | $-7796.21 | ❌ Loss | 30%     | Against trend, Near SL (15%)   | ⚠️ WATCH (Low TP prob)  |
+----------+--------+------+------+----------+----------+-----------+--------+---------+--------------------------------+-------------------------+
| 77594216 | USDCHF | SELL | 50.00| 0.79383  | 0.79593  | $-13192   | ❌ Loss | 45%     | Downtrend, Momentum -0.3%      | ⏸️ HOLD (Neutral)       |
+----------+--------+------+------+----------+----------+-----------+--------+---------+--------------------------------+-------------------------+
| 77624989 | USDJPY | BUY  | 29.49| 151.932  | 151.763  | $-3283.94 | ❌ Loss | 72%     | Strong uptrend, RSI oversold   | 🎯 HOLD (High TP prob)  |
+----------+--------+------+------+----------+----------+-----------+--------+---------+--------------------------------+-------------------------+
```

## How TP Probability is Calculated

The bot analyzes **4 key factors**:

### 1. Trend Alignment (±20%)

**For BUY positions:**
- Price > MA20 > MA50 → +20% ("Strong uptrend")
- Price > MA20 → +10% ("Uptrend")
- Price < MA20 → -15% ("Against trend")

**For SELL positions:**
- Price < MA20 < MA50 → +20% ("Strong downtrend")
- Price < MA20 → +10% ("Downtrend")
- Price > MA20 → -15% ("Against trend")

### 2. Momentum Alignment (±15%)

**For BUY positions:**
- Momentum > +0.5% → +15% ("Momentum +X%")
- Momentum < -0.5% → -10% ("Momentum -X%")

**For SELL positions:**
- Momentum < -0.5% → +15% ("Momentum -X%")
- Momentum > +0.5% → -10% ("Momentum +X%")

### 3. RSI (±10%)

**For BUY positions:**
- RSI < 40 → +10% ("RSI oversold")
- RSI > 70 → -10% ("RSI overbought")

**For SELL positions:**
- RSI > 60 → +10% ("RSI overbought")
- RSI < 30 → -10% ("RSI oversold")

### 4. Progress to TP (±10%)

- Progress > 60% → +10% ("Near TP")
- Progress < 20% → -5% ("Near SL")

## Recommended Actions

| TP Probability | Action | Meaning |
|----------------|--------|---------|
| **70-100%** | 🎯 HOLD (High TP prob) | Strong chance of hitting TP, keep position |
| **50-69%** | ⏸️ HOLD (Neutral) | Moderate chance, monitor closely |
| **30-49%** | ⚠️ WATCH (Low TP prob) | Low chance, consider closing if conditions worsen |
| **0-29%** | ❌ CLOSE (Very low TP prob) | Very low chance, consider closing to minimize loss |

## Real-World Example

Looking at your positions from the screenshot:

### Position 1: USDJPY SELL @ 151.101
- **Current:** 151.779 (moving AGAINST you)
- **P/L:** -$8,822
- **TP Prob:** ~25% (estimated)
- **Analysis:** "Against trend, Momentum +1.2%"
- **Action:** ❌ CLOSE - Price is in uptrend, momentum positive, position unlikely to recover

### Position 2: USDJPY BUY @ 151.932
- **Current:** 151.763 (slight loss)
- **P/L:** -$3,283
- **TP Prob:** ~72% (estimated)
- **Analysis:** "Strong uptrend, RSI oversold"
- **Action:** 🎯 HOLD - Good chance of recovery and hitting TP

## How to Use This Information

### High TP Probability (70%+)
✅ **Keep the position** - Market conditions favor your trade

### Medium TP Probability (50-69%)
⏸️ **Monitor closely** - Could go either way, watch for changes

### Low TP Probability (30-49%)
⚠️ **Consider partial close** - Reduce lot size to minimize risk

### Very Low TP Probability (0-29%)
❌ **Close position** - Cut losses before they grow

## Summary Statistics

At the end of the position table, you'll see:

```
💰 Total P/L: $-45,234.56 | Profitable: 5 | Losing: 29
```

This gives you a quick overview of your overall position health.

## Benefits

1. **Data-driven decisions** - Not based on emotions
2. **Early warning system** - Identify losing trades before they get worse
3. **Confidence in winners** - Know which positions to hold
4. **Risk management** - Close low-probability positions early

## Next Steps

1. Pull the latest code: `git pull`
2. Restart the bot: `python src/main_bot.py`
3. Watch the enhanced position monitoring table
4. Use TP Prob % and Action to make informed decisions

---

**The bot now thinks for you!** 🧠


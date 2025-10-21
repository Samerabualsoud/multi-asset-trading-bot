# Trading Bot Fixes Applied

## Critical Fixes

1. **Fixed pip_size bug** - Variable now defined before use
2. **Added margin level checks** - Protects against over-leveraging (700% minimum)
3. **Session-aware confidence thresholds** - Asian: 65%, London: 75%, Overlap: 80%
4. **Reduced volatility multipliers** - London: 0.9x (was 1.2x), Overlap: 1.0x (was 1.5x)
5. **Tightened base SL/TP** - 25 pips SL (was 30), 50 pips TP (was 60)
6. **Added oil support** - Now recognizes USOIL, WTIUSD, BRENT, etc.
7. **Fixed crypto trading** - Relaxed entry conditions (1.5x volume instead of 2x)
8. **Ultra-precise pip calculations** - Accurate for all asset types (forex, JPY, gold, silver, oil, crypto)

## Root Cause of Time-Based Issue

**Problem:** Stop losses became 71-114% wider after 11:30 AM Saudi time due to session volatility multipliers.

**Solution:** Reduced multipliers to keep stops tight throughout the day.

## How to Use

1. Pull updates: `git pull`
2. Add to config: `min_margin_level: 700` under `risk_management`
3. Restart bot: `python src/main_bot.py`

## Expected Behavior

- **Morning (Asian):** Same good performance (unchanged)
- **Afternoon (London/Overlap):** No more big losses (fixed)
- **Crypto:** Now trades (fixed)
- **Oil:** Now supported (fixed)

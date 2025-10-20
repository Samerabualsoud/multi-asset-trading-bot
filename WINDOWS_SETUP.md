# Windows Setup Guide

## 🪟 Quick Start for Windows Users

### Step 1: Install Python
1. Download Python 3.11+ from https://www.python.org/downloads/
2. **Important:** Check "Add Python to PATH" during installation
3. Verify installation:
```cmd
python --version
```

### Step 2: Clone Repository
```cmd
git clone https://github.com/Samerabualsoud/multi-asset-trading-bot.git
cd multi-asset-trading-bot
```

### Step 3: Install Dependencies
```cmd
pip install -r requirements.txt
```

**If you get an error about ta-lib:**
- It's optional, the bot works without it
- Or install from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

### Step 4: Configure
```cmd
:: Copy example config (Windows command)
copy config\config.example.yaml config\config.yaml

:: Edit with Notepad
notepad config\config.yaml
```

**Minimum configuration:**
```yaml
# MT5 Connection
mt5_login: YOUR_LOGIN_NUMBER
mt5_password: "YOUR_PASSWORD"
mt5_server: "YOUR_BROKER-Server"
mt5_path: "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

# Symbols
symbols:
  - EURUSD
  - GBPUSD
  - BTCUSD
  - XAUUSD

# Risk
risk_management:
  risk_per_trade: 0.005
  max_positions: 5
```

### Step 5: Run Bot
```cmd
python src\main_bot.py
```

---

## 🔧 Windows-Specific Notes

### MT5 Path
On Windows, use double backslashes:
```yaml
mt5_path: "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
```

Or single forward slashes:
```yaml
mt5_path: "C:/Program Files/MetaTrader 5/terminal64.exe"
```

### File Paths
Windows uses backslashes:
```cmd
:: Correct
copy config\config.example.yaml config\config.yaml
python src\main_bot.py

:: Not forward slashes like Linux
```

### Common Errors

**Error: "The syntax of the command is incorrect"**
- You're using Linux commands on Windows
- Use `copy` instead of `cp`
- Use `\` instead of `/` in paths

**Error: "ta-lib" installation failed**
- It's optional, bot works without it
- Or download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
- Install with: `pip install TA_Lib-0.4.28-cp311-cp311-win_amd64.whl`

**Error: "MetaTrader5 module not found"**
```cmd
pip install MetaTrader5
```

**Error: "Failed to connect to MT5"**
- Make sure MT5 is running
- Check MT5 path in config
- Enable "Allow automated trading" in MT5 settings

---

## 📁 Directory Structure (Windows)

```
C:\Users\aa\multi-asset-trading-bot\
│
├── config\
│   ├── config.yaml              <- Your config (create this)
│   ├── config.example.yaml      <- Example
│   └── strategy_weights.yaml
│
├── src\
│   ├── main_bot.py              <- Run this
│   ├── core\
│   ├── strategies\
│   └── utils\
│
├── docs\
└── README.md
```

---

## 🚀 Running the Bot

### Method 1: Command Prompt
```cmd
cd C:\Users\aa\multi-asset-trading-bot
python src\main_bot.py
```

### Method 2: PowerShell
```powershell
cd C:\Users\aa\multi-asset-trading-bot
python src\main_bot.py
```

### Method 3: Double-click
1. Create `run_bot.bat`:
```batch
@echo off
cd /d "%~dp0"
python src\main_bot.py
pause
```
2. Double-click `run_bot.bat`

---

## 🛑 Stopping the Bot

Press `Ctrl+C` in the command window

---

## 📊 Monitoring

### View Logs
```cmd
type trading_bot.log
```

### View Trades
```cmd
type trades.csv
```

### Real-time Monitoring
```cmd
:: Install
pip install colorlog

:: Run bot (will show colored logs)
python src\main_bot.py
```

---

## 🎯 Next Steps

1. ✅ Run on demo account for 2-4 weeks
2. ✅ Monitor performance daily
3. ✅ Adjust risk settings if needed
4. ✅ Deploy to live (start conservative)

---

## 💡 Pro Tips for Windows

### 1. Use Windows Terminal
- Modern, better than CMD
- Download from Microsoft Store
- Supports tabs and colors

### 2. Create Desktop Shortcut
1. Right-click `run_bot.bat`
2. Send to → Desktop (create shortcut)
3. Double-click to start bot

### 3. Run on Startup (Optional)
1. Press `Win+R`
2. Type `shell:startup`
3. Copy `run_bot.bat` shortcut here
4. Bot starts when Windows boots

### 4. Use VPS
For 24/7 trading:
- Rent Windows VPS
- Install MT5 and bot
- Run continuously

---

## 🆘 Troubleshooting

### Python Not Found
```cmd
:: Add Python to PATH manually
setx PATH "%PATH%;C:\Users\aa\AppData\Local\Programs\Python\Python311"
```

### Permission Denied
- Run Command Prompt as Administrator
- Right-click → Run as administrator

### MT5 Not Connecting
1. Open MT5 manually
2. Login to your account
3. Tools → Options → Expert Advisors
4. Check "Allow automated trading"
5. Restart bot

---

## 📞 Support

- **Issues:** https://github.com/Samerabualsoud/multi-asset-trading-bot/issues
- **Discussions:** https://github.com/Samerabualsoud/multi-asset-trading-bot/discussions

---

## ✅ Quick Command Reference

```cmd
:: Navigate
cd multi-asset-trading-bot

:: Install
pip install -r requirements.txt

:: Configure
copy config\config.example.yaml config\config.yaml
notepad config\config.yaml

:: Run
python src\main_bot.py

:: Stop
Ctrl+C

:: View logs
type trading_bot.log
```

**You're ready to trade on Windows!** 🚀


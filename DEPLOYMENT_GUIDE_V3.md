# Complete Deployment Guide for ML+LLM Trading Bot V3

**Author:** Manus AI
**Date:** 2025-10-23

---

## 1. Overview

This guide provides comprehensive instructions for deploying and running the **ML+LLM Trading Bot V3**, which now includes a powerful **auto-discovery system** for identifying and trading all available symbols on your MT5 account.

The new system is composed of three main components:

1.  **Enhanced Symbol Discovery (`symbol_discovery_enhanced.py`)**: A script to scan your MT5 account, identify all tradeable symbols, and filter them based on liquidity, volume, and volatility. It produces a detailed report and can automatically update your configuration.

2.  **Auto-Retrain System V2 (`auto_retrain_system_v2.py`)**: An automated system that fetches the symbol list (either from auto-discovery or a manual list) and trains/retrains ML models for every symbol. It runs on a 12-hour schedule to keep the models fresh.

3.  **ML+LLM Trading Bot V3 (`ml_llm_trading_bot_v3.py`)**: The main trading bot. It loads all the trained models and dynamically trades the symbols based on ML predictions and LLM validation.

## 2. Installation

First, ensure you have all the required Python libraries installed. If you have an existing environment, you may only need to add a few new ones.

```bash
# It is recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate  # On Windows

# Install all required packages
pip install -r requirements.txt
```

If you don\'t have a `requirements.txt` file, install the packages manually:

```bash
pip install MetaTrader5 pandas numpy pyyaml scikit-learn openai
```

## 3. Configuration (`config/config_v3.yaml`)

A new configuration file, `config_v3.yaml`, has been created with settings for the new system. **Please rename it to `config.yaml` or update the scripts to point to this new file.**

Key new settings:

-   `auto_discover_symbols` (boolean): 
    -   Set to `true` to enable the bot to automatically discover and trade all symbols found by the discovery script.
    -   Set to `false` to manually specify which symbols to trade using the `symbols` list.

-   `discovery_mode` (string):
    -   Controls the filtering strictness during symbol discovery. Options are:
        -   `conservative`: Major forex pairs and metals with low spread and high volume.
        -   `balanced`: A mix of forex, crypto, metals, and indices. **(Recommended)**
        -   `aggressive`: Includes exotic pairs and commodities with looser filters.
        -   `all`: All tradeable symbols, regardless of type or liquidity.

**Action Required:**

1.  Open `config/config_v3.yaml`.
2.  Enter your **MT5 login credentials** and your **DeepSeek API key**.
3.  Review the `auto_discover_symbols` and `discovery_mode` settings to match your preference.
4.  Save the file as `config.yaml` in the root directory of the project.

## 4. Execution Flow (3 Simple Steps)

Follow these steps in order to run the system.

### Step 1: Discover Symbols (Optional but Recommended)

Before training, you can run the discovery script to see which symbols will be used. This script will print a summary table and save a detailed `symbol_discovery_report.csv`.

```bash
# Run this command in your terminal
python src/symbol_discovery_enhanced.py --mode balanced --report
```

-   `--mode`: Can be `conservative`, `balanced`, `aggressive`, or `all`.
-   `--report`: Generates a detailed CSV file of all discovered symbols and their stats.

This step does **not** train any models. It only shows you what the auto-retrain system will work on.

### Step 2: Train the Models

This is the most critical step. The `auto_retrain_system_v2.py` script will either discover symbols or use your manual list, and then train an ML model for each one. **This will take a long time for the first run**, especially with hundreds of symbols (expect 1-3 hours).

```bash
# Run this command to start the initial training
python src/auto_retrain_system_v2.py --once
```

-   `--once`: This flag tells the system to run the training cycle just once and then exit. 

After this completes, you will have a `ml_models_simple/` directory full of `.pkl` models, ready for the trading bot.

To run the system continuously (so it retrains every 12 hours), simply run it without the flag:

```bash
# This will run forever and retrain every 12 hours
python src/auto_retrain_system_v2.py
```

### Step 3: Run the Trading Bot

Once the models are trained, you can start the trading bot. It will automatically load all the models it finds in the `ml_models_simple/` directory and begin scanning the market.

```bash
# Run the main trading bot
python src/ml_llm_trading_bot_v3.py
```

**Important:**

-   Make sure **"Allow algorithmic trading"** is enabled in your MT5 terminal (Tools -> Options -> Expert Advisors).
-   The bot will log its actions to `ml_llm_bot_v3.log`.

## 5. Pushing to GitHub

All new and updated files have been added to your local repository. Once you have tested the system and are satisfied with its performance, you can commit and push the changes to your GitHub repository.

```bash
git add .
git commit -m "feat: Implement auto-discovery and V3 trading system"
git push
```

---

This completes the deployment of the V3 system. Monitor the bot\'s log file closely during the initial runs to ensure everything is working as expected.


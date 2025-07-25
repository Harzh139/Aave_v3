# Compound Wallet Risk Scoring System

## Overview
This project analyzes Ethereum wallet addresses that have interacted with the Compound protocol and assigns each a risk score (0-1000) based on their on-chain activity. The results help identify wallets with higher risk profiles for DeFi protocols, risk managers, or researchers.

---

## 1. Data Collection Method
- **Source:**
  - Wallet addresses are loaded from a Google Sheet or Excel file provided by the user.
  - The code is set up to fetch wallet addresses from a Google Sheet link (CSV export) or a local Excel file.
- **Transaction Data:**
  - For each wallet, transaction history is (simulated in this demo) but in production should be fetched from blockchain APIs (e.g., Etherscan, The Graph, or direct node queries).
  - Each transaction includes type (borrow, repay, mint, redeem, liquidate), amount, token, and timestamp.

---

## 2. Feature Selection Rationale
The following features are selected for their direct impact on risk in DeFi lending/borrowing:
- **Liquidation History:** Indicates past risk events where the wallet's collateral was insufficient.
- **Leverage Ratio:** High leverage increases risk of liquidation.
- **Health Factor Violations:** Simulated metric for how often the wallet's health factor dropped below safe levels.
- **Transaction Volatility:** High frequency of transactions may indicate risky or automated strategies.
- **Portfolio Concentration:** Wallets holding only a few tokens are more exposed to price swings in those assets.
- **Large Transaction Count:** Frequent large transactions may signal manipulation or high-risk strategies.
- **Activity Pattern:** New or inactive wallets are less predictable and may be riskier.

These features are chosen because they are observable on-chain, quantifiable, and have a clear relationship to risk in lending protocols.

---

## 3. Scoring Method
- Each feature is assigned a maximum number of points, with higher values indicating higher risk:
  - **Liquidation History:** Up to 300 points (100 per event, max 3 events)
  - **Leverage Ratio:** Up to 200 points (50 per leverage unit, max 4)
  - **Health Factor Violations:** Up to 150 points (15 per violation, max 10)
  - **Transaction Volatility:** Up to 100 points (scaled by activity)
  - **Portfolio Concentration:** 100 points for <=2 tokens, 50 for 3 tokens, 0 otherwise
  - **Large Transaction Count:** Up to 100 points (10 per event, max 10)
  - **Activity Pattern:** 50 points for <30 days active, 0 for >365 days, linearly interpolated
- The total risk score is the sum of these, capped at 1000.
- Higher scores mean higher risk.

---

## 4. Justification of the Risk Indicators Used
- **Liquidation History:** Past liquidations are strong indicators of risky behavior or poor risk management.
- **Leverage Ratio:** High leverage increases the chance of liquidation during market volatility.
- **Health Factor Violations:** Frequent violations suggest the wallet often operates near risk thresholds.
- **Transaction Volatility:** High activity can indicate aggressive strategies or bots, which may be riskier.
- **Portfolio Concentration:** Lack of diversification increases exposure to single-asset risk.
- **Large Transaction Count:** Large, infrequent transactions may be attempts to game the system or signal high-stakes strategies.
- **Activity Pattern:** New or sporadically active wallets are less predictable and may be used for one-off risky actions.

These indicators are widely recognized in DeFi risk research and are supported by both academic and industry analyses as meaningful predictors of risk.

---

## Usage
1. Place your wallet addresses in a Google Sheet or Excel file.
2. Update the `EXCEL_URL` or `EXCEL_FILE_PATH` in `main.py`.
3. Run the script: `python main.py`
4. Review the output CSV and report files for results and analysis.

---

## References
- [Compound Protocol Documentation](https://compound.finance/docs)
- [DeFi Risk Research](https://arxiv.org/abs/2107.09654)
- [Etherscan API](https://docs.etherscan.io/) 
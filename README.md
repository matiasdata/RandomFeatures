# Random Features

This project implements a random features based stochastic discount factor model.

To be able to run this script you need access to WRDS for accessing CRSP and the Financial Ratios by WRDS database.

Specifically, you'll need:
1) CRSP Monthly stock data, available at: https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/quarterly-update/stock-security-files/monthly-stock-file/
   (Columns needed: "permno", "date", "ticker", "cusip", "dlstcd", "dlret", "prc",
 "ret", "shrout", "spread".)
2) Financial Ratios Firm Level by WRDS, available at: https://wrds-www.wharton.upenn.edu/pages/get-data/financial-ratios-suite-wrds/financial-ratios/financial-ratios-firm-level-by-wrds-beta/
   ("permno" plus all characteristics columns)
3) Market Capitalization data (NYSE/AMEX/NASDAQ), available at: https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/annual-update/index-stock-file-indexes/market-cap-monthly/
   (only Total Market Value, i.e. "totval" column, is used)

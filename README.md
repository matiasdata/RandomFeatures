# Random Features

This project implements a random features based stochastic discount factor model.

To run it with the Global Factor Data from https://jkpfactors.com/ you'll need access to WRDS, and run the notebook "pre_processing_jkp.ipynb" (to compute the instruments from the raw characteristics) followed by the notebook "random_features.ipynb".

To be able to run this script with the CRSP/WRDS Financial Ratios, you'll need access to WRDS as well and download the data below. Then, you'll need to run the sql script in "database_script.sql", followed by the notebooks "pre_processing.ipynb" (to compute the instruments from the raw characteristics) and "random_features.ipynb".


1) CRSP Monthly stock data, available at: https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/quarterly-update/stock-security-files/monthly-stock-file/
   
   (Columns needed: "permno", "date", "ticker", "cusip", "dlstcd", "dlret", "prc",
 "ret", "shrout", "spread".)
2) Financial Ratios Firm Level by WRDS, available at: https://wrds-www.wharton.upenn.edu/pages/get-data/financial-ratios-suite-wrds/financial-ratios/financial-ratios-firm-level-by-wrds-beta/
   
   ("permno" plus all characteristics columns)
3) Market Capitalization data (NYSE/AMEX/NASDAQ), available at: https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/annual-update/index-stock-file-indexes/market-cap-monthly/
   
   (only Total Market Value, i.e. "totval" column, is used)

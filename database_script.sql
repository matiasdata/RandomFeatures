/* In the terminal use the following command to create the table */
sqlite3 db2023.db

/* use these sql commands so that a table is shown with header and separated columns */
.headers on
.mode columns

.mode csv
.import Market_2023.csv Market
.import CRSP_Monthly_Stock_2023_delist.csv Prices_old
.import WFR_2023.csv Ratios
.import Fama_French.csv Fama_French

/* In Ratios adate is "Fiscal year end", qdate is "Fiscal quarter end" and public_date is "Date" */

/* Prices contains the following columns PRC (Stock Price), RET (Stock adjusted return), SPREAD (bid-ask spread), SHROUT (Shares Outstanding), MktFlag (defined below) */
/* PRC is the closing price or it is negative if unavailable, in that case (negative) bid-ask average is reported, if that is not available it is zero. */

/* Formatting the Ratios table */

ALTER TABLE Ratios RENAME TO Ratios_old;

CREATE TABLE IF NOT EXISTS "Ratios" (
"gvkey" INTEGER,
  "permno" INTEGER,
  "adate" TEXT,
  "qdate" TEXT,
  "public_date" TEXT,
  "CAPEI" REAL,
  "bm" REAL,
  "evm" REAL,
  "pe_op_basic" REAL,
  "pe_op_dil" REAL,
  "pe_exi" REAL,
  "pe_inc" REAL,
  "ps" REAL,
  "pcf" REAL,
  "dpr" REAL,
  "npm" REAL,
  "opmbd" REAL,
  "opmad" REAL,
  "gpm" REAL,
  "ptpm" REAL,
  "cfm" REAL,
  "roa" REAL,
  "roe" REAL,
  "roce" REAL,
  "efftax" REAL,
  "aftret_eq" REAL,
  "aftret_invcapx" REAL,
  "aftret_equity" REAL,
  "pretret_noa" REAL,
  "pretret_earnat" REAL,
  "GProf" REAL,
  "equity_invcap" REAL,
  "debt_invcap" REAL,
  "totdebt_invcap" REAL,
  "capital_ratio" REAL,
  "int_debt" REAL,
  "int_totdebt" REAL,
  "cash_lt" REAL,
  "invt_act" REAL,
  "rect_act" REAL,
  "debt_at" REAL,
  "debt_ebitda" REAL,
  "short_debt" REAL,
  "curr_debt" REAL,
  "lt_debt" REAL,
  "profit_lct" REAL,
  "ocf_lct" REAL,
  "cash_debt" REAL,
  "fcf_ocf" REAL,
  "lt_ppent" REAL,
  "dltt_be" REAL,
  "debt_assets" REAL,
  "debt_capital" REAL,
  "de_ratio" REAL,
  "intcov" REAL,
  "intcov_ratio" REAL,
  "cash_ratio" REAL,
  "quick_ratio" REAL,
  "curr_ratio" REAL,
  "cash_conversion" REAL,
  "inv_turn" REAL,
  "at_turn" REAL,
  "rect_turn" REAL,
  "pay_turn" REAL,
  "sale_invcap" REAL,
  "sale_equity" REAL,
  "sale_nwc" REAL,
  "rd_sale" REAL,
  "adv_sale" REAL,
  "staff_sale" REAL,
  "accrual" REAL,
  "ptb" REAL,
  "PEG_trailing" REAL,
  "divyield" REAL,
  "TICKER" TEXT,
  "cusip" TEXT
);

INSERT INTO Ratios SELECT * FROM Ratios_old;

DROP TABLE Ratios_old;

/* divyield was in percentage format, thus generating NaN after conversion. Here we reformat it to a float number in [0,1]. */
ALTER TABLE Ratios ADD COLUMN divyield_float REAL;

UPDATE Ratios
SET divyield_float = CASE
    WHEN divyield IS NOT NULL AND divyield != '' THEN REPLACE(divyield, '%', '') / 100.0
    ELSE NULL
END;

ALTER TABLE Ratios DROP COLUMN divyield;
ALTER TABLE Ratios RENAME COLUMN divyield_float TO divyield;

/* Formatting the Prices Table */

/* Delisted Returns */

CREATE TABLE IF NOT EXISTS "Prices"(
"permno" INTEGER, "date" TEXT, "ticker" TEXT, "cusip" TEXT, "dlstcd" INTEGER, "dlret" REAL, "prc" REAL,
 "ret" REAL, "shrout" INTEGER, "spread" REAL);

INSERT INTO Prices SELECT * FROM Prices_old;

ALTER TABLE Prices ADD COLUMN retadj REAL;

UPDATE Prices
SET retadj = CASE
WHEN dlstcd = "" AND RET != "" THEN RET
WHEN dlstcd != "" AND RET = "" AND NOT dlret GLOB '[A-Za-z]' THEN DLRET 
WHEN dlstcd != "" AND RET != "" AND NOT dlret GLOB '[A-Za-z]' THEN (1+DLRET)*(1+RET)-1
WHEN dlstcd != "" AND dlret GLOB '[A-Za-z]' AND dlstcd >= 500 AND dlstcd < 600 THEN -1.0
WHEN dlstcd != "" AND dlret GLOB '[A-Za-z]' AND ret != "" THEN RET
WHEN dlstcd != "" AND dlret GLOB '[A-Za-z]' AND ret = "" THEN 0.0
ELSE ""
END;

/* 
In words: 
If there is no delisting code and return is not missing then RET 
Else if there is delisting code:
    1) if return is missing but delisting return is present then DLRET (usual case)
    2) if return is present and delisting return is present then DLRET (end of month delisting case case)
    3) if delisting return is missing and delisting due to performance (500 codes) then -1 (conservative estimate, alternative -0.3)
    4) if delisting return is missing, delisting not due to performance, and return is present then RET.
    5) if delisting return is missing, delisting not due to performance, and return is missing then 0.0.
else ""
*/

/* Market flag */

ALTER TABLE Prices ADD COLUMN mktflag INTEGER;

UPDATE Prices
SET mktflag = (ABS(Prices.PRC)*Prices.SHROUT > 0.0001*Market.totval)
FROM Market WHERE Prices.date = Market.date;

/* Possible: missing returns which are not from delistings, e.g. RET is missing or it is some code "B", "C". */
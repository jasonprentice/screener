import psycopg2

columns = ['NetCashFlowsContinuing', 'NetCashFlowsFinancingDiscontinued', 'NetIncomeAvailableToCommonStockholdersBasic', 'NonoperatingIncomeLossPlusInterestAndDebtExpense', 'NoncurrentLiabilities', 'IncomeFromContinuingOperationsBeforeTax', 'Equity', 'NetCashFlowsOperatingContinuing', 'OtherComprehensiveIncome', 'ComprehensiveIncomeAttributableToNoncontrollingInterest', 'NetCashFlowsInvestingContinuing', 'ROS','NetIncomeAttributableToParent','SGR','NetCashFlowsInvestingDiscontinued','ROE','PreferredStockDividendsAndOtherAdjustments','NonoperatingIncomePlusInterestAndDebtExpensePlusIncomeFromEquityMethodInvestments','NetCashFlowsOperating','CostsAndExpenses','CurrentAssets','IncomeFromEquityMethodInvestments','NoncurrentAssets','IncomeTaxExpenseBenefit','CostOfRevenue','ExchangeGainsLosses','CurrentLiabilities','Assets','NetCashFlowsDiscontinued','LiabilitiesAndEquity','OperatingIncomeLoss','TemporaryEquity','NonoperatingIncomeLoss','OtherOperatingIncome','EquityAttributableToParent','GrossProfit','NetCashFlow','IncomeFromDiscontinuedOperations','NetCashFlowsInvesting','ComprehensiveIncome','Revenues','CommitmentsAndContingencies','OperatingExpenses','Liabilities','NetCashFlowsFinancingContinuing','EquityAttributableToNoncontrollingInterest','ComprehensiveIncomeAttributableToParent','NetIncomeLoss','IncomeBeforeEquityMethodInvestments','NetCashFlowsOperatingDiscontinued','NetCashFlowsFinancing','ROA','ExtraordaryItemsGainLoss','IncomeFromContinuingOperationsAfterTax','NetIncomeAttributableToNoncontrollingInterest','InterestAndDebtExpense']

conn = psycopg2.connect("dbname=SECData user=postgres password=pwd")
cur = conn.cursor()


sqlstring = "CREATE TABLE IF NOT EXISTS tmp (cik varchar, year int, start_price numeric, end_price numeric, return numeric,"
for item in columns:
    sqlstring = sqlstring + " " + item + " numeric,"
sqlstring = sqlstring + " marketcap numeric, entitycommonstocksharesoutstanding bigint, PRIMARY KEY (cik, year));"

cur.execute(sqlstring)
cur.execute("INSERT INTO tmp SELECT DISTINCT * FROM financials;")
conn.commit()
cur.close()
conn.close()
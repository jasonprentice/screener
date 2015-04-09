
import psycopg2
import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer


class notesCountReader:
  def __init__(self):
    self.conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")    
    self.featureData = pd.DataFrame(columns=['num_words','num_notes'])

  def __del__(self):
    self.conn.close()

  def load_data(self, rel_to, datestr, exchanges=['NASDAQ','N','A','OTC']):
    direction = {'before': '<', 'after': '>'}
    xchng_string = 'exchange=' + ' OR exchange='.join(exchanges)    
    time_string = "to_date( month || ' ' || year, 'MM YYYY') %s to_date( %s, 'MM YYYY')" % (direction[rel_to], datestr)      
    where_string = "(%s) AND (%s) AND companies.cik IS NOT NULL;" % (xchng_string, time_string)
    cols = ['notes.cik', 'to_date(month || \' \' || year, \'MM YYYY\') AS date', 'note_wordcount']

    sqlstring = "SELECT %s FROM notes INNER JOIN companies ON (notes.cik=companies.cik) WHERE %s" % (', '.join(cols), where_string)

    
    individual_notes = pd.read_sql(sqlstring, self.conn, index_col = ['date','cik'])    

    def aggregator(df):
      return pd.Series([df.sum(), df.shape[0]], index=['num_words','num_notes'])
    return individual_notes.groupby(level=['date','cik']).apply(aggregator)    

  def train(self, params):    
    self.featureData = self.load_data(params['years'],params['exchanges'])
    self.index = self.featureData.index

  def test(self, params):
    self.featureData = self.load_data(params['years'],params['exchanges'])
    self.index = self.featureData.index
    


class notesTextReader:
  def __init__(self):
    self.conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")
    self.train_vectorizer = HashingVectorizer(stop_words='english')
                                     
    self.featureData = pd.DataFrame()

  def __del__(self):
    self.conn.close()

  def load_data(self, index):
   
    base_dir = '../data/edgar'       
    def gen_notes():
      for tup in index:
        yr = tup[0]
        mo = tup[1]
        cik = tup[2]
        fname = base_dir + '/' + cik + '/' + yr + '/' + mo + '/notes_text.txt'
        if os.path.isfile(fname):
          f = open(fname)
          yield f.read()
          f.close()      
        else:
          yield ''
    
    return gen_notes

  def train(self, index):
    gen_notes = self.load_data(index)    
    
    self.featureData = self.train_vectorizer.fit_transform(gen_notes())
        
    tfidf = TfidfTransformer().fit(self.featureData);
    self.featureData = tfidf.transform(self.featureData, copy=False)
    
    self.index = index

  def test(self, index):
    gen_notes = self.load_data(index)
    #self.test_vectorizer = TfidfVectorizer(stop_words='english', min_df=3, vocabulary=self.train_vectorizer.vocabulary_)
    self.featureData = HashingVectorizer(stop_words='english').fit_transform(gen_notes())
    tfidf = TfidfTransformer().fit(self.featureData);
    self.featureData = tfidf.transform(self.featureData, copy=False)
    self.index = index


class financialsReader:

  def __init__(self):        
    self.features = {}
    self.featureData = pd.DataFrame()
    
    self.conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")
    self.init_features()
            
  def __del__(self):
    self.conn.close()

  def init_features(self):
    # Define features that will be extracted from DB and fed to classifier 

    columns = [# Cash flow statement items
               'NetCashProvidedByUsedInOperatingActivities',
               'NetCashProvidedByUsedInFinancingActivities',
               'NetCashProvidedByUsedInInvestingActivities',                  
               'EffectOfExchangeRateOnCashAndCashEquivalents',
               'CashAndCashEquivalentsPeriodIncreaseDecrease',
                # Income statement items
               'Revenues',
               'CostOfRevenue', 
               'GrossProfit',
               'OperatingExpenses',
               'OtherOperatingIncome',
               'OperatingIncomeLoss',
               'NonoperatingIncomeExpense',
               'InterestAndDebtExpense',
               'IncomeLossFromEquityMethodInvestments',
               'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
               'IncomeTaxExpenseBenefit',
               'IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest',
               'IncomeLossFromDiscontinuedOperationsNetOfTax',
               'ExtraordinaryItemNetOfTax',
               'ProfitLoss',
               'NetIncomeLossAttributableToNoncontrollingInterest',
               'NetIncomeLoss',
               'PreferredStockDividendsAndOtherAdjustments',
               'NetIncomeLossAvailableToCommonStockholdersBasic'
               'CostsAndExpenses',                                              
                # Balance sheet items
               'AssetsCurrent',
               'CashCashEquivalentsAndShortTermInvestments',
               'ReceivablesNetCurrent',
               'InventoryNet',
               'PrepaidExpenseAndOtherAssetsCurrent',
               'DeferredCostsCurrent',
               'DerivativeInstrumentsAndHedges',
               'AssetsNoncurrent',
               'PropertyPlantAndEquipmentNet',
               'Assets',
               'LiabilitiesCurrent',
               'AccountsPayableAndAccruedLiabilitiesCurrent',
               'DebtCurrent',
               'DerivativeInstrumentsAndHedgesLiabilities',
               'LiabilitiesNoncurrent',
               'LongTermDebtAndCapitalLeaseObligations',
               'LiabilitiesOtherThanLongtermDebtNoncurrent',
               'Liabilities',            
               'CommitmentsAndContingencies',
               'Equity',
               #'LiabilitiesAndStockholdersEquity',
               'TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests',
               'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
               #Market cap
               'MarketCap']
  
    colname_maxlen = 63
    for item in columns:
      self.features[item] = item.lower()
      if len(item) > colname_maxlen:
        self.features[item] = self.features[item][:colname_maxlen]
                    
    # Derived features
    self.features['EV'] = '(marketcap+longtermdebtandcapitalleaseobligations)'
    # Ratios
    self.features['BP'] = '(equity / marketcap)'
    self.features['EP'] = '(netincomeloss / marketcap)'
    self.features['ROA'] = '(netincomeloss / NULLIF(assets,0))'
    
    self.features['ROE'] = '(netincomeloss / NULLIF(equity,0))'
    self.features['EarningsYield'] = '((profitloss - incometaxexpensebenefit) / NULLIF(marketcap+longtermdebtandcapitalleaseobligations,0))'
    self.features['ROC'] = '((profitloss - incometaxexpensebenefit) / NULLIF(assets,0))'
    self.features['DebtToEquity'] = '(longtermdebtandcapitalleaseobligations / NULLIF(equity,0))'
    
    self.features['CurrentRatio'] = '(assetscurrent / NULLIF(liabilitiescurrent,0))'
    self.features['Overhead'] = '(operatingexpenses / NULLIF(grossprofit,0))'
    self.features['CurAssetsRatio'] = '(assetscurrent / NULLIF(assetscurrent+assetsnoncurrent,0))'
    self.features['CurLiabilitiesRatio'] = '(liabilitiescurrent / NULLIF(liabilitiescurrent+liabilitiesnoncurrent,0))'
    self.features['CFRatioInvesting'] = '(netcashprovidedbyusedininvestingactivities / NULLIF(cashandcashequivalentsperiodincreasedecrease,0))'
    self.features['CFRatioOperating'] = '(netcashprovidedbyusedinoperatingactivities / NULLIF(cashandcashequivalentsperiodincreasedecrease,0))'
    self.features['CFRatioFinancing'] = '(netcashprovidedbyusedinfinancingactivities / NULLIF(cashandcashequivalentsperiodincreasedecrease,0))'

  def load_data(self, years, exchanges=['NASDAQ','N','A','OTC']):
    # Read data, limiting to specified years and exchanges
    MAX_mktcap = 1e12

    xchng_string = ""
    for exchange in exchanges:
      xchng_string = xchng_string + " OR exchange='" + exchange + "'"

    yr_string = ""
    for yr in years:
        yr_string = yr_string + " OR year='" + str(yr)+"'"
      
    where_string = "(" + xchng_string[4:] + ") AND (" + yr_string[4:] + ") AND companies.cik IS NOT NULL AND one_year_return <> 'NaN' AND assets > 0 AND marketcap > 0 AND marketcap < " + str(MAX_mktcap) + ";"
    
    sqlstring = "SELECT financials.cik, year, month, "
    for (i,name) in enumerate(self.features.keys()):
        sqlstring = sqlstring + self.features[name] + " AS " + name + ", "
    
    sqlstring = sqlstring[:-2] + " FROM financials INNER JOIN companies ON (financials.cik=companies.cik) WHERE " + where_string
    
    featureData = pd.read_sql(sqlstring, self.conn, index_col = ['year','month','cik'])
    featureData = featureData.fillna(0)
    
    self.company_info = pd.read_sql("SELECT cik, name, ticker FROM companies;", self.conn, index_col = 'cik')

    cik_idx = set()
    for yr in years:
        ix = featureData.loc[yr].index
        for cik in ix:
            cik_idx.add(cik[1])
    self.company_info = self.company_info.loc[cik_idx]
    
    def strip_coname(name):
        ix = name.lower().find("common stock")
        name = name[:ix].rstrip()
        if name[-1] == '-':
            name = name[:-2].rstrip()
        return name
        
    self.company_info.name = self.company_info.name.map(strip_coname)
    return featureData
           

  def train(self, params):    
    self.featureData = self.load_data(params['years'],params['exchanges'])
    self.index = self.featureData.index
    
  def test(self, params):
    self.featureData = self.load_data(params['years'],params['exchanges'])
    self.index = self.featureData.index
    

class returnsReader:
  def __init__(self, return_dur='annual'):
    if return_dur=='annual':
      self.return_field = 'one_year_return'
    elif return_dur=='quarterly':
      self.return_field = 'three_month_return'

    self.conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")    

  def __del__(self):
    self.conn.close()

  def load_data(self, years, exchanges=['NASDAQ','N','A','OTC']):
    # Read data, limiting to specified years and exchanges
    
    xchng_string = ""
    for exchange in exchanges:
      xchng_string = xchng_string + " OR exchange='" + exchange + "'"

    yr_string = ""
    for yr in years:
        yr_string = yr_string + " OR year='" + str(yr)+"'"
      
    where_string = "(" + xchng_string[4:] + ") AND (" + yr_string[4:] + ") AND companies.cik IS NOT NULL AND " + self.return_field + " <> 'NaN';"
            
    return pd.read_sql("SELECT financials.cik, year, month, " + self.return_field + " FROM financials INNER JOIN companies ON (financials.cik=companies.cik) WHERE " + where_string, self.conn, index_col = ['year','month','cik'])
    
  def train(self, params):    
    self.featureData = self.load_data(params['years'],params['exchanges'])
    self.index = self.featureData.index
    
  def test(self, params):
    self.featureData = self.load_data(params['years'],params['exchanges'])
    self.index = self.featureData.index
    

  def apply_threshold(self, threshold, criterion):
  # Apply a threshold to convert returns in y to binary variable. 

    output = pd.Series(index=self.index)
    rets = self.featureData[self.return_field]
    if criterion == 'gt':     
      # Above threshold       
        output.loc[rets > threshold] = 1
        output.loc[rets <= threshold] = 0                      
    elif criterion == 'lt':   
      # Below threshold                   
        output.loc[rets <= threshold] = 1
        output.loc[rets > threshold] = 0                            
    elif criterion == 'in':       
      # Absolute value within threshold   
        output.loc[rets < abs(threshold) and rets > -abs(threshold)] = 1
        output.loc[rets >= abs(threshold) or rets <= -abs(threshold)] = 0          
    elif criterion == 'out':          
      # Absolute value outside threshold  
        output.loc[rets < abs(threshold) and rets > -abs(threshold)] = 0
        output.loc[rets >= abs(threshold) or rets <= -abs(threshold)] = 1          
    elif criterion == 'topquant':  
      # Top quantile within time period
        def thresh_quant(ret):
          q = np.percentile(ret, threshold)                                    
          y_out = ret
          y_out.loc[ret > q] = 1
          y_out.loc[ret <= q] = 0
          return y_out

        output = self.featureData[self.return_field].groupby(level=['year','month']).transform(thresh_quant)            
    elif criterion == 'bottomquant':
      # Bottom quantile within time period
        def thresh_quant(ret):
          q = np.percentile(ret, threshold)                                    
          y_out = ret
          y_out.loc[ret > q] = 0
          y_out.loc[ret <= q] = 1
          return y_out

        output = self.featureData[self.return_field].groupby(level=['year','month']).transform(thresh_quant)            
        
    else:
        raise error("Criterion must be one of: 'lt', 'gt', 'in', 'out', 'topquant', 'bottomquant'")
            
    return output



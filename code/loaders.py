
import psycopg2
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def query_db(conn, table, cols, before=None, after=None, constraints=[], exchanges=['NASDAQ','N','A','OTC']):
  direction = {'before': '<', 'after': '>'} 
  xchng_string = '(exchange=' + ' OR exchange='.join(["\'%s\'"%x for x in exchanges]) + ')'
  cons = constraints + [xchng_string]
  if before:
    cons = cons + ["(to_date( month || ' ' || year, 'MM YYYY') < to_date(\'%s\', 'MM YYYY')) " % before]  
  if after:
    cons = cons + ["(to_date( month || ' ' || year, 'MM YYYY') > to_date(\'%s\', 'MM YYYY')) " % after]
  
  cons = cons + ['companies.cik IS NOT NULL']
  
  where_string = ' AND '.join(cons)     
  cols = cols + ['%s.cik'%table, 'to_date(month || \' \' || year, \'MM YYYY\') AS date']
  
  sqlstring = "SELECT %s FROM %s INNER JOIN companies ON (%s.cik=companies.cik) WHERE %s;" % (', '.join(cols), table,table, where_string)  
  
  data = pd.read_sql(sqlstring, conn, index_col = ['date','cik'], parse_dates=['date'])        
  return data

def query_pickle(conn, pickle_file, before=None, after=None, exchanges=['NASDAQ','N','A','OTC']):
  if before:
    t0 = pd.to_datetime(before, '%m %Y')
  if after:
    t1 = pd.to_datetime(after, '%m %Y')

  f = open(pickle_file,'r')
  df = pickle.load(f)
  f.close()

  # Find CIKs matching exchanges
  xchng_string = '(exchange=' + ' OR exchange='.join(["\'%s\'"%x for x in exchanges]) + ')'  
  cons = [xchng_string, 'cik IS NOT NULL']
  where_string = ' AND '.join(cons)       
  sqlstring = "SELECT cik FROM companies WHERE %s;" % (where_string)  
  
  data = pd.read_sql(sqlstring, conn, index_col = ['cik'])

  sub_index = [(date, cik) for (date,cik) in df.index if (date > after and date < before and cik in data.index)]
  return df.reindex(sub_index)


class notesCountReader:
  def __init__(self, source='db'):
    self.source = source
    self.pickle_file = '../data/pickles/notes_count.pickle'
    self.conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")    
    self.featureData = pd.DataFrame(columns=['num_words','num_notes'])    

  def __del__(self):
    self.conn.close()

  def load_data(self, before=None, after=None, exchanges=['NASDAQ','N','A','OTC']):            
    if self.source == 'db':
      individual_notes = query_db(self.conn, 'notes', ['note_wordcount'], before=before, after=after, exchanges=exchanges)
    elif self.source == 'pickle':
      individual_notes = query_pickle(self.conn, self.pickle_file, before=before, after=after, exchanges=exchanges)
    
    def aggregator(df):
      return pd.Series([df.sum(), df.shape[0]], index=['num_words','num_notes'])
    return individual_notes.groupby(level=['date','cik']).apply(aggregator)    

  def train(self, params):    
    self.featureData = self.load_data(before = params['before'], exchanges= params['exchanges'])
    self.index = self.featureData.index
    
  def test(self, params, most_recent=False):
    self.featureData = self.load_data(before = params['before'], after=params['after'], exchanges=params['exchanges'])
    if most_recent:
      def max_date(g):
        g = g.reset_index()
        g = g.iloc[g['date'].argmax()]
        g = g.drop('cik')
        return g

      self.featureData = self.featureData.groupby(level='cik').apply(max_date).reset_index().set_index(['date','cik'])
    
    self.index = self.featureData.index
    

class notesTextReader:
  def __init__(self, source='db'):
    self.source = source
    self.pickle_file = '../data/pickles/notes_text.pickle'
    self.conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")
    self.train_vectorizer = CountVectorizer(stop_words='english', min_df=3, ngram_range=(3,3))
                                     
    self.featureData = pd.DataFrame()

  def __del__(self):
    self.conn.close()

  def load_data(self, index, vocabulary=None):
    
    base_dir = '../data/edgar'       
    def gen_notes():
      for tup in index:
        yr = '%4d' % tup[0].year        
        mo = '%02d' % tup[0].month        
        cik = tup[1]
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
        
    #tfidf = TfidfTransformer().fit(self.featureData);
    #self.featureData = tfidf.transform(self.featureData, copy=False)
    
    self.index = index

  def test(self, index):
    gen_notes = self.load_data(index)
    #self.test_vectorizer = TfidfVectorizer(stop_words='english', min_df=3, vocabulary=self.train_vectorizer.vocabulary_)
    self.featureData = TfidfVectorizer(stop_words='english', vocabulary=self.train_vectorizer.vocabulary_).fit_transform(gen_notes())
    # tfidf = TfidfTransformer().fit(self.featureData);
    # self.featureData = tfidf.transform(self.featureData, copy=False)
        
    self.index = index


class financialsReader:

  def __init__(self, source='db'):        
    self.source = source
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

  def load_data(self, before=None, after=None, exchanges=['NASDAQ','N','A','OTC']):
    # Read data, limiting to specified years and exchanges
    MAX_mktcap = 1e12           
    constraints = ['one_year_return <> \'NaN\'', 'assets>0', 'marketcap>0', 'marketcap<%s'%str(MAX_mktcap)]
    cols = ['%s AS %s' % (v,k) for (k,v) in self.features.iteritems()]
    featureData = query_db(self.conn, 'financials', cols, before=before, after=after, constraints=constraints, exchanges=exchanges)
    featureData = featureData.fillna(0)
    
    self.company_info = pd.read_sql("SELECT cik, name, ticker FROM companies;", self.conn, index_col = 'cik')
    self.company_info = self.company_info.groupby(level='cik').apply(lambda df: df.iloc[0])
    self.company_info = self.company_info.reindex(featureData.reset_index('cik').cik.unique())
  
    def strip_coname(name):
        ix = name.lower().find("common stock")
        name = name[:ix].rstrip()
        if name[-1] == '-':
            name = name[:-2].rstrip()
        return name
        
    self.company_info.name = self.company_info.name.map(strip_coname)
    return featureData
           

  def train(self, params):    
    self.featureData = self.load_data(before=params['before'], exchanges=params['exchanges'])
    self.index = self.featureData.index
    
  def test(self, params, most_recent=False):
    self.featureData = self.load_data(after=params['after'], before=params['before'], exchanges=params['exchanges'])    
    if most_recent:
      def max_date(g):
        g = g.reset_index()
        g = g.iloc[g['date'].argmax()]
        g = g.drop('cik')
        return g

      self.featureData = self.featureData.groupby(level='cik').apply(max_date).reset_index().set_index(['date','cik'])

    self.index = self.featureData.index
    



class returnsReader:
  def __init__(self, return_dur='annual', source='db'):
    self.source = source

    if return_dur=='annual':
      self.return_field = 'one_year_return'
    elif return_dur=='quarterly':
      self.return_field = 'three_month_return'

    self.conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")    

  def __del__(self):
    self.conn.close()

  def load_data(self, before=None, after=None, exchanges=['NASDAQ','N','A','OTC']):
    # Read data, limiting to specified years and exchanges              
    if self.source == 'db':
      constraints = ['%s <> \'NaN\''%self.return_field]    
      cols = [self.return_field]
      return query_db(self.conn, 'financials', cols, before=before, after=after, constraints=constraints, exchanges=exchanges)
    elif self.source == 'pickle':
      return query_pickle(self.conn, self.pickle_file, before=before, after=after, constraints=constraints, exchanges=exchanges)

  def train(self, params):    
    self.featureData = self.load_data(before=params['before'], exchanges=params['exchanges'])
    self.index = self.featureData.index
    
  def test(self, params, most_recent=False):
    self.featureData = self.load_data(after=params['after'], before=params['before'], exchanges= params['exchanges'])
    if most_recent:
      def max_date(g):
        g = g.reset_index()
        g = g.iloc[g['date'].argmax()]
        g = g.drop('cik')
        return g

      self.featureData = self.featureData.groupby(level='cik').apply(max_date).reset_index().set_index(['date','cik'])
    
    self.index = self.featureData.index
        

  def apply_threshold(self, threshold, criterion, field=None):    
  # Apply a threshold to convert returns in y to binary variable. 
    if field==None:
      field = self.return_field

    output = pd.Series(index=self.index)
    rets = self.featureData[field]
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

        output = self.featureData[field].groupby(level=['date']).transform(thresh_quant)            
    elif criterion == 'bottomquant':
      # Bottom quantile within time period
        def thresh_quant(ret):
          q = np.percentile(ret, threshold)                                    
          y_out = ret
          y_out.loc[ret > q] = 0
          y_out.loc[ret <= q] = 1
          return y_out

        output = self.featureData[field].groupby(level=['date']).transform(thresh_quant)            
        
    else:
        raise error("Criterion must be one of: 'lt', 'gt', 'in', 'out', 'topquant', 'bottomquant'")
            
    return output



class dailyReturnsReader (returnsReader):
  
  def time_increment(self, time, delta):
    # delta = time difference in months
    new_time = time + np.timedelta64(delta,'M')
    return pd.to_datetime('%02d %04d' % (new_time.month,new_time.year), '%m %Y')
  def time_decrement(self, time, delta):
    # delta = time difference in months
    new_time = time - np.timedelta64(delta,'M')
    return pd.to_datetime('%02d %04d' % (new_time.month,new_time.year), '%m %Y')

  def load_data(self, before=None, after=None, exchanges=['NASDAQ','N','A','OTC']):

    def next_trading_price(date, pr): 
      try:
        return pr.loc[date]
      except:
        return None      

    # Read data, limiting to specified years and exchanges              
    constraints = ['%s <> \'NaN\''%self.return_field]        
    tmp = query_db(self.conn, 'financials', [], before=before, after=after, constraints=constraints, exchanges=exchanges)
    df = pd.DataFrame(columns=['previous_return',self.return_field], index=tmp.index)
    f = open('../data/pickles/prices_all_stocks.pickle','r')
    all_pr = pickle.load(f)
    f.close()
    for (date,cik) in df.index:
      base_dir = '../data/edgar/'
      cik_exists = False
      try:
        f = open(base_dir + cik + '/all_daily_prices.pickle','r')
        pr = pickle.load(f)        
        f.close() 
        cik_exists = True
      except:
        print 'No price information for %s' % cik

      if cik_exists:
        
        # pr = pr.iloc[:-2].astype('float')        
        # pr = pr.fillna(method='backfill',limit=3)
        start_pr = next_trading_price(self.time_increment(date,1), pr)
        if self.return_field == 'one_year_return':
          end_pr = next_trading_price(self.time_increment(date,13), pr)        
        else:
          end_pr = next_trading_price(self.time_increment(date,4), pr)        
        

        if (end_pr == None) or (start_pr == None):
          df.loc[(date,cik),self.return_field] = float('NaN')
        else:
          df.loc[(date,cik), self.return_field] = (end_pr - start_pr) / start_pr


        start_pr = next_trading_price(self.time_decrement(date,3),pr)
        end_pr = next_trading_price(date,pr)
        if (end_pr == None) or (start_pr == None):
          df.loc[(date,cik),'previous_return'] = float('NaN')
        else:
          df.loc[(date,cik),'previous_return'] = 0.25 * (end_pr - start_pr) / start_pr
      else:
        df.loc[(date,cik),self.return_field] = float('NaN')
        df.loc[(date,cik),'previous_return'] = float('NaN')

    
    df = df[pd.notnull(df[self.return_field])]#.set_index(['date','cik'])    
    return df

 

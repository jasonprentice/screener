import psycopg2
import numpy as np

from scipy.stats import rankdata

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.externals.six import StringIO
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from multiprocessing import Process

"""
Example use:

 from returnClassifier import returnPredictor
 pred = returnPredictor()
 
 # Train model to find stocks with positive return
 pred.load_data([2010,2011,2012])
 pred.train(0,'gt')
 
 # Evaluate performance on withheld year's data
 pred.load_data([2013])
 pred.evaluate(0,'gt',plot=True)

"""

class returnPredictor:

    def __init__(self, pre_pca = False):
        
        self.pre_pca = pre_pca          # Boolean, whether or not to apply PCA dimensionality reduction to features
        self.features = {}              # Dictionary mapping feature names to defining SQL formula
        self.featureData = []           # Array of feature data for all companies loaded from database
        self.column_index = {}          # Dictionary mapping feature names to columns of featureData
        self.returns = []               # Stock return for each company
        self.mktcaps = []               # Market capitalization (approximate) of each company
        self.ciks = []                  # EDGAR identifiers (CIKs)
        self.tickers = []               # Ticker symbol
        self.co_names = []              # Company name
        self.princomp = []
        self.p_optim = 0
        
        self.init_features()
        # Database object
        self.conn = psycopg2.connect("dbname=SECData user=postgres password=pwd")
        # Initialize classifier
        self.clf = RandomForestClassifier(n_estimators=30,criterion='gini',max_features='auto',max_depth=5)


    def __del__(self):
        self.conn.close()

    def init_features(self):
        # Define financial data and ratios to use in classifier
        
        columns = [# Cash flow statement items
                   'NetCashFlow',
                   'NetCashFlowsContinuing',
                   'NetCashFlowsDiscontinued',
                   'NetCashFlowsOperating',
                   'NetCashFlowsOperatingContinuing',
                   'NetCashFlowsOperatingDiscontinued',
                   'NetCashFlowsFinancing',
                   'NetCashFlowsFinancingContinuing',
                   'NetCashFlowsFinancingDiscontinued',
                   'NetCashFlowsInvesting',
                   'NetCashFlowsInvestingContinuing',
                   'NetCashFlowsInvestingDiscontinued',
                   # Income statement items
                   'CostOfRevenue',
                   'CostsAndExpenses',
                   'GrossProfit',
                   'OperatingExpenses',
                   'OtherOperatingIncome',
                   'IncomeFromContinuingOperationsBeforeTax',
                   'IncomeFromContinuingOperationsAfterTax',
                   'OperatingIncomeLoss',
                   'IncomeFromDiscontinuedOperations',
                   'ExchangeGainsLosses',
                   'OtherComprehensiveIncome',
                   'InterestAndDebtExpense',
                   'PreferredStockDividendsAndOtherAdjustments',
                   'IncomeTaxExpenseBenefit',
                   'ExtraordaryItemsGainLoss',
                   'NonoperatingIncomeLossPlusInterestAndDebtExpense',
                   'IncomeBeforeEquityMethodInvestments',
                   'IncomeFromEquityMethodInvestments',
                   'NonoperatingIncomePlusInterestAndDebtExpensePlusIncomeFromEquityMethodInvestments',
                   'NonoperatingIncomeLoss',
                   'ComprehensiveIncome',
                   'ComprehensiveIncomeAttributableToParent',
                   'NetIncomeAvailableToCommonStockholdersBasic',
                   'ComprehensiveIncomeAttributableToNoncontrollingInterest',
                   'NetIncomeAttributableToParent',
                   'NetIncomeAttributableToNoncontrollingInterest',
                   'NetIncomeLoss',
                   # Balance sheet items
                   'CurrentLiabilities',
                   'NoncurrentLiabilities',
                   'CommitmentsAndContingencies',
                   'Liabilities',
                   'CurrentAssets',
                   'NoncurrentAssets',
                   'Assets',
                   'TemporaryEquity',
                   'EquityAttributableToParent',
                   'EquityAttributableToNoncontrollingInterest',
                   'Equity',
                   #'LiabilitiesAndEquity',
                   # Price items
                    'MarketCap']
        for item in columns:
            self.features[item] = item.lower()

        # Derived features
        self.features['EV'] = '(marketcap+noncurrentliabilities)'
        # Ratios
        self.features['B/P'] = '(equityattributabletoparent / marketcap)'
        self.features['E/P'] = '(netincomeloss / marketcap)'
        self.features['ROA'] = '(netincomeloss / NULLIF(assets,0))'
        self.features['ROE'] = '(netincomeloss / NULLIF(equity,0))'
        self.features['EarningsYield'] = '((IncomeFromContinuingOperationsBeforeTax + InterestAndDebtExpense) / NULLIF(marketcap+noncurrentliabilities,0))'
        self.features['ROC'] = '((IncomeFromContinuingOperationsBeforeTax + InterestAndDebtExpense) / NULLIF(assets,0))'
        self.features['Debt/Equity'] = '(noncurrentliabilities / NULLIF(equity,0))'
        self.features['CurrentRatio'] = '(currentassets / NULLIF(currentliabilities,0))'
        self.features['Overhead'] = '(operatingexpenses / NULLIF(grossprofit,0))'
        self.features['CurAssetsRatio'] = '(currentassets / NULLIF(currentassets+noncurrentassets,0))'
        self.features['CurLiabilitiesRatio'] = '(currentliabilities / NULLIF(currentliabilities+noncurrentliabilities,0))'
        self.features['CFRatioInvesting'] = '(netcashflowsinvesting / NULLIF(netcashflow,0))'
        self.features['CFRatioOperating'] = '(netcashflowsoperating / NULLIF(netcashflow,0))'
        self.features['CFRatioFinancing'] = '(netcashflowsfinancing / NULLIF(netcashflow,0))'
    
    
    def load_data(self, years):
        # Import all data from postgreSQL database, for specified years
        
        MAX_mktcap = 1e12
        yr_string = ""
        for yr in years:
            yr_string = yr_string + " OR year=" + str(yr)
        where_string = "(" + yr_string[4:] + ") AND marketcap > 0 AND marketcap < " + str(MAX_mktcap) + ";"
        
        sqlstring = "SELECT cik, return, "
        for (i,name) in enumerate(self.features.keys()):
            sqlstring = sqlstring + self.features[name] + ", "
            self.column_index[name] = i
        
        sqlstring = sqlstring[:-2] + " FROM financials WHERE " + where_string

        cur = self.conn.cursor()
        cur.execute(sqlstring)

        tmp = []
        self.ciks = []
        self.returns = []
        self.mktcaps = []
        for row in cur:
            self.ciks.append(row[0])
            self.returns.append(row[1])
            features = list(row[2:])
            self.mktcaps.append(features[self.column_index['MarketCap']])
            for (i,f) in enumerate(features):
                if f==None:
                    features[i]=0.0
            tmp.append(features)


        self.featureData = np.array(tmp,dtype='float_')
        
        self.returns = np.array(self.returns,dtype='float_')
        self.mktcaps = np.array(self.mktcaps,dtype='float_')

        self.tickers = []
        self.co_names = []
        for cik in self.ciks:
            cur.execute("SELECT ticker FROM companies WHERE cik=%s;",(cik,))
            item = cur.fetchone()
            self.tickers.append(item[0])

            cur.execute("SELECT name FROM companies WHERE cik=%s;",(cik,))
            item = cur.fetchone()
            self.co_names.append(item[0])


        cur.close()



    def train(self, threshold,dir):
        # Train classifier
        
        if self.pre_pca:
            ranks = self.convert_features_to_ranks(self.featureData)
            (train_data,self.princomp) = self.pca_whiten_features(ranks, 0.95)
        else:
            train_data = self.featureData
        
        X_train, X_test, ret_train, ret_test = train_test_split(train_data, self.returns, test_size=0.)
        y_test = self.apply_threshold(ret_test, threshold,dir)
        y_train = self.apply_threshold(ret_train, threshold,dir)
        
        (N,nfeatures) = X_train.shape
        print "Training random forest: " + str(N) + " training examples, " + str(nfeatures) + " features."#, %(f).2f" % {'f':np.mean(y_train)}
        self.clf = self.clf.fit(X_train, y_train)

    def evaluate(self, threshold,dir, plot=True):
        # Load 2014 returns from database
        cur = self.conn.cursor()
            YTDreturns = np.zeros((len(self.ciks),1),dtype='float_')
            for (i,cik) in enumerate(self.ciks):
                cur.execute("SELECT ytdreturn FROM companies WHERE cik=%s;",(cik,))
                item = cur.fetchone()
                YTDreturns[i] = 100*item[0]
        cur.close()
            output = self.apply_threshold(self.returns,threshold,dir)
            
            # Get classifier scores for portfolio weighting
            p = self.apply_model()
            w = p[:,1]
            
# REMAINDER CUT OFF DUE TO CHARACTER LIMIT
            # Get return of weighted portfolios
            thresh = np.linspace(0,1,100)
            ret = np.zeros(thresh.shape)
            n = np.zeros(thresh.shape)
            
            for (i,t) in enumerate(thresh):
                w_above = w[w>t]
                n[i] = len(w_above)
                if len(w_above) > 0:
                    ret[i] = np.average(YTDreturns[w>t], axis=0, weights=w[w>t])
                else:
                    ret[i] = None

    if self.pre_pca:
        ranks = self.convert_features_to_ranks(self.featureData)
            X = self.princomp.transform(ranks)
            else:
                X = self.featureData

        auc = self.ROC(X, output, plot=plot, title='Validation set')
            
            if plot:
                # Sort to find top-scoring companies
                tmp = zip(self.tickers, w, self.returns,self.co_names)
                tmp = sorted(tmp, key = lambda item: item[1], reverse=True)
                
                print "Ticker | Score | YTD return (%) | Company name "
                print "-----------------------------------------------"
                for item in tmp[:10]:
                    print "%(ticker)5s  | %(p)4.2f  |     %(ret)6.2f     | %(name)s" % {'ticker':item[0],'p':item[1],'ret':item[2],'name':item[3]}
                
                self.plot_returns(thresh,ret)


    def apply_threshold(self, y, threshold,dir):
        # Define binary variable targeted by classifier
        
        output = np.zeros(len(y))
        for i,x in enumerate(y):
            if dir == 'gt':
                if x>threshold:
                    output[i] = 1
            elif dir == 'lt':
                if x<threshold:
                    output[i] = 1
            elif dir == 'in':
                if abs(x) < abs(threshold):
                    output[i] = 1
            elif dir == 'out':
                if abs(x) > abs(threshold):
                    output[i] = 1
            else:
                raise error("Direction must be 'lt', 'gt', 'in', or 'out'")
        return output


    def convert_features_to_ranks(self, X):
        (n,m) = X.shape
        for i in range(m):
            X[:,i] = rankdata(X[:,i], method='average')
        return X

    def pca_whiten_features(self, X, compression):
        # Reduce to PCA dimensions capturing fraction of variance given by compression
        
        if compression <= 0 or compression >= 1:
            error("Compression must be in (0,1)")
        pca = PCA(n_components = compression, whiten=True)
        X_reduced = pca.fit_transform(X)
        
        return (X_reduced, pca)

    def apply_model(self):
        # Get classifier score on all loaded data
        if self.pre_pca:
            ranks = self.convert_features_to_ranks(self.featureData)
            X = self.princomp.transform(ranks)
        else:
            X = self.featureData

        p = self.clf.predict_proba(X)
        return p




    def plot_returns(self,thresh,ret):
        plt.figure(figsize=[5,5])
        plt.plot(thresh, ret)
        plt.title('Performance of score-weighted portfolio')
        plt.xlabel('Minimum allowed score')
        plt.ylabel('Portfolio annual return (%)')
        plt.show()
    
    
    def ROC(self, X,y, plot=True, title=''):
        y_score = self.clf.predict_proba(X)

        y_score = y_score[:, 1]
        fpr,tpr,thresh = roc_curve(y,y_score)
        
        roc_auc = auc(fpr,tpr)

        if plot:
             self.show_ROC(fpr,tpr,thresh,roc_auc, title)

        return roc_auc

       
    def show_ROC(self, fpr,tpr,thresh,roc_auc, title):
        plt.ion()
        plt.figure(figsize=[5,5])
        plt.plot(fpr,tpr)
        plt.title(title + ' ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.show()



import psycopg2
import numpy as np
import pandas as pd
import os

from scipy.stats import rankdata

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sbn


class returnPredictor:

    def __init__(self, pre_pca = False):
        # pre_pca: Turns on preprocessing that reduces feature space dimensionality by keeping top few principal components.
        # This adversely effects interprateability!

        self.pre_pca = pre_pca
        self.features = {}
        self.featureData = []
        self.column_index = {}
        self.returns = []
        self.mktcaps = []
        self.ciks = []
        self.tickers = []
        self.co_names = []
        self.princomp = []
        self.p_optim = 0
                
        self.conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")
        self.init_features()
        self.clf = RandomForestClassifier(n_estimators=30,criterion='gini',max_features='auto',max_depth=5)
                

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

    def feature_stats(self, feature_name):
      # Summary stats of the feature feature_name
      if feature_name in self.column_index.keys():
        print self.featureData[feature_name].describe()
      else:
        print "No feature %(name)s" % {'name':feature_name}


    def split_on_feature(self, feature_name, threshold):
        if feature_name in self.column_index.keys():
            idx = self.column_index[feature_name]
            feature = self.featureData[:,idx]
            ret_up = self.returns[feature > threshold]
            ret_down = self.returns[feature < threshold]
            n_up = len(ret_up)
            print feature_name + " > %(th).2f:  %(p).2f%%" % {'th':threshold, 'p':100.0 * (float(n_up) / float(len(feature)))}
            if n_up > 0:
                print "Above mean return (std): %(m).2f (%(s).2f)" % {'m':np.mean(ret_up), 's':np.std(ret_up)}
            if n_up < len(feature):
                print "Below mean return (std): %(m).2f (%(s).2f)" % {'m':np.mean(ret_down), 's':np.std(ret_down)}
    
        else:
            print "No feature %(name)s" % {'name':feature_name}

    
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
            self.column_index[name] = i
        
        sqlstring = sqlstring[:-2] + " FROM financials INNER JOIN companies ON (financials.cik=companies.cik) WHERE " + where_string
        
        self.featureData = pd.read_sql(sqlstring, self.conn, index_col = ['year','month','cik'])
        self.featureData = self.featureData.fillna(0)

        self.returns = pd.read_sql("SELECT financials.cik, year, month, one_year_return FROM financials INNER JOIN companies ON (financials.cik=companies.cik) WHERE " + where_string, self.conn, index_col = ['year','month','cik'])

        self.company_info = pd.read_sql("SELECT cik, name, ticker FROM companies;", self.conn, index_col = 'cik')


        cik_idx = set()
        for yr in years:
            ix = self.featureData.loc[yr].index
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
                


    def train(self, threshold, criterion, plot=False):
      # Train classifier: threshold, criterion - see apply_threshold() below
      #                   if plot=True, chart most important features

        if self.pre_pca:
            ranks = self.convert_features_to_ranks(self.featureData)
            (train_data,self.princomp) = self.pca_whiten_features(ranks, 0.95)
        else:
            train_data = self.featureData
               
        y = self.apply_threshold(self.returns, threshold, criterion)
        X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.)
        
        
        (N,nfeatures) = X_train.shape
        print "Training random forest: " + str(N) + " training examples, " + str(nfeatures) + " features."#, %(f).2f" % {'f':np.mean(y_train)}
        self.clf = self.clf.fit(X_train, y_train)

        if plot:
          self.feature_importance()
            #        self.ROC(X_test,y_test, title='Test set')
    
        #self.mean_std(X_test, ret_test)



    def apply_threshold(self, y, threshold, criterion):
      # Apply a threshold to convert returns in y to binary variable. 

        output = pd.Series(index=y.index)
        rets = y['one_year_return']
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

            output = y['one_year_return'].groupby(level=['year','month']).transform(thresh_quant)            
        elif criterion == 'bottomquant':
          # Bottom quantile within time period
            def thresh_quant(ret):
              q = np.percentile(ret, threshold)                                    
              y_out = ret
              y_out.loc[ret > q] = 0
              y_out.loc[ret <= q] = 1
              return y_out

            output = y['one_year_return'].groupby(level=['year','month']).transform(thresh_quant)            
            
        else:
            raise error("Criterion must be one of: 'lt', 'gt', 'in', 'out', 'topquant', 'bottomquant'")
                
        return output


    def convert_features_to_ranks(self, X):
      # Preprocessing for PCA, convert to ranked data for better uniformity
        (n,m) = X.shape
        for i in range(m):
            X[:,i] = rankdata(X[:,i], method='average')
        return X


    def pca_whiten_features(self, X, compression):
        if compression <= 0 or compression >= 1:
            error("Compression must be in (0,1)")
        pca = PCA(n_components = compression, whiten=True)
        X_reduced = pca.fit_transform(X)
        
        return (X_reduced, pca)

    def show_PCA(self, X_reduced,y):
        #        X_reduced = self.princomp.transform(self.featureData)
        plt.figure()
        plt.subplot(2,2,1)
        plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y)
        plt.subplot(2,2,2)
        plt.scatter(X_reduced[:,2],X_reduced[:,3],c=y)
        plt.subplot(2,2,3)
        plt.scatter(X_reduced[:,4],X_reduced[:,5],c=y)
        plt.subplot(2,2,4)
        plt.scatter(X_reduced[:,6],X_reduced[:,7],c=y)
        
        plt.show()


    def apply_model(self):
        if self.pre_pca:
            ranks = self.convert_features_to_ranks(self.featureData.values)
            X = self.princomp.transform(ranks)
        else:
            X = self.featureData.values

        p = self.clf.predict_proba(X)
        
        return p

    def recommend(self, threshold, criterion, n=10, avoid=False):
      # List top n stocks and their scores

        tmp = pd.merge(self.returns.reset_index(), self.company_info.reset_index(), on='cik', how='inner')        
#        tmp = tmp.set_index(['year','month','cik'])
        
        p = self.apply_model()
        w = p[:,1]
        tmp['w'] = pd.Series(w, index=self.returns.reset_index().index)
        tmp = tmp.sort(columns=['w'],ascending=False,)
        
        if avoid:
          top_ciks = list(tmp['cik'].values)[-n:]
          scores = list(tmp['w'].values)[-n:]
        else:
          top_ciks = list(tmp['cik'].values)[0:n]
          scores = list(tmp['w'].values)[0:n]

        return zip(top_ciks, scores)


    def evaluate(self, threshold,criterion, plot=True):
        # Evaluate performance. If plot=True, display ROC curve and list top-scoring companies with their actual returns.
        
        p = self.apply_model()
        w = p[:,1]
        w = pd.Series(w, index=self.returns.index)        

        tmp = pd.merge(self.returns.reset_index(), self.company_info.reset_index(), on='cik', how='inner')
        tmp = tmp.set_index(['year','month','cik'])
        tmp['w'] = w
        output = self.apply_threshold(self.returns, threshold, criterion)        

        thresh = np.linspace(0,1,100)
        ret = np.zeros(thresh.shape)
        n = np.zeros(thresh.shape)

        for (i,t) in enumerate(thresh):
            w_above = w[w>t]
            n[i] = len(w_above)
            if len(w_above) > 0:
                ret[i] = np.median(self.returns.values[w>t], axis=0)
            else:
                ret[i] = None
        ret_mkt = np.median(self.returns.values, axis=0)
        
        if self.pre_pca:
            ranks = self.convert_features_to_ranks(self.featureData)
            X = self.princomp.transform(ranks)
        else:
            X = self.featureData
        auc = self.ROC(X, output, plot=plot, title='Validation set')
        
        zipped = zip(tmp['ticker'].values, tmp['w'].values, 100 * tmp['one_year_return'].values, tmp['name'].values)
        zipped = sorted(zipped, key = lambda item: item[1], reverse=True)

        if plot:
             print "Ticker | Score | Annual return (%) | Company name "
             print "-----------------------------------------------"
             for item in zipped[:10]:
                 print "%(ticker)5s  | %(p)4.2f  |      %(ret)6.2f       | %(name)s" % {'ticker':item[0],'p':float(item[1]),'ret':item[2],'name':item[3]}
                

    def plot_returns(self,thresh,ret,ret_mkt=None):
        plt.figure(figsize=[5,5])
        plt.plot(thresh, ret)
        plt.title('Performance of score-weighted portfolio')
        if ret_mkt:
          plt.plot(plt.xlim(),[ret_mkt,ret_mkt],'k--',label='Market return')
        #        plt.plot([min(nrange),max(nrange)],[ret_mkt,ret_mkt],'k--',label='Market return')
        #        plt.plot([self.p_optim,self.p_optim],plt.ylim(),'r--',label='Optimized portfolio: return=%(r).2f' % {'r':ret_optim})
        #plt.plot([0,1],[ret_optim,ret_optim],'r--',label='Classifier top %(n)d'%{'n':ntop})
        #plt.legend()
        plt.xlabel('Minimum allowed score')
        plt.ylabel('Portfolio annual return (%)')
        
        plt.show()


    def print_trees(self):
        trees = self.clf.estimators_

        for (i,DT) in enumerate(trees):
            with open("../TreeDiagrams/tree"+str(i)+".dot", 'w') as f:
                f = tree.export_graphviz(DT, out_file=f, feature_names=self.features.keys())
                f.close()


    def feature_importance(self):
        
        feature_values = enumerate(self.clf.feature_importances_)
        feature_values = sorted(feature_values, key = lambda item: item[1], reverse=True)
        if self.pre_pca:
            sorted_names = [str(ix) for (ix,_) in feature_values]
        else:
            col_names = self.column_index.keys()
            col_ix = self.column_index.values()
            sorted_names = [col_names[col_ix.index(ix)] for (ix,_) in feature_values]
        
        importance = [f for (_,f) in feature_values]

        p = self.apply_model()
        p = p[:,1]
        quantiles = np.percentile(p, [25,75])
        med_bottom = lambda x: x[p<quantiles[0]].median()
        med_top = lambda x: x[p>quantiles[1]].median()
        bottom_fv = self.featureData.apply(med_bottom)
        top_fv = self.featureData.apply(med_top)
        
        pos = np.arange(10)
        plt.title('Top 10 features')
        plt.barh(pos, importance[9::-1], color='green')
        plt.yticks(pos+0.5,sorted_names[9::-1])
        plt.xlim(0,importance[0] + 0.02)
        plt.xlabel('Feature importance')
        for pp, name, imp in zip(pos, sorted_names[9::-1], importance[9::-1]):
            try:
                low = bottom_fv[name.lower()]
                hi = top_fv[name.lower()]
            except:
                low = None
                hi = None
            if hi > low:
                s = '+'
            else:
                s = '-'
            plt.annotate(s, xy=(importance[0] + 0.01, pp + .5), va='center', fontsize=20, weight='bold', ha='center')
        
#        plt.figure()
#        plt.title('Bottom 10 features')
#        plt.barh(pos, importance[-10:], color='red')
#        plt.yticks(pos+0.5,sorted_names[-10:])
        plt.show()

#        print ""
#        print ""
#        print "            Top 10 features             |           Bottom 10 features           "
#        print "---------------------------------------------------------------------------------"
#        for i in range(10):
#            print "%(f1)-39s | %(f2)-39s" % {'f1':sorted_names[i], 'f2':sorted_names[-1-i]}
#        print ""
#        print ""
#
#        fig = plt.figure(figsize=(10,8))
#        ax = Axes(fig,[0,0.5,0,0.5])
#        width = 2
#        ind = np.linspace(0,width*len(feature_values),len(feature_values))
#        plt.bar(ind+width/2, [f for (name,f) in feature_values],width,color="r",axes=ax)
#        plt.xticks(ind+width/2, [name for (name,f) in feature_values],rotation=90)


    
    
    def mean_std(self, X, ret):
        score = self.clf.predict_proba(X)
        score = score[:,1]
        p = np.linspace(0,1,100)
        mean = np.zeros(p.shape)
        std = np.zeros(p.shape)
        for (i,t) in enumerate(p):
            mean[i] = np.mean(ret[score > t])
            std[i] = np.std(ret[score > t])
        
        plt.figure()
#        plt.plot(p,mean,'b',label='Mean return')
#        plt.plot(p,std,'r',label='Std. dev.')
#        plt.legend()
        plt.plot(p,mean/std)
        plt.show()
    
    def validation_maxdepth(self, X,y):
        max_depths = np.array(range(1,20))
        train_scores = np.zeros(max_depths.shape)
        valid_scores = np.zeros(max_depths.shape)
        
        for (i,d) in enumerate(max_depths):
            clf = RandomForestClassifier(n_estimators=30,max_features='auto',max_depth=d,oob_score=True)
            clf = clf.fit(X,y)
            train_scores[i] = clf.score(X,y)
            valid_scores[i] = clf.oob_score_
    
        plt.figure()
        plt.plot(max_depths, train_scores,'r',label='Training score')
        plt.plot(max_depths, valid_scores,'g',label='Validation score')
        plt.ylabel('Score')
        plt.xlabel('max_depth')
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
        
        plt.figure(figsize=[5,5])
        #        plt.subplot(1,2,1)
        plt.plot(fpr,tpr)
        plt.title(title + ' ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        #        plt.plot([fpr[ix],fpr[ix]], [0,1], 'r--', label="p = " + str(thresh[ix]))
        #        plt.legend()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        

        plt.show()



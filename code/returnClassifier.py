import numpy as np
import pandas as pd
import sys

from scipy.stats import rankdata

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

from sklearn.svm import SVC, LinearSVC

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sbn

import loaders

class pcaTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, compression=0.95):  
    if compression <= 0 or compression >= 1:
          error("Compression must be in (0,1)")
    self.compression = compression
    self.princomp = []    
  

  def fit(self, X, y=None):
    return self

  def transform(self, X):      
    # Preprocessing for PCA, convert to ranked data for better uniformity
    (n,m) = X.shape
    for i in range(m):
        X[:,i] = rankdata(X[:,i], method='average')

    pca = PCA(n_components = self.compression, whiten=True)
    X_reduced = pca.fit_transform(X)
    self.princomp = pca
    return X_reduced

  def plot(self, X_reduced,y):    
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

class classifierTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,clf):
    self.clf = clf    

  def fit(self,X,y):
    self.clf = self.clf.fit(X,y)

  def transform(self,X):
    try:
      p = self.clf.decision_function(X)
    except:
      p = self.clf.predict_proba(X)[:,1]
    print p[:100]
    return p

class Selector(BaseEstimator, TransformerMixin):
  def __init__(self, key):
    self.key = key

  def fit(self, X,y=None):
    return self

  def transform(self,X):
    return X[self.key]


class returnPredictor:

  def __init__(self):
    
    self.financials = loaders.financialsReader()    
    self.notescount = loaders.notesCountReader()
    self.notestext = loaders.notesTextReader()
    self.returns = loaders.returnsReader(return_dur='quarterly')

    self.rfc = RandomForestClassifier(n_estimators=30,criterion='gini',max_features='auto',max_depth=5)
    self.svc = LinearSVC()
    self.final_svc = SVC(kernel='linear')
    self.pca = pcaTransformer()

    
    self.pipe = {}
    self.pipe['values_pca'] = Pipeline([ ('select', Selector('values')),
                                         ('pca', self.pca),
                                         ('classify', self.rfc) ])    
    self.pipe['values'] = Pipeline([ ('select', Selector('values')),                               
                                     ('classify', self.rfc) ])          
    self.pipe['notes'] = Pipeline([ ('select', Selector('notes')),                               
                                    ('classify', self.svc) ])      
    
    self.pipe['all_pca'] = Pipeline([ ('combine', FeatureUnion(transformer_list=[ 
                                          ('values', Pipeline([ ('select', Selector('values')),
                                                                ('pca', self.pca),
                                                                ('classify', classifierTransformer(self.rfc)) ])),
                                          ('notes', Pipeline([ ('select', Selector('notes')),
                                                               ('classify', classifierTransformer(self.svc)) ]))])),
                           ('classify', self.final_svc)])  
    self.pipe['all'] = Pipeline([ ('combine', FeatureUnion(transformer_list=[ 
                                        ('values', Pipeline([ ('select', Selector('values')),                                                                  
                                                              ('classify', classifierTransformer(self.rfc)) ])),
                                        ('notes', Pipeline([ ('select', Selector('notes')),
                                                             ('classify', classifierTransformer(self.svc)) ]))])),
                                   ('classify', self.final_svc)])



  def train(self, threshold, criterion, datestr=None, exchanges=[], model='values'):
    # Train classifier: threshold, criterion - see apply_threshold() below
    #                   if plot=True, chart most important features

    self.model = model
    if datestr and exchanges:
      params = {'before': datestr, 'exchanges': exchanges}
      
      # print "Reading financial values..."
      # sys.stdout.flush()
      self.financials.train(params)
      index = self.financials.index

      # print "Reading notes wordcount..."
      # sys.stdout.flush()
      self.notescount.train(params)
      index = index.intersection(self.notescount.index)

      # print "Reading returns..."
      # sys.stdout.flush()
      self.returns.train(params)
      index = index.intersection(self.returns.index)
          
      X = pd.concat([self.financials.featureData.reindex(index), self.notescount.featureData.reindex(index)], axis=1)
      
      if self.model != 'values':
        # print "Reading notes text..."
        # sys.stdout.flush()
        self.notestext.train(index)
        notesData = self.notestext.featureData
      else:
        notesData = []

      self.train_index = index
      self.train_data = {'values': X, 'notes': notesData}

    self.train_y = self.returns.apply_threshold(threshold, criterion)    
    self.train_y = self.train_y.reindex(self.train_index)

    print "Training model %s (%d examples)..." % (model, len(self.train_index))
    sys.stdout.flush()
    
    #self.notes_model = self.pipe['notes'].fit(self.train_data, self.train_y)

    if model=='all':
      self.svc = self.svc.fit(self.train_data['notes'], self.train_y)
      self.rfc = self.rfc.fit(self.train_data['values'], self.train_y)
      
      svc_feature = np.expand_dims(self.svc.decision_function(self.train_data['notes']), axis=1)
      rfc_feature = np.expand_dims(self.rfc.predict_proba(self.train_data['values'])[:,1], axis=1)
      plt.scatter(svc_feature, rfc_feature, c=self.train_y, s=1,alpha=0.5, cmap=plt.get_cmap('seismic'))
      plt.show()
      self.final_svc = self.final_svc.fit(np.concatenate([svc_feature, rfc_feature], axis=1), self.train_y)
    else:
      self.pipe[model] = self.pipe[model].fit(self.train_data, self.train_y)

    print 'Min train period: '
    print self.train_y.reset_index().date.min()
    print 'Max train period: '
    print self.train_y.reset_index().date.max()


  def top_ngrams(self, n=10):
    vocab = self.notestext.train_vectorizer.get_feature_names()
    X = self.train_data['notes']
    y = self.train_y
    pos_counts = sum(X[y==1,:]) / sum(y==1)
    neg_counts = sum(X[y==-1,:]) / sum(y==-1)
    
    diff = -pos_counts + neg_counts
    sorted_ix = diff.indices[diff.data.argsort()]
    for ind in sorted_ix[:n]:
      print vocab[ind]


  def evaluate(self, threshold,criterion, before=None, after=None, exchanges=[], plot=True):
    # Evaluate performance. If plot=True, display ROC curve and list top-scoring companies with their actual returns.
    
    if (before or after) and exchanges:
      params = {'before': before, 'after':after, 'exchanges': exchanges}

     
      # print "Reading financial values..."
      # sys.stdout.flush()
      most_recent=True
      self.financials.test(params,most_recent=most_recent)
      index = self.financials.index
      
      # print "Reading notes wordcount..."
      # sys.stdout.flush()
      self.notescount.test(params, most_recent=most_recent)
      index = index.intersection(self.notescount.index)
      
      # print "Reading returns..."
      # sys.stdout.flush()
      self.returns.test(params, most_recent=most_recent)
      index = index.intersection(self.returns.index)
      

      X = pd.concat([self.financials.featureData.reindex(index), self.notescount.featureData.reindex(index)], axis=1)
     
      if self.model != 'values':
        # print "Reading notes text..."
        # sys.stdout.flush()
        self.notestext.test(index)
        notesData = self.notestext.featureData
      else:
        notesData = []

      self.test_data = {'values': X, 'notes': notesData}
      self.test_index = index

    self.test_y = self.returns.apply_threshold(threshold, criterion)    
    self.test_y = self.test_y.reindex(self.test_index)

    print "Evaluating model %s (%d examples)..." % (self.model, len(self.test_index))

    print 'Min train period: '
    print self.test_y.reset_index().date.min()
    print 'Max train period: '
    print self.test_y.reset_index().date.max()
    sys.stdout.flush()

    if self.model=='all':
      svc_feature = np.expand_dims(self.svc.decision_function(self.test_data['notes']), axis=1)
      rfc_feature = np.expand_dims(self.rfc.predict_proba(self.test_data['values'])[:,1], axis=1)
      plt.scatter(svc_feature, rfc_feature, c=self.test_y, s=1,alpha=0.5, cmap=plt.get_cmap('seismic'))
      plt.show()
      #p = rfc_feature
      p = self.final_svc.decision_function(np.concatenate([svc_feature, rfc_feature], axis=1))
    else:      
      try:
        p = self.pipe[self.model].decision_function(self.test_data)    
      except:
        p = self.pipe[self.model].predict_proba(self.test_data)[:,1]
    w = pd.Series(p.ravel(), index=self.test_index)
    
    fpr,tpr,thresh = roc_curve(self.test_y, w)
    roc_auc = auc(fpr,tpr)
    if plot:
      self.show_ROC(fpr,tpr,thresh,roc_auc)
    return roc_auc


    # tmp = pd.merge(self.returns.reset_index(), self.company_info.reset_index(), on='cik', how='inner')
    # tmp = tmp.set_index(['year','month','cik'])
    # tmp['w'] = w
    # output = self.apply_threshold(self.returns, threshold, criterion)        

    # thresh = np.linspace(0,1,100)
    # ret = np.zeros(thresh.shape)
    # n = np.zeros(thresh.shape)

    # for (i,t) in enumerate(thresh):
    #     w_above = w[w>t]
    #     n[i] = len(w_above)
    #     if len(w_above) > 0:
    #         ret[i] = np.median(self.returns.values[w>t], axis=0)
    #     else:
    #         ret[i] = None
    # ret_mkt = np.median(self.returns.values, axis=0)
    
    # if self.pre_pca:
    #     ranks = self.convert_features_to_ranks(self.featureData)
    #     X = self.princomp.transform(ranks)
    # else:
    #     X = self.featureData
    # auc = self.ROC(X, output, plot=plot, title='Validation set')
    
    # zipped = zip(tmp['ticker'].values, tmp['w'].values, 100 * tmp['one_year_return'].values, tmp['name'].values)
    # zipped = sorted(zipped, key = lambda item: item[1], reverse=True)

    # if plot:
    #      print "Ticker | Score | Annual return (%) | Company name "
    #      print "-----------------------------------------------"
    #      for item in zipped[:10]:
    #          print "%(ticker)5s  | %(p)4.2f  |      %(ret)6.2f       | %(name)s" % {'ticker':item[0],'p':float(item[1]),'ret':item[2],'name':item[3]}
            


  
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

  def apply_model(self):
    if self.model=='all':
      svc_feature = np.expand_dims(self.svc.decision_function(self.test_data['notes']), axis=1)
      rfc_feature = np.expand_dims(self.rfc.predict_proba(self.test_data['values'])[:,1], axis=1)      
      p = self.final_svc.decision_function(np.concatenate([svc_feature, rfc_feature], axis=1))
    else:      
      try:
        p = self.pipe[self.model].decision_function(self.test_data)    
      except:
        p = self.pipe[self.model].predict_proba(self.test_data)[:,1]
    w = pd.Series(p.ravel(), index=self.test_index)
      
    return w

  def recommend(self, threshold, criterion, n=10, avoid=False):
    # List top n stocks and their scores

      #tmp = pd.merge(self.returns.featureData.reset_index(), self.financials.company_info.reset_index(), on='cik', how='inner')        
#        tmp = tmp.set_index(['year','month','cik'])

      tmp = pd.DataFrame(index=self.test_index, columns=['w'])
      tmp['w'] = self.apply_model()      
      tmp = tmp.reset_index()      

      # max_score = tmp['w'].max()
      # scores = tmp.loc[tmp['w'] > 0.75*max_score, 'w']
      # top_ciks = tmp.loc[tmp['w']>0.75*max_score, 'cik']

      tmp = tmp.sort(columns=['w'],ascending=False,)
      

      if avoid:
        top_ciks = list(tmp['cik'].values)[-n:]
        scores = list(tmp['w'].values)[-n:]
      else:
        top_ciks = list(tmp['cik'].values)[0:n]
        scores = list(tmp['w'].values)[0:n]

      print '%d scores in range (%0.2f, %0.2f)' % (len(scores), min(scores), max(scores))
      return zip(top_ciks, scores)


 
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

  def mkt_portfolio(self, n=500):     
      
      tmp = self.test_data['values'].reset_index().sort(columns=['marketcap'],ascending=False)
      top_ciks = list(tmp['cik'].values)[0:n]
      #scores = list(tmp['marketcap'].values)[0:n]
      scores = [1.0]*n
      return zip(top_ciks, scores)

  def momentum_portfolio(self, n=100):
      def max_date(g):
        g = g.reset_index()
        g = g.iloc[g['date'].argmax()]
        g = g.drop('cik')
        return g
      
      df = pd.DataFrame({'y':self.train_y}, index=self.train_y.index)
      last_y = df.groupby(level='cik').apply(max_date).reset_index().set_index(['date'])
      
      ciks = last_y.cik.values
      y = last_y.y.values
      pos = ciks[y==1]
      np.random.shuffle(pos)
      return zip(pos[:n], [1.0]*n)



  def feature_importance(self):
      
      feature_values = enumerate(self.rfc.feature_importances_)
      feature_values = sorted(feature_values, key = lambda item: item[1], reverse=True)
            
      col_names = self.train_data['values'].columns
      sorted_names = [col_names[ix] for (ix,_) in feature_values]
  
      importance = [f for (_,f) in feature_values]

      try:
        p = self.pipe.decision_function(self.train_data)    
      except:
        p = self.pipe.predict_proba(self.train_data)[:,1]
      
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

     
  def show_ROC(self, fpr,tpr,thresh,roc_auc, title=''):
      
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



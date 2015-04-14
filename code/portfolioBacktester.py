import pandas as pd
import numpy as np
import psycopg2
import pickle
import heapq
import datetime
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy.stats import linregress
import sys

def next_trading_price(date, pr):	
	date0 = date
	while date <= date0 + np.timedelta64(3,'D'):		
		
		try:
			cur_pr = pr.loc[date]
			if not pd.isnull(cur_pr):
				return cur_pr
			else:
				date += np.timedelta64(1,'D')	
		except:
			date += np.timedelta64(1,'D')
	return None	

def tickers_to_ciks(tickers):
	conn = psycopg2.connect('dbname=secdata user=vagrant password=pwd')
	ciks = []
	for ticker in tickers:
		cik = pd.read_sql('SELECT cik FROM companies WHERE ticker=%s',conn,params=(ticker,))
		ciks.append(cik.loc[0,'cik'])
	conn.close()
	return ciks

Event = namedtuple('Event', ['time','ciks','data','type'])

class Portfolio:
	def __init__(self, ciks=[], weights=[], date=None):		
		self.max_time = pd.to_datetime('12 2015','%m %Y')
		self.min_time = pd.to_datetime('01 2009','%m %Y')
		self.index = pd.date_range(self.min_time, self.max_time, freq='D')

		self.all_prices = pd.DataFrame(columns=['cash'] + ciks, index=self.index)
		self.all_prices['cash'] = 1
		self.num_shares = pd.Series(0,index=['cash'] + ciks)		

		self.event_queue = []
		self.event_handlers = {'liquidate': self.liquidation_handler, 
		                       'rebalance': self.rebalance_handler, 
		                       'buy': self.buy_handler, 
		                       'sell': self.sell_handler}

		if date:
			self.construct(ciks,weights,date)		

		self.last_trade_date = date				

	def construct(self, ciks, weights, date, value=1):
		for (cik, weight) in zip(ciks,weights):							
			pr = self.load_stock(cik)			
			cur_price = next_trading_price(date, pr)

			if cur_price:
				self.num_shares.loc[cik] = value * weight / cur_price
			else:
				self.num_shares.loc[cik] = 0

		if not ciks:
			self.num_shares.loc['cash'] = value
		
		self.value_ = self.all_prices.dot(self.num_shares)
		self.value_.loc[self.value_.index < date] = 0
		self.last_trade_date = date
		self.inception_date = date
		
		return self

	def load_stock(self, cik):		
		try: 
			return self.all_prices[cik]
		except:			
			#try:
			base_dir = '../data/edgar/'
			f = open(base_dir + cik + '/all_daily_prices.pickle','r')
			pr = pickle.load(f)
			f.close()	

			pr = pr.iloc[:-2].astype('float')

			# On latest date for which we have a price, assume position gets liquidated at that price			
			heapq.heappush(self.event_queue, Event(time=pr.index[0], ciks=[cik], data=None, type='liquidate'))
			# except:				
			# 	pr = pd.Series(float('NaN'),index=self.index)
				#raise ValueError('No price information for CIK %s' % cik)	
			
			self.all_prices[cik] = pr			
			self.num_shares[cik] = 0
			return pr.reindex(self.index)
			
	def values(self):
		return self.value_.loc[np.logical_and(self.value_.index >= self.inception_date, self.value_.index <= self.last_trade_date)]

	def liquidation_handler(self, event):
		# Sell all shares of a stock (or cover short position), then reinvest proceeds into remaining holdings
		date = event.time
		cik = event.ciks[0]

		if self.num_shares[cik] != 0:
			V = next_trading_price(date,self.value_)	
			p = next_trading_price(date,self.all_prices[cik])
			C = p*self.num_shares[cik]

			self.num_shares[cik] = 0
			self.num_shares = self.num_shares * (V / (V-C))
		

	def rebalance_handler(self, event):
		# Reweight holdings without net buying/selling
		date = event.time
		ciks = event.ciks
		weights = event.data

		alpha = 0.5

		tot = 0
		V = next_trading_price(date,self.value_)						
		
		self.num_shares = pd.Series(0, index=self.num_shares.index)
		for (cik, weight) in zip(ciks,weights):	
			pr = self.load_stock(cik)
			cur_price = next_trading_price(date, pr)			
			if cur_price:				
				# print cur_price
				# sys.stdout.flush()
				tot += weight
				self.num_shares.loc[cik] += (weight * V) / cur_price
			
		# if (tot > 0 and V <= 0) or (tot < 0 and V >= 0) or (tot==0 and V != 0):
		# 	raise ValueError('Rebalancing may not change sign of net portfolio value!')	
		self.num_shares = self.num_shares / tot
		
	def sell_handler(self,event):		
		date = event.time
		ciks = event.ciks
		nshares = event.data

		for (cik,N) in zip(ciks,nshares):
			pr = self.load_stock(cik)
			cur_price = next_trading_price(date, pr)
			if cur_price:
				self.num_shares.loc['cash'] += cur_price * N
				self.num_shares.loc[cik] -= N

	def buy_handler(self,event):
		date = event.time
		ciks = event.ciks
		nshares = event.data

		for (cik,N) in zip(ciks,nshares):
			pr = self.load_stock(cik)
			cur_price = next_trading_price(date, pr)
			if cur_price:
				self.num_shares.loc['cash'] -= cur_price * N
				self.num_shares.loc[cik] += N

	def checkdate(fun):
		def wrapper(self,*args,**kwargs):
			if kwargs['date'] > self.last_trade_date:
				fun(self,*args,**kwargs)
				return self
			else:				
				raise ValueError('Transaction cannot precede previous commit date!')
		return wrapper

	
	def commit(self, date):		
		# Handle all events before date, then set last_trade_date = date
		# After calling commit, self.value_ is accurate for all times before date
		# and self.num_shares holds between date and next unprocessed event

		while self.event_queue and self.event_queue[0].time <= date:			
			event = heapq.heappop(self.event_queue)			
			V0 = next_trading_price(event.time,self.value_)						
			self.event_handlers[event.type](event)
			
			tmp = self.all_prices.dot(self.num_shares)
			self.value_.loc[self.value_.index >= event.time] = tmp.loc[self.value_.index >= event.time]
			V1 = next_trading_price(event.time,self.value_)						
			#print '%s event: %0.2f -> %0.2f' % (event.type, V0, V1)
		self.last_trade_date = date
		
		return self

	@checkdate	
	def fastforward(self,date=None):
		self.commit(date)
		return self

	@checkdate	
	def sell(self, ciks, amount, date, commit=True):
		heapq.heappush(self.event_queue, Event(time=date, ciks=ciks, data=amount, type='sell'))				
		if commit:
			self.commit(date)

	@checkdate	
	def buy(self, ciks, amount, date, commit=True):		
		heapq.heappush(self.event_queue, Event(time=date, ciks=ciks, data=amount, type='buy'))					
		if commit:
			self.commit(date)

	@checkdate	
	def rebalance(self, ciks, weights, date, commit=True):				
		heapq.heappush(self.event_queue, Event(time=date, ciks=ciks, data=weights, type='rebalance'))		
		if commit:
			self.commit(date)
		


class portfolioBacktester:
	def __init__(self, inception_date):
		self.inception_date = inception_date
		self.max_time = pd.to_datetime('12 2015','%m %Y')
		self.strategies = {}
		self.FF = self.read_FF()
		self.FF = self.FF.loc[self.FF.index >= self.inception_date]


	def read_FF(self):
	    fname = '../data/F-F_Research_Data_Factors.txt'
	    f = open(fname,'r')
	    rows = f.read().split('\n')
	    vals = []
	    index = []
	    for row in rows[1:]:
	        items = row.split()
	        if items:            
	            date = items[0]
	            vals.append({'Mkt-RF': float(items[1]), 
	            	         'SMB': float(items[2]),
	            	         'HML': float(items[3]), 
	            	         'RF': float(items[4])})
	            year = date[:4]
	            mo = date[-2:]
	            index.append(pd.Period(year+'-'+mo,'M'))
	    return pd.DataFrame(vals, index=index)


	def create_strategy(self, name, ciks=[], weights=[]):
		if weights:
			norm = abs(sum(weights))
			weights = map(lambda w: w/norm, weights)
		self.strategies[name] = Portfolio(ciks, weights, self.inception_date)

	def values(self, begin=None, end=None):
		if begin==None:
			begin = self.inception_date
		if end==None:
			end = self.max_time
		tindex = pd.date_range(begin,end,freq='D')
		df = pd.DataFrame(columns=self.strategies.keys(), index=tindex)
		for name, p in self.strategies.iteritems():
			try:
				v = p.fastforward(date=end).values()
			except:
				v = p.values()
			df[name] = pd.Series(map(lambda t: next_trading_price(t,v), tindex), index=tindex)			
		return df

	def binned_returns(self, begin, end):
		tbins = pd.date_range(begin,end,freq='M')
		df = pd.DataFrame(columns=self.strategies.keys(), index=pd.PeriodIndex(tbins[:-1].values, freq='M'))
		for name, p in self.strategies.iteritems():
			try:
				v = p.fastforward(date=end).values()
			except:
				v = p.values()
			
			vsub = pd.Series(map(lambda t: next_trading_price(t,v), tbins), index=tbins)
			delta = np.diff(vsub.values)
			df[name] = pd.Series(12 * 10 * delta / vsub.values[:-1], index=pd.PeriodIndex(tbins[:-1].values, freq='M'))
		return df

	def CAPM(self,end, strat):
		ret = self.binned_returns(self.inception_date, end)		
		beta = self.regress(ret[strat], ['Mkt-RF'])
		print 'alpha = %0.2f' % beta[0]
		print 'Mkt beta = %0.2f' % beta[1] 
		
		plt.scatter(self.FF['Mkt-RF'], ret[strat] - self.FF['RF'])
		xmin, xmax = plt.xlim()

		plt.plot([xmin,xmax], [beta[0] + beta[1]*xmin, beta[0] + beta[1]*xmax])
		plt.show()

		return {'alpha':beta[0], 'Mkt beta':beta[1]}

	def FamaFrench(self,end,strat):
		ret = self.binned_returns(self.inception_date, end)		
		beta = self.regress(ret[strat], ['Mkt-RF','HML','SMB'])

		print 'alpha = %0.2f' % beta[0]
		print 'Mkt beta = %0.2f' % beta[1] 
		print 'HML beta = %0.2f' % beta[2]
		print 'SMB beta = %0.2f' % beta[3]

		plt.scatter(self.FF['Mkt-RF'], ret[strat] - self.FF['RF'])
		xmin, xmax = plt.xlim()

		plt.plot([xmin,xmax], [beta[0] + beta[1]*xmin, beta[0] + beta[1]*xmax])
		plt.show()
		return {'alpha':beta[0], 'Mkt beta':beta[1], 'HML beta':beta[2], 'SMB beta':beta[3]}

	def regress(self, ret, cols):
		
		tmp = pd.DataFrame(index = ret.index)
		tmp['y'] = (ret - self.FF['RF'])
		tmp['ones'] = 1
		for col in cols:
			tmp[col] = self.FF[col]
		
		tmp = tmp.loc[pd.notnull(tmp['y'])]
		
		y = tmp.y.values
		X = tmp.drop('y',1).values		
		
		return np.linalg.lstsq(X, y)[0]











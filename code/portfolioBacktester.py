import pandas as pd
import numpy as np
import psycopg2
import pickle
import heapq
from collections import namedtuple

def next_trading_price(date, pr):	
	while date <= pr.index.max():
		try:
			cur_pr = pr.loc[date]
			if not pd.isnull(cur_pr):
				return cur_pr
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
		self.all_prices = pd.DataFrame()
		self.num_shares = pd.Series(0,index=ciks)		

		self.event_queue = []
		self.event_handlers = {'liquidate': self.liquidation_handler, 
		                       'rebalance': self.rebalance_handler, 
		                       'buy': self.buy_handler, 
		                       'sell': self.sell_handler}

		if date:
			self.construct(ciks,weights,date)		

		self.last_trade_date = date
		self.cash = 0

	def construct(self, ciks, weights, date, value=1):
		for (cik, weight) in zip(ciks,weights):	
			pr = self.load_stock(cik)
			cur_price = next_trading_price(date, pr)
			if cur_price:
				self.num_shares.loc[cik] = value * weight / cur_price
			else:
				self.num_shares.loc[cik] = 0

		self.value_ = self.all_prices.dot(self.num_shares)
		self.value_.loc[self.value_.index < date] = 0
		self.last_trade_date = date
		self.inception_date = date
		self.cash = -value * sum(weights)

		return self

	def load_stock(self, cik):
		try: 
			return self.all_prices[cik]
		except:			
			try:
				base_dir = '../data/edgar/'
				f = open(base_dir + cik + '/all_daily_prices.pickle','r')
				pr = pickle.load(f)
				f.close()	
			except:				
				raise ValueError('No price information for CIK %s' % cik)	
			pr = pr.iloc[:-2].astype('float')

			# On latest date for which we have a price, assume position gets liquidated at that price			
			heapq.heappush(self.event_queue, Event(time=pr.index[0], ciks=[cik], data=None, type='liquidate'))

			self.all_prices[cik] = pr			
			self.num_shares[cik] = 0
			return pr.reindex(self.all_prices.index)					
			
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

		tot = sum(weights)
		V = next_trading_price(date,self.value_)		
		if (tot > 0 and V <= 0) or (tot < 0 and V >= 0) or (tot==0 and V != 0):
			raise ValueError('Rebalancing may not change sign of net portfolio value!')

		self.num_shares = pd.Series(0, index=self.num_shares.index)
		for (cik, weight) in zip(ciks,weights):	
			pr = self.load_stock(cik)
			cur_price = next_trading_price(date, pr)
			if cur_price:
				self.num_shares.loc[cik] = (weight * V) / (tot * cur_price)
			else:
				self.num_shares.loc[cik] = 0		

	def sell_handler(self,event):		
		date = event.time
		ciks = event.ciks
		nshares = event.data

		for (cik,N) in zip(ciks,nshares):
			pr = self.load_stock(cik)
			cur_price = next_trading_price(date, pr)
			if cur_price:
				self.cash += cur_price * N
				self.num_shares.loc[cik] -= N

	def buy_handler(self,event):
		date = event.time
		ciks = event.ciks
		nshares = event.data

		for (cik,N) in zip(ciks,nshares):
			pr = self.load_stock(cik)
			cur_price = next_trading_price(date, pr)
			if cur_price:
				self.cash -= cur_price * N
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
			self.event_handlers[event.type](event)
			tmp = self.all_prices.dot(self.num_shares)
			self.value_.loc[self.value_.index >= event.time] = tmp.loc[self.value_.index >= event.time]

		self.last_trade_date = date
		
		return self

	@checkdate	
	def fastforward(self,date):
		self.commit(date)

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
	def __init__(self):
		self.conn = psycopg2.connect('dbname=secdata user=vagrant password=pwd')
		f = open('../data/edgar/prices_all_months_all_stocks.pickle','r')
		self.prices = pickle.load(f)
		f.close()

		self.weights = pd.Series(np.zeros(prices.shape[1]), index=prices.columns)


	def __del__(self):
		self.conn.close()

	#def binned_returns(self, begin, duration=np.timedelta64(1,'M')):
	
	#def rebalance(weights, t):

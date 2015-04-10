import pandas as pd
import numpy as np
import psycopg2
import pickle


def next_trading_price(date, pr):	
	while date <= pr.index[0]:
		try:
			return pr.loc[date]
		except:
			date += np.timedelta64(1,'D')
	return None	


class Portfolio:
	def __init__(self, ciks=[], weights=[], date=None):
		base_dir = '../data/edgar/'
		self.all_prices = pd.DataFrame()
		self.num_shares = pd.Series(index=ciks)
		for (cik, weight) in zip(ciks,weights):
			try:
				f = open(base_dir + cik + '/all_daily_prices.pickle','r')
				pr = pickle.load(f)
				f.close()		
				pr = pr.iloc[:-2].astype('float')						
				self.all_prices[cik] = pr
					
			except:
				print 'No price information for CIK %s' % cik
				raise

			if date:
				cur_price = next_trading_price(date, pr)
				self.num_shares.loc[cik] = weight / cur_price
			else:
				self.num_shares.loc[cik] = 0
		
		self.value = self.all_prices.dot(self.num_shares)
		if date:
			self.value.loc[self.value.index < date] = 0
		self.last_trade_date = date

	def construct(ciks, weights, date):
		for (cik, weight) in zip(ciks,weights):	
			try:
				f = open(base_dir + cik + '/all_daily_prices.pickle','r')
				pr = pickle.load(f)
				f.close()			
				pr = pr.iloc[:-2].astype('float')
				self.all_prices[cik] = pr								
			except:
				print 'No price information for CIK %s' % cik
				raise
			cur_price = next_trading_price(date, pr)
			self.num_shares.loc[cik] = weight / cur_price
		self.value = self.all_prices.dot(self.num_shares)
		self.value.loc[self.value.index < date] = 0
		self.last_trade_date = date

	def rebalance(ciks, weights, date):
		if date < self.last_trade_date:
			print 'Cannot rebalance before previous trade date!'
			raise
		else:
			self.lat_trade_date = date
		




class portfolioBacktester:
	def __init__(self):
		self.conn = psycopg2.connect('dbname=secdata user=vagrant password=pwd')
		f = open('../data/edgar/prices_all_months_all_stocks.pickle','r')
		self.prices = pickle.load(f)
		f.close()

		self.weights = pd.Series(np.zeros(prices.shape[1]), index=prices.columns)


	def __del__(self):
		self.conn.close()

	#def rebalance(weights, t):

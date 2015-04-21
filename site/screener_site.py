from flask import Flask, render_template, redirect, url_for, request, jsonify
import pandas as pd
import psycopg2
import numpy as np
from returnClassifier import returnPredictor
import portfolioBacktester as ptf
from io import StringIO
import pickle
import sys
import time
from math import floor

app = Flask(__name__)
conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")
pred_pos = returnPredictor()
pred_neg = returnPredictor()

@app.route('/')
def main_site():
	return render_template('chart_ui.html')

#@app.route('/filter/<filter_tag>/')
@app.route('/filter/<filter_tag>/<page>')
def main_site_filtered(filter_tag, page=0):
	#cur = conn.cursor()
	rows_per_page = 100
	pages_to_display = 15
	exchange_codes = {'amex':'A','nyse':'N','nasdaq':'NASDAQ','otc':'OTC','p':'P','z':'Z'}
	df = pd.read_sql("SELECT name,ticker,cik FROM companies WHERE exchange=%s ORDER BY name;",conn, params=(exchange_codes[filter_tag],))	
	
	
	def format_coname(row):
		coname = row[0]
		coname = coname.split('Common')[0].strip()
		if coname[-1] == '-':
			coname = coname[:-1]
		row[0] = coname		
		return row

	rows = map(format_coname, [row for row in df.values if row[2]])
	numrows = len(rows)
	numpages = numrows / rows_per_page + 1
	minpage = pages_to_display * ((int(page)-1) / pages_to_display)  
	maxpage = min(numpages,minpage + pages_to_display)
	min_row = rows_per_page * (int(page)-1)
	max_row = min_row + rows_per_page
	# for row in cur:		
	# 	if row[0]:			
	# 		rows = rows.append(list(row))			
	# cur.close()
	headings = ['Name', 'Ticker', 'CIK']
	return render_template('list_by_exchange.html', filter=filter_tag, rows=rows[min_row:max_row], headings=headings, minpage=minpage, maxpage=maxpage, numpages=numpages, curpage=int(page))

@app.route('/get_chart')
def get_chart():
	date = request.args['date']
	n = int(request.args['n'])
	rebalance_freq = int(request.args['rebalance'])
	include_otc = request.args['otc']
	include_exchange = request.args['exchange']

	venue = []
	if include_otc=='true':
		venue = venue + ['OTC']
	if include_exchange=='true':
		venue = venue + ['exchange']
	
#	ciks = ptf.tickers_to_ciks([ticker])
	
	end_date = pd.to_datetime('2015-01-01','%Y-%m-%d')
	max_date = pd.to_datetime('2014-06-01','%Y-%m-%d')
	init_date = pd.to_datetime(date, '%Y-%m-%d')
	rebalance_date = init_date
	
	portfolio.strategies['S&P 500'].rebalance(['SPY'],[1.0], date=rebalance_date)
	while rebalance_date < max_date:		
		(short_ciks, short_weights) = top_scorers(rebalance_date, shorts_lookup, n=n, venue=venue)
		(long_ciks,long_weights) = top_scorers(rebalance_date, longs_lookup, n=n, venue=venue)
		# short_ciks = ['0001092367']
		# long_ciks = ['0000001750']
		portfolio.strategies['Suggested portfolio'].rebalance(short_ciks + long_ciks, [-0.25]*len(short_ciks) + [1.0]*len(long_ciks), date=rebalance_date)		
		if rebalance_freq > 0:
			rebalance_date += np.timedelta64(rebalance_freq,'M')
		else:
			break

	v = portfolio.values(begin=inception_date,end=end_date)	

	FF_SPY = portfolio.FamaFrench(init_date, end_date, 'S&P 500')	
	FF_screener = portfolio.FamaFrench(init_date, end_date, 'Suggested portfolio')

	portfolio.strategies['Suggested portfolio'].reset()
	portfolio.strategies['S&P 500'].reset()

	v = v.fillna(method='ffill')
	
	t = v.index
	X = map(lambda d: '%d-%02d-%02d' %(d.year, d.month, d.day), t)
	Y = v['Suggested portfolio'].values
				
	values = [{'x':x, 'y':y} for (x,y) in zip(X,Y) if pd.notnull(y)]
	spy = [{'x':x, 'y':y} for (x,y) in zip(X, v['S&P 500'].values) if pd.notnull(y)]

	
	mo_names = {'01':'Jan', '02':'Feb', '03':'Mar', '04':'Apr', '05':'May', '06':'Jun', '07':'Jul', '08':'Aug', '09':'Sep', '10':'Oct', '11':'Nov', '12':'Dec'}
	
	short_companies = [{'name':name, 'ticker':ticker, 'cik':cik} for (name,ticker,cik) in get_company_info(short_ciks)]
	long_companies = [{'name':name, 'ticker':ticker, 'cik':cik} for (name,ticker,cik) in get_company_info(long_ciks)]
	return jsonify([('result', [
		                 {'values':values, 'key':'Suggested portfolio', 'color':'#009999'},
		                 {'values':spy, 'key':'S&P 500', 'color':'#990000'}
		                       ]),
	                ('short_companies', short_companies), 
	                ('long_companies', long_companies),
	                ('last_rebalance_date', '%s, %d' % (mo_names['%02d' % rebalance_date.month], rebalance_date.year)),
	                ('FF_SPY', FF_SPY),
	                ('FF_screener', FF_screener)])
	                
def get_company_info(ciks):	
	conames = []
	tickers = []
	for cik in ciks:		
		db = pd.read_sql("SELECT name,ticker FROM companies WHERE cik=%s;", conn, params=(cik,))	
		coname = db.loc[0,'name']
		ticker = db.loc[0,'ticker']
		coname = coname.split('Common')[0].strip()
		if coname[-1] == '-':
			coname = coname[:-1]
		conames.append(coname)
		tickers.append(ticker)		
	return zip(conames, tickers,ciks)


@app.route('/chart/')
def chart():
	return render_template('chart_ui.html')

@app.route('/screener/')
def screener():
	recommended = pred_pos.recommend(50,'topquant')
	to_avoid = pred_neg.recommend(-0.2,'lt')

	def fetch_coinfo(tup):
		cik = tup[0]
		score = '%0.2f' % tup[1]
		exchanges = {'A':'AMEX','N':'NYSE','NASDAQ':'NASDAQ','OTC':'OTC','P':'P','Z':'Z'}
		df = pd.read_sql("SELECT name,ticker,cik,exchange FROM companies WHERE cik=%s;",conn, params=(cik,))	
		coname = df.loc[0,'name']
		coname = coname.split('Common')[0].strip()
		if coname[-1] == '-':
			coname = coname[:-1]
		ticker = df.loc[0,'ticker']
		exchange = exchanges[df.loc[0,'exchange']]

		return (coname, ticker, exchange, cik, score)

	rec_list = map(fetch_coinfo, recommended)
	avoid_list = map(fetch_coinfo, to_avoid)

	return render_template('screener_template.html', rec_list=rec_list, avoid_list=avoid_list)


@app.route('/companies/search')
def lookup_by_ticker():		
	ticker = request.args['ticker']
	cik = pd.read_sql("SELECT cik FROM companies WHERE ticker=%s;", conn, params=(ticker,))
	cik = cik.loc[0,'cik']
	return redirect(url_for('lookup_cik',cik=cik))		

@app.route('/companies/<cik>')
def lookup_cik(cik):	
	
	
	pretty_names_cf = [('CashAndCashEquivalentsPeriodIncreaseDecrease', 'Cash flows'),
                   ('netcashprovidedbyusedinoperatingactivities', 'Cash flows from operations'),                   
                   ( 'NetCashProvidedByUsedInFinancingActivities', 'Cash flows from financing'),                   
                   ('NetCashProvidedByUsedInInvestingActivities', 'Cash flows from investing')]                  
    # Income statement items
	pretty_names_soi = [('Revenues', 'Revenues'),
						('costofrevenue', 'Cost of revenue'),
					    ('grossprofit' , 'Gross profit'),
					    ('operatingexpenses' , 'Operating expenses'),
				        ('operatingincomeloss' , 'Earnings from operations'),
				        ('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 'Earnings from operations, before tax'),
				        ('incometaxexpensebenefit' , 'Income tax expense'),
				        ('interestanddebtexpense' , 'Interest expense'),				       
				        ('nonoperatingincomeexpense' , 'Non-operating earnings'),
				        ('profitloss' , 'Net earnings')]
				        #('NetIncomeLossAvailableToCommonStockholdersBasic' , 'Net earnings available to common')]
	pretty_names_sfp = [('CashCashEquivalentsAndShortTermInvestments', 'Cash and equivalents'),						
						('ReceivablesNetCurrent','Net receivables'),
						('InventoryNet', 'Net inventory'),
						('PrepaidExpenseAndOtherAssetsCurrent','Prepaid expenses and other current assets'),
						('DeferredCostsCurrent','Deferred costs'),
                   		('DerivativeInstrumentsAndHedges','Derivative instruments and hedges'),
						('AssetsCurrent', 'Current assets'),
						('PropertyPlantAndEquipmentNet', 'Property, plant and equipment, net of depreciation'),
	                    ('AssetsNoncurrent', 'Noncurrent assets'),
	                    ('assets' , 'Assets'),
                   		('AccountsPayableAndAccruedLiabilitiesCurrent', 'Accounts payable'),
                   		('DebtCurrent', 'Current portion of long-term debt'),
                   		('DerivativeInstrumentsAndHedgesLiabilities', 'Derivative liabilities'),
	                    ('LiabilitiesCurrent', 'Current liabilities'),
	                    ('LongTermDebtAndCapitalLeaseObligations', 'Long-term debt and capital lease obligations'),
	                    ('LiabilitiesOtherThanLongtermDebtNoncurrent','Other long-term liabilities'),
	                    ('LiabilitiesNoncurrent', 'Noncurrent liabilities'),
	                    ('liabilities' , 'Liabilities'),                                                         
	                    ('equity' , 'Equity')]
	                    #'LiabilitiesAndEquity',
    # Price items   
	pretty_names_stock = [('marketcap' , 'Market capitalization'),
	                      ('entitycommonstocksharesoutstanding' , 'Common stock shares'),
	                      ('start_price' , 'Price at beginning of next quarter'),
	                      ('one_year_price' , 'Price after one year'),
	                      ('one_year_return' , 'Annualized return')]
	
	def normalize_colnames(tup):
		item = tup[0]
		colname_maxlen = 63
		item = item.lower()
		if len(item) > colname_maxlen:
			item = item[:colname_maxlen]
		return (item, tup[1])
	pretty_names_cf = map(normalize_colnames, pretty_names_cf)
	pretty_names_soi = map(normalize_colnames, pretty_names_soi)
	pretty_names_sfp = map(normalize_colnames, pretty_names_sfp)


	financials = pd.read_sql("SELECT * FROM financials WHERE cik=%s ORDER BY year,month;", conn, index_col = ['year','month'], params=(cik,))#, index_col='year')
	db = pd.read_sql("SELECT name,ticker FROM companies WHERE cik=%s;", conn, params=(cik,))	
	coname = db.loc[0,'name']
	ticker = db.loc[0,'ticker']
	coname = coname.split('Common')[0].strip()
	if coname[-1] == '-':
		coname = coname[:-1]
	# #coname = coname.
	
	units = ('Millions',1e6)
	# if abs(financials['profitloss'].min()) > 1e6:
	# 	units = ('Millions', 1e6)
	# else:
	# 	units = ('Thousands', 1e3)

	tables = [('Income Statement ($ ' + units[0] + ')', pretty_names_soi),
			  ('Cash Flow Statement ($ ' + units[0] + ')', pretty_names_cf),     
     		  ('Balance Sheet ($ ' + units[0] + ')',pretty_names_sfp),
      		  ('Stock Performance',pretty_names_stock)]
	max_cols = 6
	report_tables = []
	for tname, tlabels in tables:
		rows = []	
		for col,name in tlabels:
			
			def format_val(val):
				if val and not np.isnan(val):
					if tname == 'Stock Performance':
						return '{:0,.2f}'.format(val)
					else:
						return '{:0,.2f}'.format(val/units[1])
				else:
					return '--'

			formatted = map(format_val, financials[col].values)			
			if len(formatted) > max_cols:
				formatted = formatted[-max_cols:]
			rows = rows + [[name] + formatted]
		report_tables.append((tname, rows))
	
	mo_names = {'01':'Jan', '02':'Feb', '03':'Mar', '04':'Apr', '05':'May', '06':'Jun', '07':'Jul', '08':'Aug', '09':'Sep', '10':'Oct', '11':'Nov', '12':'Dec'}
	
	headings = [(yr,mo_names[mo]) for (yr,mo) in list(financials.index.values)]
	if len(headings) > max_cols:
		headings = headings[-max_cols:]

	# years = np.unique([yr for (yr,_) in list(financials.index.values)])

	# yrs_cnt = []
	# for yr in years:
	# 	yrs_cnt.append( (yr, len([mo for (x,mo) in list(financials.index.values) if x==yr])) )
	return render_template('report_table.html', headings=headings, tables=report_tables, title=coname, ticker=ticker)

# @app.route('/screener/')
# def screener():


def load_scores(pickle_file):
	yrs = range(2010,2015)
	mos = range(1,13)
	yrmos = [('%d'%yr,'%02d'%mo) for yr in yrs for mo in mos]

	df = pd.DataFrame(columns=['cik','date','w'])	
	for (yr,mo) in yrmos:    
	    fname = '../data/pickles/%s/%s/%s' % (yr,mo,pickle_file)
	    ok = False
	    try:
	        f = open(fname,'r')
	        d = pickle.load(f)
	        f.close()
	        ok = True
	    except:
	        pass
	    if ok:	    	
	        df = df.append(d['scores'].reset_index(), ignore_index=True)
	#all_scores = df.reset_index().groupby(['cik','date']).apply(lambda g: g['w'].mean())
	
	all_scores = df.groupby(['cik','date']).mean()	
	#all_scores = all_scores.to_frame('w').reset_index()
	all_scores = all_scores.reset_index()
	all_scores = all_scores.sort(columns=['date'])

	return all_scores

def score_lookup_table(all_scores, window=4):
	yrs = range(2010,2015)
	mos = range(1,13)
	dates = ['%d-%02d' % (yr,mo) for yr in yrs for mo in mos]	
	dicts_table = [dict() for d in dates]
	
	for index,row in all_scores.iterrows():										
		ix = dates.index('%d-%02d' % (row.date.year, row.date.month))			
		for i in range(1,window):						
			try:
				dicts_table[ix + i][row.cik] = row.w															
			except IndexError:
				break	

	exchange_lookup = pd.read_sql("SELECT cik,exchange FROM companies;", conn)
	exchange_lookup = dict(zip(exchange_lookup.cik.values, exchange_lookup.exchange.values))
	def unpack_and_sort_otc(d):
		l = ((cik,score) for (cik,score) in d.items() if exchange_lookup[cik]=='OTC')
		s = sorted(l, key=lambda (_,w): w, reverse=True)			
		return s
	def unpack_and_sort_exchange(d):
		l = ((cik,score) for (cik,score) in d.items() if exchange_lookup[cik] in ['N','A','NASDAQ'])
		s = sorted(l, key=lambda (_,w): w, reverse=True)			
		return s

	tuples_table_otc = map(unpack_and_sort_otc, dicts_table)	
	tuples_table_exchange = map(unpack_and_sort_exchange, dicts_table)	
	lookup_table = {'OTC':[], 'exchange':[]}
	lookup_table['OTC'] = dict(zip(dates, tuples_table_otc))
	lookup_table['exchange'] = dict(zip(dates, tuples_table_exchange))
	return lookup_table


def top_scorers(date, score_lookup, n=30, venue=['OTC','exchange']):
	
	if venue: 
		datestr = '%d-%02d' % (date.year, date.month)
		top = []
		for item in venue:
			l = score_lookup[item][datestr]
			top = top + l[:n]
		top = sorted(top, key=lambda (_,w): w, reverse=True)
		top = top[:n]	
		return zip(*top)
	else:
		return ([],[])

# def bottom_scorers(date, n=30, exchanges=['OTC','N','A','NASDAQ']):
# 	datestr = '%d-%02d' % (date.year, date.month)
# 	i = 0
# 	top = 
# 	while i < n:

# 	return zip(*score_lookup[datestr][-n:])	
	#in_range = all_scores[(all_scores.date > after) & (all_scores.date < before)] 
	# def max_date(g):
	# 	g = g.reset_index()
	# 	g = g.iloc[g['date'].argmax()]
	# 	g = g.drop('cik')
	# 	return g
	# #in_range = in_range.groupby('cik').apply(max_date).reset_index()
	# in_range = in_range.groupby('cik').mean().reset_index()
	#in_range = in_range.sort(columns=['w'],ascending=False)
	
	#return (in_range.iloc[:n].cik.values, in_range.iloc[:n].w.values)


if __name__ == '__main__':    		
	# pred_pos.train(50,'topquant', datestr='12 2013', exchanges=['NASDAQ','N','A','OTC'])
	# pred_neg.train(-0.05,'lt', datestr='12 2013', exchanges=['NASDAQ','N','A','OTC'])
	# pred_pos.evaluate(50,'topquant', after='01 2014', exchanges=['NASDAQ','N','A','OTC'])
	# pred_neg.evaluate(-0.05,'lt', after='01 2014', exchanges=['NASDAQ','N','A','OTC'])
	
	shorts_lookup = score_lookup_table(load_scores('negative_model.pickle'))
	longs_lookup = score_lookup_table(load_scores('model.pickle'))
	inception_date = pd.to_datetime('2010-09-01','%Y-%m-%d')
	portfolio = ptf.portfolioBacktester(inception_date)	
	portfolio.create_strategy('Suggested portfolio')
	portfolio.create_strategy('S&P 500')
	app.run('0.0.0.0',debug=True)
	conn.close()



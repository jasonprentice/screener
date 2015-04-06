from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import psycopg2
import numpy as np
from returnClassifier import returnPredictor


app = Flask(__name__)
conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")
pred_pos = returnPredictor(pre_pca=False)
pred_neg = returnPredictor(pre_pca=False)

@app.route('/')
def main_site():
	return render_template('list_by_exchange.html')

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


@app.route('/companies/search', methods=['POST'])
def lookup_by_ticker():		
	ticker = request.form['ticker']
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


if __name__ == '__main__':    	
	pred_pos.load_data(['2010','2011','2012','2013'])
	pred_neg.load_data(['2010','2011','2012','2013'])
	pred_pos.train(50,'topquant')
	pred_neg.train(-0.2,'lt')
	pred_pos.load_data(['2014'])
	pred_neg.load_data(['2014'])
	app.run('0.0.0.0',debug=True)
	conn.close()



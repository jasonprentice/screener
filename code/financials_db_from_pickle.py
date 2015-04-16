import pickle
import os
from sqlalchemy import create_engine
import psycopg2
from math import isnan

report_type = '10-Q'
base_dir = '../data/edgar/'
stock_price_pickle_file = 'stock_price.pickle'
financials_pickle = '../data/pickles/edgar_dataframe_filled_v2.pickle'

f = open(financials_pickle,'r')
edgar_df = pickle.load(f)
f.close()

new_fields = ['start_price', 'one_year_price','three_month_price','one_year_return','three_month_return']
for field in new_fields:
	edgar_df[field] = None

def add_prices(report_dir):
	components = report_dir.split('/')
	CIK = components[-3].strip()
	year = components[-2].strip()
	month = components[-1].strip()    

	f = open(report_dir + '/' + stock_price_pickle_file,'r')
	price_dict = pickle.load(f)
	f.close()

	for field in new_fields:
		edgar_df.loc[(CIK,year,month)][field] = price_dict[field]

reports = (parent_dir for (parent_dir,_,files) in os.walk(base_dir) if report_type + '.xml.gz' in files and stock_price_pickle_file in files)
map(add_prices, reports)    	
edgar_df['MarketCap'] = edgar_df['start_price'] * edgar_df['EntityCommonStockSharesOutstanding']
edgar_df = edgar_df.reset_index()
cols = edgar_df.columns
cols = map(lambda name: str(name), cols)
edgar_df.columns = cols

#engine = create_engine('postgresql://vagrant:pwd@localhost/secdata')
conn = psycopg2.connect('dbname=secdata user=vagrant password=pwd')
cur = conn.cursor()
cur.execute('DROP TABLE IF EXISTS financials;')
# Create table
sqlstring = 'CREATE TABLE financials ('
for col in edgar_df.columns:		
	if col in ['CIK' ,'year','month']:
		sqlstring = sqlstring + col + ' varchar, '	
	else:
		sqlstring = sqlstring + col + ' numeric, '
sqlstring = sqlstring[:-2] + ');'
cur.execute(sqlstring)
# Fill rows

def pad_str(s, num_chars):	
	s_formatted = ['0']*num_chars
	s_formatted[-len(s):] = s
	return ''.join(s_formatted)

for row in edgar_df.values:
	sqlstring = 'INSERT INTO financials VALUES ('
	sqlstring = sqlstring + '\'' + pad_str(row[0],10) + '\','		# CIK
	sqlstring = sqlstring + '\'' + pad_str(row[1],4) + '\','			# Year
	sqlstring = sqlstring + '\'' + pad_str(row[2],2) + '\','			# Month
	for item in row[3:]:				
		if item and str(item) != 'nan':
			sqlstring = sqlstring + str(item) + ','
		else:
			sqlstring = sqlstring + 'NULL,'			
	sqlstring = sqlstring[:-1] + ');'	
	cur.execute(sqlstring)
conn.commit()
cur.close()
#edgar_df.to_sql('financials', engine, if_exists='replace', index=False)
conn.close()

                              
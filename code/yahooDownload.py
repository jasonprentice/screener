# Walk ../data/edgar, and fetch stock price information for each listed company, for every statement time period

from retrying import retry
import requests


def retry_if_io_error(exception):
    """Return True if we should retry (in this case when it's an IOError), False otherwise"""
    return isinstance(exception, IOError)

#@retry(wait_fixed=2000, retry_on_exception=retry_if_io_error)
@retry(wait_exponential_multiplier=1000, wait_exponential_max=120000)
def fetchFromYahoo(ticker, mo, yr):
    print 'Requesting ' + ticker + ': ' + str(yr) + ' ' + str(mo) + '...'

    # dur = duration in months
    mo_final = mo
    yr_final = yr + 1

    params = {'s': ticker,
              'a': mo-1,
              'b': 1,
              'c': yr,
              'd': mo_final-1,
              'e': 1,
              'f': yr_final,
              'g': 'm',
              'ignore': '.csv'}

    csv_data = requests.get('http://ichart.finance.yahoo.com/table.csv', params=params)
    status = csv_data.status_code
    if status == requests.codes.ok:
        rows = csv_data.text.split('\n')    
        num_months = len(rows)-1    
        start_pr = float(rows[-2].split(',')[-1])        
        one_year_pr = float(rows[1].split(',')[-1])
        if len(rows) > 5:
            three_mo_pr = float(rows[-5].split(',')[-1])
        else:
            three_mo_pr = one_year_pr
            
        
        return (start_pr, three_mo_pr, one_year_pr)
    elif status == 404:
        return None
    else:
        raise IOError('Yahoo returned error code ' + str(csv_data.status_code))
    

if __name__ == '__main__':
#    fetchFromYahoo('GOOGL',7,2012)
    import os
    import psycopg2
    import pandas as pd
    import pickle
    from multiprocessing import Pool

    conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")
    cur = conn.cursor()
    #    cur.execute("CREATE TABLE IF NOT EXISTS financials (cik varchar, year integer, start_price numeric, end_price numeric, return numeric);")

    report_type = '10-Q'
    basedir = '../data/edgar/'
    def ticker_lookup(report_dir):
        components = report_dir.split('/')
        CIK = components[-3].strip()
        year = components[-2].strip()
        month = components[-1].strip()        
        cur.execute("SELECT ticker FROM companies WHERE cik = %s;", (CIK,))
        item = cur.fetchone()
        if item != None:
            ticker = item[0]
        else:
            ticker = None

        return (CIK, year, month, ticker)
        

    def increment_month(mo, yr, dur):
        # Increment (mo,yr) by dur months
        mo = int(mo)
        yr = int(yr)
        dur = int(dur)
        mo_final = ((mo-1) + dur) % 12 + 1
        yr_final = yr + ((mo-1)+dur)/12
        return (mo_final, yr_final)

    def fetch_prices(tup):
        (CIK,year,month,ticker) = tup        
        if ticker:

            (start_month, start_yr) = increment_month(month,year,1)
            fetch = fetchFromYahoo(ticker, start_month, start_yr)
            if fetch != None:
                (start_pr, three_mo_pr, one_year_pr) = fetch
                if start_pr > 0:
                    one_year_return = (one_year_pr - start_pr) / start_pr
                    three_month_return = 0.25 * (three_mo_pr - start_pr) / start_pr 
                else:
                    one_year_return = None
                    three_month_return = None

                price_dict = {'ticker': ticker,
                              'start_price': start_pr,
                              'one_year_price': one_year_pr, 
                              'three_month_price': three_mo_pr, 
                              'one_year_return': one_year_return, 
                              'three_month_return': three_month_return}
                if one_year_return: 
                    print ticker + ' (' + CIK + ') ' + year + ' ' + month + ': 1-yr return = %0.2f' % one_year_return
                else:
                    print ticker + ' (' + CIK + ') ' + year + ' ' + month + ': 1-yr return = NaN'

                f = open(basedir + CIK + '/' + year + '/' + month + '/stock_price.pickle','w')
                pickle.dump(price_dict,f)
                f.close()
            else:
                print 'Failed to access ' + ticker + ' ' + year + ' ' + month + ' from Yahoo.'
        else:
            print 'No ticker found for CIK ' + CIK            

    print 'Walking directory structure and resolving ticker symbols...'
    reports = (parent_dir for (parent_dir,_,files) in os.walk(basedir) if report_type + '.xml.gz' in files and 'stock_price.pickle' not in files)
    statements_to_process = map(ticker_lookup, reports)
    print 'Will process ' + str(len(statements_to_process)) + ' statements.'
    # index = pd.MultiIndex.from_tuples([(CIK,year,month) for (CIK,year,month,_) in statements_to_process], names = ['CIK','year','month'])
    # df = pd.DataFrame(index = index, columns = ['start_price', 'one_year_price', 'three_month_price', 'one_year_return', 'three_month_return'])
    
    workers = Pool(10)
    print 'Downloading prices.'
    workers.map(fetch_prices, statements_to_process)
    #map(fetch_prices, statements_to_process)
    cur.close()
    conn.close()

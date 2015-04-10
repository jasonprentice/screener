from retrying import retry
import requests
import pandas as pd
import sys

def retry_if_io_error(exception):
    """Return True if we should retry (in this case when it's an IOError), False otherwise"""
    return isinstance(exception, IOError)


def parse_csv(text):
    rows = text.split('\n')    
    num_months = len(rows)-1        
    def parse_row(row):
        items = row.split(',')
        return (items[0], items[-1])
    parsed = map(parse_row, rows[1:])    

    dates = [d for (d,_) in parsed]
    pr = [p for (_,p) in parsed]
    return pd.Series(pr, index=pd.to_datetime(dates))
    

#@retry(wait_fixed=2000, retry_on_exception=retry_if_io_error)
@retry(wait_exponential_multiplier=1000, wait_exponential_max=120000)
def fetchFromYahoo(ticker):
    print 'Requesting ' + ticker + '...'
    sys.stdout.flush()

    params = {'s': ticker,
              'a': 0,
              'b': 1,
              'c': 2009,
              'd': 11,
              'e': 31,
              'f': 2015,
              'g': 'd',
              'ignore': '.csv'}

    csv_data = requests.get('http://ichart.finance.yahoo.com/table.csv', params=params)
    status = csv_data.status_code
    if status == requests.codes.ok:
        return parse_csv(csv_data.text)        
    elif status == 404:
        return pd.Series()
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

    def ticker_lookup(CIK):    
        cur.execute("SELECT ticker FROM companies WHERE cik = %s;", (CIK,))
        item = cur.fetchone()
        if item != None:
            ticker = item[0]
        else:
            ticker = None

        return (CIK, ticker)
            

    def fetch_prices(tup):
        (CIK, ticker) = tup        
        if ticker:            
            prices = fetchFromYahoo(ticker)            
            if not prices.empty:                
                f = open(basedir + CIK + '/all_daily_prices.pickle','w')
                pickle.dump(prices,f)
                f.close()
            else:
                print 'Failed to access ' + ticker + ' from Yahoo.'
        else:
            print 'No ticker found for CIK ' + CIK       


    print 'Walking directory structure and resolving ticker symbols...'
    basedir = '../data/edgar/'
    reports = (dirname for dirname in os.listdir(basedir) if dirname[0] != '.' and not os.path.isfile(basedir + dirname + '/all_daily_prices.pickle'))
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
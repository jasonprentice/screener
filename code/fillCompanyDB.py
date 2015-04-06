# Crawl ../data/edgar/, parse XBRL files to extract financial quantities, and save to edgar_dataframe.pickle.

import psycopg2
from xbrl import XBRL
import gzip
import os
import pandas as pd
import parse_xbrl
import pickle

parser = parse_xbrl.parseXBRL()

report_type = '10-Q'
base_dir = "../data/edgar/"



# conn = psycopg2.connect("dbname=SECData user=postgres password=pwd")
# cur = conn.cursor()
#for item in items:
#    cur.execute("ALTER TABLE financials ADD COLUMN " + item + " numeric;")
#conn.commit()

def crawl_dirs():
# Traverses CIK directory structure and calls func inside each month folder
    
    dirs = os.listdir(base_dir)
    startCIK = '0000001750'
    start_downloading = True
    reports = []
    for CIK in dirs:
        for CIK in dirs:
            if CIK[0] != '.' and start_downloading:  
                yr_dirs = [x for x in os.listdir(base_dir + CIK) if os.path.isdir(os.path.join(base_dir+CIK, x))]          
                #co_name = open(base_dir + CIK + '/company_name.txt').read()
                for yr in yr_dirs:
                    if yr[0] != '.':
                        mo_dirs = os.listdir(base_dir + CIK + "/" + yr)
                        for mo in mo_dirs:
                            if mo[0] != '.':
                                report_file = base_dir + CIK + "/" + yr + "/" + mo + "/" + report_type + ".xml.gz"
                                if os.path.isfile(report_file):
                                    reports.append(Report(CIK=CIK, year=yr, month=mo, file=report_file, name=''))                                   

            elif CIK == startCIK:
                start_downloading = True
    return reports


def split_dir(report_dir):
    components = report_dir.split('/')
    CIK = components[-3].strip()
    year = components[-2].strip()
    month = components[-1].strip()
    return (CIK, year, month)

def insert_row(num_fails, tup):
    (CIK,year,month) = tup
    success = False
    
    try:
        item_vals = parser.read_statement(CIK, year, month, report_type)        
        success = True
    except:        
        print "Failed to parse " + CIK + " " + year + " " + month
        num_fails += 1
    if success:
        print CIK + " " + year + " " + month + ": " + str(item_vals['EntityCommonStockSharesOutstanding']) + " shares outstanding."
        df.loc[(CIK, year, month)] = pd.Series(item_vals)
    return num_fails
            #cur.execute("UPDATE financials SET " + item + " = %s WHERE cik=%s AND year=%s;", (value,CIK,int(yr)))
        #conn.commit()
    # if CIK == '0001490490' and year == '2014' and month == '09':
    #     item_vals = parser.read_statement(CIK, year, month, report_type)    
    #     print item_vals    
        

reports = (parent_dir for (parent_dir,_,files) in os.walk(base_dir) if report_type + '.xml.gz' in files)
statements_to_process = map(split_dir, reports)
print str(len(statements_to_process)) + ' Statements.'
index = pd.MultiIndex.from_tuples(statements_to_process, names = ['CIK','year','month'])

df = pd.DataFrame(index = index, columns = parser.all_items + ['EntityCommonStockSharesOutstanding'])
num_fails = reduce(insert_row, statements_to_process, 0)
pfile = open('edgar_dataframe.pickle','w')
pickle.dump(df,pfile)
pfile.close()
print num_fails
# pfile = fopen('edgar_dataframe.pickle')
# pickle.dump(df, pfile)
#df = df.reset_index()


# cur.close()
# conn.close()

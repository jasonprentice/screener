# Create database of company names, ticker symbols, and exchange
# reads from ../data/nasdaqlisted.txt, ../data/otherlisted.txt, ../data/otclist.txt


import psycopg2

conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS companies;")
cur.execute("CREATE TABLE companies (ticker varchar, name varchar, ETF char, exchange varchar);")

cur.execute("CREATE TABLE temp (Symbol varchar, SecurityName varchar, MarketCategory varchar, TestIssue char, FinancialStatus char, RoundLotSize varchar);")

basedir = '../data/'
f = open(basedir+"nasdaqlisted.txt")
f.readline()
cur.copy_from(f, "temp", sep="|")
cur.execute("CREATE TABLE two_cols (exchange varchar, ETF char);")
cur.execute("INSERT INTO two_cols VALUES ('NASDAQ','N');")
cur.execute("INSERT INTO companies (ticker,name,etf,exchange) SELECT Symbol, SecurityName, ETF, exchange FROM (temp CROSS JOIN two_cols);")
cur.execute("DROP TABLE two_cols;")
cur.execute("DROP TABLE temp;")
f.close()


cur.execute("CREATE TABLE temp (Symbol varchar, SecurityName varchar, Exchange char, CAQSymbol varchar, ETF char, RoundLotSize varchar, TestIssue char, NASDAQSymbol varchar);")

f = open(basedir+"otherlisted.txt")
f.readline()
cur.copy_from(f, "temp", sep="|");
cur.execute("INSERT INTO companies (ticker,name,etf,exchange) SELECT Symbol, SecurityName, ETF, Exchange FROM temp;")
cur.execute("DROP TABLE temp;")
f.close()

cur.execute("CREATE TABLE temp (Symbol varchar, SecurityName varchar, MarketCategory varchar, Status varchar, TestIssue char);")
f = open(basedir+"otclist.txt")
f.readline()
cur.copy_from(f, "temp", sep="|");
cur.execute("CREATE TABLE two_cols (exchange varchar, ETF char);")
cur.execute("INSERT INTO two_cols VALUES ('OTC','N');")
cur.execute("INSERT INTO companies (ticker,name,etf,exchange) SELECT Symbol, SecurityName, ETF, exchange FROM (temp CROSS JOIN two_cols);")
cur.execute("DROP TABLE two_cols;")
cur.execute("DROP TABLE temp;")
f.close()


conn.commit()

cur.close()
conn.close()



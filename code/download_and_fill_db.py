# List of steps to download reports from Edgar, stock prices from Yahoo, and populate databases.

execfile('SECDownload.py')					# Download XBRL reports from Edgar
execfile('fillCompanyDB.py')				# Parse XBRL reports and extract financials

execfile('makeTickerDB.py')					# Create DB table 'companies', listing company names, ticker symbols, and exchange (from NASDAQ listings)
execfile('tickerLookup.py')					# Match company names between Edgar and NASDAQ, then add CIK column to 'companies' (the unique identifier used by Edgar)

execfile('yahooDownload.py')				# Download stock prices corresponding to period following each statement
execfile('financials_db_from_pickle.py') 	# Combine financials and stock prices into DB table 'financials'

execfile('fillNotesDB.py')					# Extract footnotes from XBRL reports, store in new DB table 'notes'

execfile('yahooDownloadFull.py')			# Download all daily stock prices


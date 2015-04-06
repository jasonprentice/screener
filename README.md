Use machine learning to automatically select stocks that are likely to perform well or poorly in the future, based on publicly available filings with the SEC. 

code/download_and_fill_db.py will execute the steps to acquire the data, parse it, and generate the necessary databases.
code/returnClassifier.py contains the core screener engine.
site/screener_site.py is a Flask app that puts everything together into a nice web interface.

def tickerLookup( name, conn ):
    import psycopg2

    cur = conn.cursor()

    cur.execute("SELECT ticker, name FROM companies;")
    #    conn.commit()
    
    name = normalizeStr(name)
    words = name.split()
    matches = []
    counts = []
    matches_name = []
    for row in cur:
        this_name = normalizeStr(row[1])
        this_words = this_name.split()
        num_match = 0
        for word in words:
            if word in this_words: #and word not in ["company","corporation","incorporated"]:
                num_match = num_match + 1
    
        if num_match == len(words) and ("common" in this_words or "ordinary" in this_words):
            counts = counts + [num_match]
            matches = matches + [row[0]]
            matches_name = matches_name + [row[1]]
    cur.close()

    if len(counts) == 0:
        return ("","")
    
    M = max(counts)
    
    match_ix = [i for i,x in enumerate(counts) if x==M]
    matches = [x for i,x in enumerate(matches) if i in match_ix]
    matches_name = [x for i,x in enumerate(matches_name) if i in match_ix]
    
    if len(matches) >= 1:
        return (matches[0], matches_name[0])
    elif len(matches) == 0:
        return ("","")
#    else:
#        lengths = map(len, matches)
#        ix = lengths.index(min(lengths))
#        return (matches[ix], matches_name[ix])



def normalizeStr( name ):
    import string
    import re
    

    name = name + " "
    name = name.lower()

    name = re.sub( r'/.*/', " ", name)
    name = re.sub( r'\\\\.*\\\\', " ", name)
    name = name.replace("-", " ")
    name = name.replace("/", " ")
    # strip punctuation
    name = name.translate(string.maketrans("",""), string.punctuation)
    # expand abbreviations
    name = name.replace(" co "," company ")
    name = name.replace(" corp ", " corporation ")
    name = name.replace(" inc ", " incorporated ")
    name = name.replace(" ltd ", " limited ")

#    name.replace(" co ", " ")
#    name.replace(" corp ", " ")
#    name.replace(" inc ", " ")
#    name.replace(" company ", " ")
#    name.replace(" corporation ", " ")
#    name.replace(" incorporated ", " ")
    return name

if __name__ == '__main__':
    import os
    import psycopg2
    
    basedir = "../data/edgar/"
    conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")
    cur = conn.cursor()
    cur.execute("ALTER TABLE companies ADD COLUMN cik varchar;")
    dirs = os.listdir(basedir)
    numMatch = 0
    numNonmatch = 0
    for CIK in dirs:
        if CIK[0] != '.':
            f = open(basedir + CIK + "/company_name.txt")
            lookupName = f.read()
            f.close()
            #        lookupName = "AMERICAN AIRLINES INC"
            (ticker, name) = tickerLookup(lookupName, conn)
            if ticker == "":
#                print lookupName
                numNonmatch = numNonmatch + 1
            else:
                cur.execute("UPDATE companies SET cik = %s WHERE ticker = %s;", (CIK, ticker))
                print ticker + ":   " + name + "      (" + lookupName + "; " + CIK + ")"
                numMatch = numMatch + 1
    conn.commit()
    cur.close()
    conn.close()

#    print str(numMatch) + " matched, " + str(numNonmatch) + " not matched."


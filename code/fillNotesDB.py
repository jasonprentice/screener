import psycopg2
import parse_xbrl
import os

conn = psycopg2.connect("dbname=secdata user=vagrant password=pwd")

def make_table():
	cur = conn.cursor()
	cur.execute("DROP TABLE IF EXISTS notes;")
	cur.execute("CREATE TABLE notes (CIK varchar, year varchar, month varchar, note_tag varchar, note_text varchar, note_wordcount numeric);")
	conn.commit()
	cur.close()

parser = parse_xbrl.parseXBRL()

report_type = '10-Q'
base_dir = "../data/edgar/"

def insert_row(num_fails, report_dir):
	components = report_dir.split('/')
	CIK = components[-3].strip()
	year = components[-2].strip()
	month = components[-1].strip()

	success = False    	

	try:
		notes = parser.get_notes(CIK, year, month, report_type)        
		success = True
	except:        
		print "Failed to parse " + CIK + " " + year + " " + month
		num_fails += 1
	if success:
		print CIK + " " + year + " " + month + " "
		cur = conn.cursor()
		for note in notes:			
			cur.execute("INSERT INTO notes (CIK,year,month,note_tag,note_text,note_wordcount) VALUES (%s,%s,%s,%s,%s,%s);", (CIK,year,month,note[0],note[1],note[2]))
		conn.commit()
		cur.close()
	return num_fails

make_table()
reports = (parent_dir for (parent_dir,_,files) in os.walk(base_dir) if report_type + '.xml.gz' in files)
num_fails = reduce(insert_row, reports, 0)
print str(num_fails) + " Failures."

conn.close()

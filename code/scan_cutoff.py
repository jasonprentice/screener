from returnClassifier import returnPredictor as rp
import csv
from numpy import linspace

#cutoff = -0.2
ntrials = 10
dir = 'gt'

with open('performance_vs_cutoff.csv','wb') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['Cutoff','Return','AUC'])
	cutoffs = linspace(-0.8,1,20)
	for trial in range(ntrials):
		for cutoff in cutoffs:
			x=rp()
			x.load_data([2010,2011,2012])
			x.train(cutoff,dir)
			#x.print_trees()

			x.load_data([2013])
			(ret,auc) = x.evaluate(cutoff,dir,plot=False)
			print str(cutoff) + ': Return = ' + str(ret) + '; AUC = ' + str(auc)
			writer.writerow([cutoff, ret, auc]) 



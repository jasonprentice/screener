from returnClassifier import returnPredictor 

cutoff = 0
dir = 'gt'

pred = returnPredictor(pre_pca = False)

pred.load_data([2010,2011,2012])

pred.train(cutoff,dir)

pred.load_data([2013])

pred.evaluate(cutoff,dir,plot=True)



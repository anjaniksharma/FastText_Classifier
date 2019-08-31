
from sklearn.base import BaseEstimator, ClassifierMixin
import fastText as ft
import pandas as pd
from sklearn.metrics import classification_report

class FastTextClassifier(BaseEstimator,ClassifierMixin):
	"""Base classiifer of Fasttext estimator"""

	def __init__(self, lr=0.01, dim=100, ws=5, epoch=5, minCount=1, minCountLabel=0, minn=0, maxn=0, neg=5, wordNgrams=1, loss="softmax", bucket=2000000, thread=12, lrUpdateRate=100, t=0.0001, label="__label__",verbose=2, pretrainedVectors="", output="model", train_filename=None):
        	"""
        	label_prefix   			label prefix ['__label__']
        	lr             			learning rate [0.1]
        	r_update_rate 			change the rate of updates for the learning rate [100]
        	dim            			size of word vectors [100]
        	ws             			size of the context window [5]
        	epoch          			number of epochs [5]
        	minCount      			minimal number of word occurences [1]
        	neg            			number of negatives sampled [5]
        	wordNgrams    			max length of word ngram [1]
        	loss           			loss function {ns, hs, softmax} [softmax]
        	bucket         			number of buckets [0]
        	minn           			min length of char ngram [0]
        	maxn           			min length of char ngram [0]
        	todo : Recheck need of some of the variables, present in default classifier
		"""
		self.lr=lr
		self.dim=dim
		self.ws=ws
		self.epoch=epoch
		self.minCount=minCount
		self.minCountLabel=minCountLabel
		self.minn=minn
		self.maxn=maxn
		self.neg=neg
		self.wordNgrams=wordNgrams
		self.loss=loss
		self.bucket=bucket
		self.thread=thread
		self.lrUpdateRate=lrUpdateRate
		self.t=t
		self.label=label
		self.verbose=verbose
		self.pretrainedVectors=pretrainedVectors

		self.classifier=None
		#self.result=None
		self.output=output
		self.n_features=None
		self.classes_=None
		self.train_filename = train_filename
		#self.save_model=None


	def fit(self,X, y=None):
		'''
		Input: takes input file in format
		returns classifier object
		to do: add option to feed list of X and Y or file
		'''
		self.classifier = ft.train_supervised(X, lr = self.lr,  dim=self.dim, ws=self.ws, minn=self.minn, maxn=self.maxn, neg=self.neg, minCount=self.minCount, wordNgrams=self.wordNgrams, bucket=self.bucket, thread=self.thread, epoch=self.epoch, loss=self.loss, lrUpdateRate=self.lrUpdateRate, t=self.t, verbose=self.verbose, label=self.label, pretrainedVectors = self.pretrainedVectors)
		self.n_features = len(self.classifier.get_words())
		self.classes_ = [ class_.replace(self.label,"") for class_ in self.classifier.get_labels()]
		#self.save_model = self.classifier.save_model
		return(None)
           
	#def predict(self,test_file,csvflag=True,k_best=1):
	def predict(self,X):
		'''
		Input: Takes input test finle in format
		return results object
		to do: add unit tests using sentiment analysis dataset 
		to do: Add K best labels options for csvflag = False 

		'''
		try: result=self.classifier.predict(X,k=1)
		except: 
			print "failed"
			return None
		predict = [res[0].replace(self.label,"") for res in result[0]]
		predict_proba = [float(res[0]) for res in result[1]]
		return predict, predict_proba
               

	def predict_proba(self,X):
		'''
		Input: List of sentences
		return reort of classification
		to do: check output of classifier predct_proba add label option and unit testing
		'''
		try: result=self.classifier.predict(X,k=1)
                except:
                        print "failed"
                        return None
                predict = [res[0].replace(self.label,"") for res in result[0]]
                predict_proba = [float(res[0]) for res in result[1]]
                return predict, predict_proba



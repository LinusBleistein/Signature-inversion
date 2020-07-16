from sktime.utils.load_data import load_from_arff_to_dataframe
import numpy as np


def get_pendigits():
	name="Pendigits"
	data,y=load_from_arff_to_dataframe(
			'data/{0}/{0}_TRAIN.arff'.format(name))
	X=np.zeros((data.shape[0],len(data.iloc[0,0]),data.shape[1]))
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			X[i,:,j]=data.iloc[i,j]
		#X[i,:,data.shape[1]]=np.linspace(0,1,num=len(data.iloc[0,0]))
	return(X,y)
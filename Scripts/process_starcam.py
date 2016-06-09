import numpy as np
import pandas as pd
from scipy.stats import mode

filename = 'data/data_Exp5.csv'
row_select = ['total_match','success_rate','exposure','hor_attitude_az','hor_attitude_el',
	'hor_attitude_roll','eq_point_error','sigma_roll','time_wall']

def process_StarCam_Run(filename,exposure):
	exposure = np.float(exposure)
	data = pd.read_csv(filename,delimiter='\t')
	data = data[data['total_match']>=6]
	meandata = np.mean(data,axis=0)
	stddata = np.std(data,axis=0)
	modedata = mode(data,axis=0)
	series = {'Mean':meandata,'Std':stddata,'Mode':modedata.mode[0]}
	df = pd.DataFrame(series)
	final_df = df.loc[row_select]
	print final_df
	#print data.sort('eq_attitude_ra')
	success_rate = np.float(len(data))/np.float(max(data.index))*100.
	print "Solution success rate = ",len(data),"/",max(data.index),"=",success_rate,"%"
	print "Exposure time error = ",np.abs((final_df['Mean'].loc['exposure'] - exposure))/exposure*100.,"%"
	return final_df,success_rate,len(data)




filename = 'data/data_Exp1.csv'
process_StarCam_Run(filename,exposure=250)
filename = 'data/data_Exp2.csv'
process_StarCam_Run(filename,exposure=125)
filename = 'data/data_Exp3.csv'
process_StarCam_Run(filename,exposure=62)
filename = 'data/data_Exp4.csv'
process_StarCam_Run(filename,exposure=31)

filename = 'data/data_Exp5.csv'
process_StarCam_Run(filename,exposure=62)

filelist = ['data/data_Exp1.csv','data/data_Exp2.csv','data/data_Exp3.csv','data/data_Exp5.csv','data/data_Exp4.csv']
exposure_list = [250,125,62,62,31]
name_list = ['Exp1','Exp2','Exp3','Exp4','Exp5']
def process_all_Starcam_Runs(filelist,exposure_list,name_list):
	L = len(filelist)
	result = pd.DataFrame(columns=('Exposure time','Number of images in run',
		'Fitted exposure time','Std fitted exposure time',
		'Number of matching stars','Std number of matching stars',
		'Fit ra \& dec error (arcsec)','Std fit ra \& dec error (arcsec)',
		'Fit roll error (arcsec)','Std fit roll error (arcsec)',
		'Processing time','Std processing time',
		'Solution success rate'))
		
	for i in range(L):
		df,success_rate,Nimgs = process_StarCam_Run(filelist[i],exposure_list[i])
		result.loc[name_list[i]] = [exposure_list[i],Nimgs,
			df['Mean'].loc['exposure'],df['Std'].loc['exposure'],
			df['Mean'].loc['total_match'],df['Std'].loc['total_match'],
			df['Mean'].loc['eq_point_error'],df['Std'].loc['eq_point_error'],
			df['Mean'].loc['sigma_roll']/4.848e-6,df['Std'].loc['sigma_roll']/4.848e-6,
			df['Mean'].loc['time_wall'],df['Std'].loc['time_wall'],
			success_rate]
	print result
	return result
	
data = process_all_Starcam_Runs(filelist,exposure_list,name_list)
data.to_csv('Starcam_summary.csv')
print data.to_latex()


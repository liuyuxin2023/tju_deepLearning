import pandas as pd
import matplotlib.pyplot as plt
import jqdatasdk

jqdatasdk.auth("18826072334", "Ttt246810")
'''
future_code=['ZN.SHF','CU11.SHF','CU00.SHF','CU03.SHF','CU09.SHF','IC1505.CCFX', \
	'CU10.SHF','HC.SHF','HC02.SHF','CU_S.SHF','J02.DCE','J_S.DCE','J.DCE']
'''
#future_code=['IC1505.CCFX']
df1 = pd.read_csv('future_code.csv')
future_code=df1.loc[:,'future_code']
start_time='2000-02-06 14:00:00'
end_time='2019-04-09 12:00:00'
frequent='60m'
df_fin=pd.DataFrame()
def lookup(i):
	if i>=len(future_code):
		print('searching finished...')
		return
	try: 
		df=jqdatasdk.get_price(future_code[i], start_date=start_time, end_date=end_time, skip_paused=True,frequency=frequent)
	except Exception:
		print('I cannot find the ',future_code[i])
	else: 
		print('find ',future_code[i])
		print('the size: ',df.shape)
		name=str(future_code[i])+'.csv'
		df.to_csv(name)
		
	finally: 
		i=i+1
		lookup(i)

lookup(0)
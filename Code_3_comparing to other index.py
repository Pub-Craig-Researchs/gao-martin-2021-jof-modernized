import numpy as np
import pandas as pd
import datetime as dt


def Moving_Block_BT(inputdata,U,L):
    #block size
    T = len(inputdata)
    m = int(T**(1/3))
    B = 10000
    temp_cov = np.zeros([B,1])
    for k in range(B):
        u = np.ceil((T-m)*np.random.random_sample([int(np.ceil(T/m)),1]))
        u = u+np.arange(0,m)
        u = np.ndarray.flatten(u).astype(int)
        temp_cov[k] = np.corrcoef(inputdata[u].astype('float64').T)[0][1]
    cov_sample = np.ndarray.flatten(temp_cov)    
    cov_mean = np.corrcoef(inputdata.astype('float64').T)[0][1]
    if cov_mean >0:
        p_value = sum(1*(cov_sample<0))/len(cov_sample)
    else:    
        p_value = sum(1*(cov_sample>0))/len(cov_sample)
    return(cov_mean,p_value,np.quantile(cov_sample,L),np.quantile(cov_sample,U))    
        
def IID_BT(inputdata,U,L):
    #block size
#    inputdata = np.array(DM[['Bt','DM']])
    T = len(inputdata)    
    B = 10000
    temp_cov = np.zeros([B,1])
    for k in range(B):
        u = np.floor(T*np.random.random_sample([T,1]))
        u = np.ndarray.flatten(u).astype(int)
        temp_cov[k] = np.corrcoef(inputdata[u].astype('float64').T)[0][1]
    cov_sample = np.ndarray.flatten(temp_cov)    
    cov_mean = np.corrcoef(inputdata.astype('float64').T)[0][1]
    if cov_mean >0:
        p_value = sum(1*(cov_sample<0))/len(cov_sample)
    else:    
        p_value = sum(1*(cov_sample>0))/len(cov_sample)
    return(cov_mean,p_value,np.quantile(cov_sample,L),np.quantile(cov_sample,U))    


def Merge_data(tab, date_field, dateformat, Bt_tab, BtAR3_tab):     
    if date_field == 'yyyymm':    
        tab_mthly = pd.merge(tab, Bt_tab, on = 'yyyymm', how = 'inner')
        tab_mthly = pd.merge(tab_mthly, BtAR3_tab, on = 'yyyymm', how = 'inner')        
    else:    
        tab['yyyymm'] = tab[date_field].apply(lambda x: dt.datetime.strptime(x,dateformat).year*100 + dt.datetime.strptime(x,dateformat).month)
        tab_mthly = tab.groupby(['yyyymm'])[date_field].max().reset_index() 
        tab = tab.drop(['yyyymm'], axis = 1)
        tab_mthly = pd.merge(tab_mthly, tab, on = date_field, how = 'left')    
        tab_mthly = pd.merge(tab_mthly, Bt_tab, on = 'yyyymm', how = 'inner')
        tab_mthly = pd.merge(tab_mthly, BtAR3_tab, on = 'yyyymm', how = 'inner')
    return(tab_mthly)
#------------------------------------------------------------------------------------------------------------------------
# compute bubble indicator that we need to use to compute the correlations
    
import os # this is to set filepath to currrent folder 
filepath_data = os.path.dirname(os.path.abspath(__file__))
filepath_compute = os.path.dirname(os.path.abspath(__file__))
filepath_CRSP = os.path.dirname(os.path.abspath(__file__))
filepath_figure = os.path.dirname(os.path.abspath(__file__))
filepath_table = os.path.dirname(os.path.abspath(__file__))

#------------------------------------------------------------------------------------
# Assemble results to construct Bt
# Take LVIX data
lvix = pd.read_csv(filepath_compute +'/Pseudo_results/LVIX9619.csv')

LVIX_SPX500 = lvix[lvix['secid'] == 108105][['date','12mo']]

rg_12mo_y_mthly = pd.read_excel(filepath_CRSP+'/Pseudo_results/Month_forecast_rg/CRSP 12mo (risk-free)_194612_AR1y.xlsx')
rg_12mo_y_mthly_AR3 = pd.read_excel(filepath_CRSP+'/Pseudo_results/Month_forecast_rg/CRSP 12mo (risk-free)_194612_AR3y.xlsx')        

# rf data
yieldcurve = pd.read_csv(filepath_data+'/Pseudo_Data/Monthly_yield_9619.csv')
xvals = np.array([180, 360, 720])

#end_of_month = np.array(LVIX_SPX500['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d')))
end_of_month = np.array(LVIX_SPX500['date'])
Rf_mat = np.empty([end_of_month.size,3])

for k in range(end_of_month.size):
    x = np.array(yieldcurve[yieldcurve['date'] == end_of_month[k]]['days'])
    y = np.log(np.array(yieldcurve[yieldcurve['date']== end_of_month[k]]['rate'])*0.01+1)*x/360
    Rf = np.interp(xvals, x, y)
    Rf_mat[k] = Rf

Rf_table = pd.DataFrame(np.vstack((end_of_month,Rf_mat.T)).T)
Rf_table.columns = ['date','6mo','12mo','24mo']    
        
# need to truncate rg data to match Rf and LVIX
Bt = pd.DataFrame(np.vstack((np.array(rg_12mo_y_mthly.date)[:-48],-1*np.array(rg_12mo_y_mthly.forecast_AR1)[:-48]+\
                np.array(LVIX_SPX500['12mo'])+np.array(Rf_table['12mo']))).T)
Bt.columns = ['yyyymm','Bt']        
        
      
BtfullAR3 = pd.DataFrame(np.vstack((np.array(rg_12mo_y_mthly_AR3.date)[:-48],-1*np.array(rg_12mo_y_mthly_AR3.full_sample)[:-48]+\
                np.array(LVIX_SPX500['12mo'])+np.array(Rf_table['12mo']))).T)
BtfullAR3.columns = ['yyyymm','BtAR3']        

#----------------------------------------------------------------------------------------------------------------
# import the data of other indexes

NFCI = pd.read_csv(filepath_data +'/Pseudo_Data/NFCI.csv')
ANFCI = pd.read_csv(filepath_data +'/Pseudo_Data/ANFCI.csv')
EBP = pd.read_csv(filepath_data +'/Pseudo_Data/ebp_csv.csv')


NFCI_mthly = Merge_data(NFCI,'DATE','%Y-%m-%d',Bt,BtfullAR3)
ANFCI_mthly = Merge_data(ANFCI,'DATE','%Y-%m-%d',Bt,BtfullAR3)
EBP_mthly =  Merge_data(EBP,'date','%d/%m/%Y',Bt,BtfullAR3)

writer = pd.ExcelWriter(filepath_figure + '/Pseudo_results/Otherindex.xlsx', engine='xlsxwriter')

NFCI_mthly.to_excel(writer, sheet_name='NFCI')
ANFCI_mthly.to_excel(writer, sheet_name='ANFCI')
EBP_mthly.to_excel(writer, sheet_name='EBP')
writer.close()

#-------------------------------------------------------------------------------------------------------------------
# construct bootstrap tables

tables = {'EBP': EBP_mthly,\
          'ANFCI': ANFCI_mthly,\
          'NFCI': NFCI_mthly\
          }

col_names = {'GM': ['Bt', 'BtAR3'],\
             'EBP': ['ebp'],\
             'ANFCI': ['ANFCI'],\
             'NFCI': ['NFCI']}

index_names = ['EBP', 'ANFCI', 'NFCI']
index_names_mthly = ['EBP', 'ANFCI', 'NFCI']

# individual tabel using two bootstrap method
for b_idx in col_names['GM']:    
    writer = pd.ExcelWriter(filepath_figure +'/Pseudo_results/'+b_idx+'_otherindex_corr_mth.xlsx', engine='xlsxwriter')      
    for index in index_names_mthly:
        temp_table = tables[index]
        for col in col_names[index]:
            temp_sheet = np.zeros([73,5])
            for k in range(73):
                lag = k-36
                input_table = pd.DataFrame()
                input_table[b_idx] = temp_table[b_idx].shift(-1*lag)
                input_table[col] = temp_table[col]
                input_table = input_table.dropna(axis = 0)
                datainput = np.array(input_table)
                temp_sheet[k]  = np.hstack((lag, np.around(Moving_Block_BT(datainput,0.975,0.025),decimals = 4)))
            temp_sheet = pd.DataFrame(temp_sheet)            
            temp_sheet.columns = ['lag','mean','p-value','q-0.025','q-0.975']
            temp_sheet.to_excel(writer, sheet_name=index+'_'+col)                
    writer.close()

import numpy as np
import pandas as pd
import datetime as dt

#----------------------------------------------------------------------------------------
# function to run OLS regression and return the coefs and s.d. in Hansen-Hordrick fashion
#----------------------------------------------------------------------------------------
# H is number of overlapping observation
# MAT is the matrix that has first columns as y_t+1 and the rest columns are X_t 
def HH_OLS(H,MAT):
    T, n= MAT.shape
    y = np.array(MAT[:,:1])
    X = np.hstack((np.ones([T,1]), MAT[:,1:n]))
    beta = np.linalg.inv(X.T@X)@(X.T@y)
    u = y - X@beta
    u = u*np.ones([1,n])
    err = np.array(X)*np.array(u) # estimating residuals for each beta
    V = (err.T@err)/T # regular weighiting matrix
    if H > -1:
        for ind_i in range(H):
            S = np.array(err)[:T-ind_i,:].T@np.array(err)[:T-ind_i,:]/T;
            V = V + (1-0*ind_i/(H+1))*(S + S.T);
    D = np.linalg.inv((X.T@X)/T)
    varb = 1/T*D@V@D;
    seb = np.diag(np.array(varb))
    std_HH = np.sign(seb)*(np.abs(seb)**0.5)
    y_bar = np.mean(np.array(y))
    R2 = np.array((beta.T@X.T@X@beta-T*(y_bar**2))/(y.T@y-T*(y_bar**2)))
    return(np.hstack((np.array(beta.T)[0],std_HH,R2[0]*100)))            

#----------------------------------------------------------------------------------
# function to aggregate the dividend with given reinvestment adjustment
#--------------------------------------------------------------------------------    
    
# T is the horzon we consider, for example if we use monthly data and one year as dividend sums the value is 12
# Dividend is just the vanilla dividends (1-D array)
# Reinvest is the reinvestment log returns for next period (1-D array)
def Dividend_Adjust(T,Dividend,Reinvest):
    m = Dividend.size 
    temp_array_D = np.tile(np.append(Dividend, np.nan), T)[:-T]
    D_mat = np.flip(temp_array_D.reshape([T,m]).T,axis = 1)
    R_mat = np.empty([T,m])
    R_mat[0] = np.zeros(m)
    R_mat[1,:] = np.hstack((np.nan*np.zeros(1),Reinvest[:-1]))
    for k in range(1,T-1):
        if k > 0:
           R_mat[k+1,:] = R_mat[k,:]+np.hstack((np.nan*np.zeros(k+1),Reinvest[:-(k+1)]))
    R_mat_final = np.flip(R_mat.T, axis = 1)
    R_mat_final = np.exp(R_mat_final)
    D_mat_final = D_mat*R_mat_final
    D_adjusted = D_mat_final.sum(axis = 1)    
    return(D_adjusted)

#----------------------------------------------------------------------------------------
# function to compute the forecast errors from OLS in-sample regression in OOS
#----------------------------------------------------------------------------------------
# H is number of overlapping observation
# MAT is the matrix that has first columns as y_t+1 and the rest columns are X_t     
# Vec is the current info that includes the regressors
def HH_OLS_fore_err(H,MAT,Vec):
    T, n= MAT.shape
    y = np.array(MAT[:,:1])
    X = np.hstack((np.ones([T,1]), MAT[:,1:n]))
    beta = np.linalg.inv(X.T@X)@(X.T@y)
    u = y - X@beta
    u = u*np.ones([1,n])
    err = np.array(X)*np.array(u) # estimating residuals for each beta
    V = (err.T@err)/T # regular weighiting matrix
    if H > -1:
        for ind_i in range(H):
            S = np.array(err)[:T-ind_i,:].T@np.array(err)[:T-ind_i,:]/T
            V = V + (1-0*ind_i/(H+1))*(S + S.T)
    D = np.linalg.inv((X.T@X)/T)
    varb = 1/T*D@V@D
    forecast = Vec@beta
    f_error = np.sqrt(Vec@varb@Vec.T)
    return(np.hstack((np.array(forecast).flatten(),np.array(f_error).flatten(),\
                      np.array(beta.T).flatten(), np.atleast_2d(Vec)[0,1])))   

def HH_OLS_full_err(H,MAT,Vec):
    T, n= MAT.shape
    y = np.array(MAT[:,:1])
    X = np.hstack((np.ones([T,1]), MAT[:,1:n]))
    beta = np.linalg.inv(X.T@X)@(X.T@y)
    u = y - X@beta
    u = u*np.ones([1,n])
    err = np.array(X)*np.array(u) # estimating residuals for each beta
    V = (err.T@err)/T # regular weighiting matrix
    if H > -1:
        for ind_i in range(H):
            S = np.array(err)[:T-ind_i,:].T@np.array(err)[:T-ind_i,:]/T
            V = V + (1-0*ind_i/(H+1))*(S + S.T)
    D = np.linalg.inv((X.T@X)/T)
    varb = 1/T*D@V@D
    forecast = Vec@beta
    f_error = np.sqrt(np.diag(Vec@varb@Vec.T))
    return(np.vstack((np.array(forecast).flatten(),np.array(f_error))))   

#-------------------------------------------------------------------------------
# functions to record all forecast in expending window and forecast errors with ARn or Nonlinear models

# table is the DataFrame that has information of the return, y_t, etc
# datefield is to specify the name of column that represents the date
# X, y field is the y = a + bX variables
# startdate is the earliest forecast time
# datausage is the absolute cut-off for how long we should look back
# H is the HH overlapping observation
# ARn is the forecasting model choice: integer     
# resample is to make choices of sampling when overlapping observation is presented
# H and resample is in a sense redundant if the original table's frequence is know -- we do not complify things here


# output of the function is a table that records all the forecasts, std and the coefficients

def Forecast_table_ARn(table,datefield,y_field,X_field,startdate,datausage,H,ARn,resample,mthly):
    forecastdates = np.array(table[table[datefield] >= startdate][datefield])# find the forecasting dates
    table = table[table[datefield]>=datausage] # cut-off data if customise the data usage    
    result = np.empty([forecastdates.size, ARn+4])
    full_info_current = np.empty([forecastdates.size, ARn+1])
    for m in range(forecastdates.size):        
        if resample >1:
            hist_data = (table[table[datefield]<np.floor(forecastdates[m]/100)*100-100])[::-resample]
            hist_info = np.flip(np.array(hist_data[[y_field,X_field]].dropna(axis=0)),axis = 0)        
        else:
            hist_data = table[table[datefield]<np.floor(forecastdates[m]/100)*100-100]
            hist_info = np.array(hist_data[[y_field,X_field]].dropna(axis=0))
        for k in range(ARn-1):
            new_row = np.insert(hist_info.T[k+1],0,np.nan)
            hist_info = np.vstack((hist_info.T, new_row[:-1])).T
        current_data = table[table[datefield] <= forecastdates[m]][::-resample]
        current_data = current_data[:ARn]        
        current_info = np.array(np.insert(np.array(current_data[[X_field]]).flatten(),0,1))
        full_info_current[m] = current_info
        hist_info = hist_info[~np.isnan(hist_info).any(axis=1)]  # drop all nan values  
        result[m] = HH_OLS_fore_err(H, hist_info, current_info)
        if m == forecastdates.size-1:    
            if resample >1:
                full_sample = (table[table[datefield]<np.floor(forecastdates[m]/100)*100])[::-resample]
            else:
                full_sample = table[table[datefield]<np.floor(forecastdates[m]/100)*100]
            full_info = np.flip(np.array(full_sample[[y_field,X_field]].dropna(axis=0)),axis = 0)
            for k in range(ARn-1):
                new_row = np.insert(full_info.T[k+1],0,np.nan)
                full_info = np.vstack((full_info.T, new_row[:-1])).T    
            full_info_hist = full_info[~np.isnan(full_info).any(axis=1)]            
            fullsample_result = HH_OLS_full_err(H, full_info_hist, full_info_current)
    #print(forecastdates.shape, fullsample_result.shape)
    full_time =  np.vstack((forecastdates,fullsample_result))   
    forecast_table = pd.DataFrame(np.vstack((full_time,result.T)).T)    
    forecast_table_colnames = ['date', 'full_sample', 'full_sample_std','forecast_AR'+str(ARn), 'forecast_std']
    for n in range(ARn+1):
        forecast_table_colnames = np.append(forecast_table_colnames, 'a_'+str(n))
    forecast_table_colnames = np.append(forecast_table_colnames, X_field)
    forecast_table.columns = forecast_table_colnames
    return(forecast_table)        


def Forecast_table_NLN(table,datefield,y_field,X_field,startdate,datausage,H,NLN,resample,mthly):
    forecastdates = np.array(table[table[datefield] >= startdate][datefield])# find the forecasting dates
    table = table[table[datefield]>=datausage] # cut-off data if customise the data usage    
    result = np.empty([forecastdates.size, NLN+4])
    full_info_current = np.empty([forecastdates.size, NLN+1])
    for m in range(forecastdates.size):        
        if resample >1:
            hist_data = (table[table[datefield]<np.floor(forecastdates[m]/100)*100-100])[::-resample]
        else:
            hist_data = table[table[datefield]<np.floor(forecastdates[m]/100)*100-100]
        hist_info = np.flip(np.array(hist_data[[y_field,X_field]].dropna(axis=0)),axis = 0)
        for k in range(NLN-1):            
            hist_info = np.vstack((hist_info.T,np.power(hist_info.T[1],k+2))).T
        current_data = np.array(table[table[datefield]==forecastdates[m]][[X_field]])        
        current_info = np.atleast_2d(np.power(current_data[-1:].item(),np.cumsum(np.ones([1,NLN+1]))-1)) 
        full_info_current[m] = current_info
        hist_info = hist_info[~np.isnan(hist_info).any(axis=1)]  # drop all nan values  
        result[m] = HH_OLS_fore_err(H, hist_info, current_info)
        if m == forecastdates.size-1:    
            if resample >1:
                full_sample = (table[table[datefield]<np.floor(forecastdates[m]/100)*100])[::-resample]
            else:
                full_sample = table[table[datefield]<np.floor(forecastdates[m]/100)*100]
            full_info = np.flip(np.array(full_sample[[y_field,X_field]].dropna(axis=0)),axis = 0)
            for k in range(NLN-1):            
                full_info = np.vstack((full_info.T,np.power(full_info.T[1],k+2))).T         
            full_info_hist = full_info[~np.isnan(full_info).any(axis=1)]            
            fullsample_result = HH_OLS_full_err(H, full_info_hist, full_info_current)
    #print(forecastdates.shape, fullsample_result.shape)
    full_time =  np.vstack((forecastdates,fullsample_result))   
    forecast_table = pd.DataFrame(np.vstack((full_time,result.T)).T)    
    forecast_table_colnames = ['date', 'full_sample', 'full_sample_std','forecast_NLN'+str(NLN), 'forecast_std']
    for n in range(NLN+1):
        forecast_table_colnames = np.append(forecast_table_colnames, 'a_'+str(n))
    forecast_table_colnames = np.append(forecast_table_colnames, X_field)
    forecast_table.columns = forecast_table_colnames
    return(forecast_table)        



#-------------------------------------------------------------------------------
# functions to record all forecast in expending window and Bootstrapped bands

# table is the DataFrame that has information of the return, y_t, etc
# datefield is to specify the name of column that represents the date
# X, y field is the y = a + bX variables
# startdate is the earliest forecast time
# datausage is the absolute cut-off for how long we should look back
# H is the HH overlapping observation
# ARn is the forecasting model choice: integer     
# resample is to make choices of sampling when overlapping observation is presented
# H and resample is in a sense redundant if the original table's frequence is know -- we do not complify things here


# output of the function is a table that records all the forecasts, std and the coefficients

def Quantile_func(input_data,L):
    T = len(input_data)
    out_put = np.ones([T,1])
    for m in range(T):
        out_put[m] = np.quantile(input_data[m],L)
    return(np.ndarray.flatten(out_put))

def Fulltime_table_ARn_random(table,datefield,y_field,X_field,startdate,datausage,H,ARn,resample,mthly):
    forecastdates = np.array(table[table[datefield] >= startdate][datefield])# find the forecasting dates
    table = table[table[datefield]>=datausage] # cut-off data if customise the data usage    
#    result = np.empty([forecastdates.size, ARn+4])
    full_info_current = np.empty([forecastdates.size, ARn+1])
    for m in range(forecastdates.size):        
#        if resample >1:
#            hist_data = (table[table[datefield]<np.floor(forecastdates[m]/100)*100-100])[::-resample]
#            hist_info = np.flip(np.array(hist_data[[y_field,X_field]].dropna(axis=0)),axis = 0)        
#        else:
#            hist_data = table[table[datefield]<np.floor(forecastdates[m]/100)*100-100]
#            hist_info = np.array(hist_data[[y_field,X_field]].dropna(axis=0))
#        for k in range(ARn-1):
#            new_row = np.insert(hist_info.T[k+1],0,np.nan)
#            hist_info = np.vstack((hist_info.T, new_row[:-1])).T
        current_data = table[table[datefield] <= forecastdates[m]][::-resample]
        current_data = current_data[:ARn]        
        current_info = np.atleast_2d(np.insert(np.array(current_data[[X_field]]).flatten(),0,1))
        full_info_current[m] = current_info
#        hist_info = hist_info[~np.isnan(hist_info).any(axis=1)]  # drop all nan values  
#        result[m] = HH_OLS_fore_err(H, hist_info, current_info)
#        if m == forecastdates.size-1:    
    if resample >1:
        full_sample = (table[table[datefield]<np.floor(forecastdates[m]/100)*100])[::-resample]
    else:
        full_sample = table[table[datefield]<np.floor(forecastdates[m]/100)*100]
    full_info = np.flip(np.array(full_sample[[y_field,X_field]].dropna(axis=0)),axis = 0)
    for k in range(ARn-1):
        new_row = np.insert(full_info.T[k+1],0,np.nan)
        full_info = np.vstack((full_info.T, new_row[:-1])).T    
    full_info_hist = full_info[~np.isnan(full_info).any(axis=1)]
    full_time = HH_OLS_full_err(H, full_info_hist, full_info_current)    
    T = len(full_info_hist)
    m = int(T**(1/3))
    for n in range(10000):
        u = np.ceil((T-m)*np.random.random_sample([int(np.ceil(T/m)),1]))
        u = u+np.arange(0,m)
        u = np.ndarray.flatten(u).astype(int)
        full_info_hist_temp = full_info_hist[u]  
        temp_fullsample_result = HH_OLS_full_err(H, full_info_hist_temp, full_info_current)
    #print(forecastdates.shape, fullsample_result.shape)
        full_time =  np.vstack((full_time,temp_fullsample_result))   
#    print(full_time.shape)
#    print(np.array(forecastdates).shape)
    forecast_table = pd.DataFrame()
    forecast_table['date'] = np.array(forecastdates)    
    forecast_table['full_sample_0.9'] = Quantile_func(full_time.T,0.9)
    forecast_table['full_sample_0.95'] = Quantile_func(full_time.T,0.95) 
#    for n in range(ARn+1):
#        forecast_table_colnames = np.append(forecast_table_colnames, 'a_'+str(n))
#    forecast_table_colnames = np.append(forecast_table_colnames, X_field)
#    forecast_table.columns = forecast_table_colnames
    return(forecast_table)        

#------------------------------------------------------------------------------
# CRSP data
#------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
import os # 
filepath_data = os.path.dirname(os.path.abspath(__file__))
filepath_compute = os.path.dirname(os.path.abspath(__file__))
filepath_table = os.path.dirname(os.path.abspath(__file__))

#------------------------------------------------------------------------------

# load different data source
CRSP_mthly = pd.read_csv(filepath_data + '/Pseudo_Data/CRSP_SPX_mthly.csv')
CRSP_mthly = CRSP_mthly[['caldt','spindx','vwretd','vwretx']]
CRSP_mthly['unadj_D'] = CRSP_mthly['spindx'].shift(1)*(CRSP_mthly['vwretd']-CRSP_mthly['vwretx'])
CRSP_mthly['mkt_ret'] = np.log(CRSP_mthly['vwretd'].shift(-1)+1)
CRSP_mthly['caldt'] = CRSP_mthly.caldt.apply(lambda x: dt.datetime.strptime(x,'%d/%m/%Y').year*100+\
          dt.datetime.strptime(x,'%d/%m/%Y').month)

Rf_mthly = pd.read_csv(filepath_data + '/Pseudo_Data/1-3mRf.csv')
Rf_mthly = Rf_mthly[Rf_mthly['kytreasnox']==2000001] # to choose the shorter bonds
Rf_mthly = Rf_mthly[['mcaldt','tmytm','tmduratn']]

Rf_mthly['rf'] = np.log(0.01*Rf_mthly['tmytm']+1)/12
CRSP_mthly['rf'] = Rf_mthly['rf'] 


#------------------------------------------------------------------------------
CRSP12m = pd.DataFrame()
CRSP12m_Rf = pd.DataFrame()
CRSP12m_Rm = pd.DataFrame()

CRSP_full_table = { 'CRSP 12mo (no adjustment)': CRSP12m, \
                    'CRSP 12mo (risk-free)': CRSP12m_Rf, \
                    'CRSP 12mo (market)': CRSP12m_Rm}

# HH overlapping observation
CRSP_HH = {'CRSP 12mo (no adjustment)': 0, \
           'CRSP 12mo (risk-free)': 0, \
           'CRSP 12mo (market)': 0}

# dividend adjustment category
CRSP_adjustment = {'CRSP 12mo (no adjustment)': 0, \
                   'CRSP 12mo (risk-free)': 1, \
                   'CRSP 12mo (market)': 2}

# horizon in terms of months
CRSP_horizon = {'CRSP 12mo (no adjustment)': 12, \
                'CRSP 12mo (risk-free)': 12, \
                'CRSP 12mo (market)': 12}

CRSP_table_names = {#'CRSP 12mo (no adjustment)', \
                    'CRSP 12mo (risk-free)'}#, \
                    #'CRSP 12mo (market)'}

#------------------------------------------------------------------------------
# Computing monthly data of r, g, y etc

for table_title in CRSP_table_names:
    temp_table = CRSP_full_table[table_title]  
    temp_table['date'] = CRSP_mthly['caldt']
    temp_table['P_t'] = CRSP_mthly['spindx']
    # adjust for the dividend reinvestments
    if CRSP_adjustment[table_title] == 0:
       temp_table['D_t'] = Dividend_Adjust(CRSP_horizon[table_title],\
                 np.array(CRSP_mthly['unadj_D']), np.zeros(np.array(CRSP_mthly['unadj_D']).size))
    elif CRSP_adjustment[table_title] == 1:
       temp_table['D_t'] = Dividend_Adjust(CRSP_horizon[table_title],\
                 np.array(CRSP_mthly['unadj_D']), np.array(CRSP_mthly['rf']))          
    elif CRSP_adjustment[table_title] == 2:
       temp_table['D_t'] = Dividend_Adjust(CRSP_horizon[table_title],\
                 np.array(CRSP_mthly['unadj_D']), np.array(CRSP_mthly['mkt_ret']))         
    temp_table['-g_t+1']  = np.log(temp_table['D_t']/temp_table['D_t'].shift(-CRSP_horizon[table_title]))
    temp_table['r_t+1'] = np.log(temp_table['P_t'].shift(-CRSP_horizon[table_title])\
              +temp_table['D_t'].shift(-CRSP_horizon[table_title]))\
                        -np.log(temp_table['P_t'])
    temp_table['r-g_t+1'] = temp_table['r_t+1']+temp_table['-g_t+1'] 
    temp_table['dp_t'] = np.log(temp_table['D_t']/temp_table['P_t'])
    temp_table['y_t'] = np.log(1+temp_table['D_t']/temp_table['P_t'])
    temp_table_ = temp_table[::CRSP_horizon[table_title]]

#------------------------------------------------------------------------------
# expanding window forecasts for Er-g
# give set of months to choose from when the data shall be used

#initial_dates  = np.array([192512,194612])
initial_dates  = np.array([194612])

# record the forecasted results between 199601 and the most recent using different models
for table_title in CRSP_table_names:
    table = CRSP_full_table[table_title]  
    for datausage in initial_dates:
        for ARn in range(3):
             dp_result = Forecast_table_ARn(table,'date','r-g_t+1','dp_t', 199601,\
                                            datausage, 0, ARn+1,CRSP_horizon[table_title],1)
             dp_result.to_excel(filepath_compute+'/Pseudo_results/Month_forecast_rg/' +table_title+'_'+str(datausage)+'_'+'AR'+str(ARn+1)+'dp'+'.xlsx')
             y_result =  Forecast_table_ARn(table,'date','r-g_t+1','y_t',  199601,\
                                            datausage, 0, ARn+1,CRSP_horizon[table_title],1)
             y_result.to_excel(filepath_compute+'/Pseudo_results/Month_forecast_rg/'+table_title+'_'+str(datausage)+'_'+'AR'+str(ARn+1)+'y'+'.xlsx')
        for NLN in range(3):
             dp_result = Forecast_table_NLN(table,'date','r-g_t+1','dp_t', 199601,\
                                            datausage, 0, NLN+1,CRSP_horizon[table_title],1)
             dp_result.to_excel(filepath_compute+'/Pseudo_results/Month_forecast_rg/' +table_title+'_'+str(datausage)+'_'+'NLN'+str(NLN+1)+'dp'+'.xlsx')
             y_result =  Forecast_table_NLN(table,'date','r-g_t+1','y_t',  199601,\
                                            datausage, 0, NLN+1, CRSP_horizon[table_title], 1)
             y_result.to_excel(filepath_compute+'/Pseudo_results/Month_forecast_rg/'+table_title+'_'+str(datausage)+'_'+'NLN'+str(NLN+1)+'y'+'.xlsx')
             
# computing bootstrapped bands for ARn forecasts and save them in seperate tables
for table_title in CRSP_table_names:
    table = CRSP_full_table[table_title]  
    for datausage in initial_dates:
        for ARn in range(3):
             dp_result = Fulltime_table_ARn_random(table,'date','r-g_t+1','dp_t', 199601,\
                                            datausage, 0, ARn+1,CRSP_horizon[table_title],1)
             dp_result.to_excel(filepath_compute+'/Pseudo_results/Month_forecast_rg/' +table_title+'_'+str(datausage)+'_'+'AR'+str(ARn+1)+'dp'+'_BT.xlsx')
             y_result =  Fulltime_table_ARn_random(table,'date','r-g_t+1','y_t',  199601,\
                                            datausage, 0, ARn+1,CRSP_horizon[table_title],1)
             y_result.to_excel(filepath_compute+'/Pseudo_results/Month_forecast_rg/'+table_title+'_'+str(datausage)+'_'+'AR'+str(ARn+1)+'y'+'_BT.xlsx')
             
             
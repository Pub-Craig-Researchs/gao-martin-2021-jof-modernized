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
# full sample regression table
# we illustrate how to create tables using "fake CRSP data"
# the results is saved in the "/Pseudo_results/" folder where all latex code is generated in a txt file


CRSP12m = pd.DataFrame()
CRSP12m_Rf = pd.DataFrame()
CRSP12m_Rm = pd.DataFrame()

CRSP24m = pd.DataFrame()
CRSP24m_Rf = pd.DataFrame()
CRSP24m_Rm = pd.DataFrame()

CRSP_full_table = { 'CRSP 12mo (no adjustment)': CRSP12m, \
                    'CRSP 12mo (risk-free)': CRSP12m_Rf, \
                    'CRSP 12mo (market)': CRSP12m_Rm, \
                    'CRSP 24mo (no adjustment)': CRSP24m, \
                    'CRSP 24mo (risk-free)': CRSP24m_Rf, \
                    'CRSP 24mo (market)': CRSP24m_Rm}

CRSP_HH = {'CRSP 12mo (no adjustment)': 0, \
           'CRSP 12mo (risk-free)': 0, \
           'CRSP 12mo (market)': 0, \
           'CRSP 24mo (no adjustment)': 1, \
           'CRSP 24mo (risk-free)': 1, \
           'CRSP 24mo (market)': 1}


CRSP_adjustment = {'CRSP 12mo (no adjustment)': 0, \
                   'CRSP 12mo (risk-free)': 1, \
                   'CRSP 12mo (market)': 2, \
                   'CRSP 24mo (no adjustment)': 0, \
                   'CRSP 24mo (risk-free)': 1, \
                   'CRSP 24mo (market)': 2}


CRSP_horizon = {'CRSP 12mo (no adjustment)': 12, \
                'CRSP 12mo (risk-free)': 12, \
                'CRSP 12mo (market)': 12, \
                'CRSP 24mo (no adjustment)': 12, \
                'CRSP 24mo (risk-free)': 12, \
                'CRSP 24mo (market)': 12}

CRSP_table_names = ['CRSP 12mo (no adjustment)', \
                    'CRSP 12mo (risk-free)', \
                    'CRSP 12mo (market)', \
                    'CRSP 24mo (no adjustment)', \
                    'CRSP 24mo (risk-free)', \
                    'CRSP 24mo (market)']

initial_dates = np.array([192512, 194612])
#initial_dates = np.array([198212])
text_file = open(filepath_table+'/Pseudo_results/CRSP_fullsample_table.txt', "w")
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
    for initial_day in initial_dates:
        temp_data = temp_table_[temp_table_['date']>initial_day].dropna(axis=0)
        result_1 = np.around(HH_OLS(CRSP_HH[table_title],np.array(temp_data[['r-g_t+1','y_t']])), decimals=3)
        result_2 = np.around(HH_OLS(CRSP_HH[table_title],np.array(temp_data[['r_t+1','y_t']])), decimals=3)
        result_3 = np.around(HH_OLS(CRSP_HH[table_title],np.array(temp_data[['-g_t+1','y_t']])), decimals=3)
        result_4 = np.around(HH_OLS(CRSP_HH[table_title],np.array(temp_data[['r-g_t+1','dp_t']])), decimals=3)
        result_5 = np.around(HH_OLS(CRSP_HH[table_title],np.array(temp_data[['r_t+1','dp_t']])), decimals=3)
        result_6 = np.around(HH_OLS(CRSP_HH[table_title],np.array(temp_data[['-g_t+1','dp_t']])), decimals=3)            
        text_file.write('\\begin{table}[tbp]\n\
        \\begin{center}\n\
        \\setlength\\tabcolsep{8pt}\n\
        \\renewcommand{\\arraystretch}{1.3}\n')
        text_file.write('\\begin{tabular}{|c|}\n\
        \\hline\n\
        $\\text{RHS}_{t}$ \\\ \n\
        \\hline \n\
        \\hline \n')
        text_file.write('\\\ \n')
        text_file.write('$y_t$  \\\ \n\
        \\\ \n\
        \\hline \n\
        \\\ \n\
        $dp_t$\\\ \n\
        \\\  \n\
        \\hline \n\
        \\end{tabular} \n')
        text_file.write('\\begin{tabular}{|c|}\n\
        \\hline\n\
        $\\text{LHS}_{t+1}$\\\ \n\
        \\hline \n\
        \\hline \n')
        text_file.write('$r_{t+1}-g_{t+1}$ \\\ \n')
        text_file.write(' $r_{t+1}$ \\\ \n\
        $-g_{t+1}$ \\\ \n\
        \\hline \n\
        $r_{t+1}-g_{t+1}$\\\ \n\
        $r_{t+1}$\\\ \n\
        $-g_{t+1}$\\\  \n\
        \\hline \n\
        \\end{tabular} \n')
        text_file.write('\\begin{tabular}{|c|c|c|c|c|} \n\
        \\hline \n\
        $\widehat{a}_0$&$s.e.$&$\widehat{a}_1$&$s.e.$&$R^2$ \\\ \n\
        \\hline \n\
        \\hline \n')
        text_file.write('$'+str(result_1[0])+'$'+'&'+ '$['+str(result_1[2])+']$'+'&'\
                        +'$'+str(result_1[1])+'$'+'&'+'$['+str(result_1[3])+']$'+'&'+ str(np.around(result_1[4],2))+'\%\\\ \n')
        text_file.write('$'+str(result_2[0])+'$'+'&'+ '$['+str(result_2[2])+']$'+'&'\
                        +'$'+str(result_2[1])+'$'+'&'+'$['+str(result_2[3])+']$'+'&'+ str(np.around(result_2[4],2))+'\%\\\ \n')
        text_file.write('$'+str(result_3[0])+'$'+'&'+ '$['+str(result_3[2])+']$'+'&'\
                        +'$'+str(result_3[1])+'$'+'&'+'$['+str(result_3[3])+']$'+'&'+ str(np.around(result_3[4],2))+'\%\\\ \n')
        text_file.write('\\hline  \n')
        text_file.write('$'+str(result_4[0])+'$'+'&'+ '$['+str(result_4[2])+']$'+'&'\
                        +'$'+str(result_4[1])+'$'+'&'+'$['+str(result_4[3])+']$'+'&'+ str(np.around(result_4[4],2))+'\%\\\ \n')
        text_file.write('$'+str(result_5[0])+'$'+'&'+ '$['+str(result_5[2])+']$'+'&'\
                        +'$'+str(result_5[1])+'$'+'&'+'$['+str(result_5[3])+']$'+'&'+ str(np.around(result_5[4],2))+'\%\\\ \n')
        text_file.write('$'+str(result_6[0])+'$'+'&'+ '$['+str(result_6[2])+']$'+'&'\
                        +'$'+str(result_6[1])+'$'+'&'+'$['+str(result_6[3])+']$'+'&'+ str(np.around(result_6[4],2))+'\%\\\ \n')
        text_file.write('\\hline  \n\
        \\end{tabular} \n\
        \\caption{'+table_title+' '+str(initial_day)+'}  \n\
        \\end{center} \n\
        \\end{table}\n \n \n \n')
text_file.close()





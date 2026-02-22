import numpy as np
import pandas as pd
import datetime as dt


#------------------------------------------------------------------------------
# function to check if there is missing R_f data, i.e. days bond market does not open when option does
def Check_Rf_data():
    missing_days = set(mthly_option['date'].unique()) -set(yieldcurve['date'].unique()) 
    if len(missing_days) >0: 
       print('missing Rf data')
    else: 
       print('no missing Rf data')

#------------------------------------------------------------------------------
# function to compute LVIX

# input data shall be a table of all OTM option prices on the same day with same maturity
# inputdata's columns
# 1)bid  2)ask 3)strike
# riskfree gives the interpolated risk free rate at maturity required
# theta is integer
# T is the required maturity in days

def LVIX(xvals_,pair,inputdata,spotprice):
    # compute the risk-free rates
    xvals = np.array(inputdata['maturity'].unique())
    vix = np.empty([xvals.size,2])
    for m in range(xvals.size):
    #take price info
        bid = np.array(inputdata[inputdata['maturity'] == xvals[m]]['best_bid']) 
        ask = np.array(inputdata[inputdata['maturity'] == xvals[m]]['best_offer'])
        mid = (bid+ask)/2
        strike = np.array(inputdata[inputdata['maturity'] == xvals[m]]['strike_price'])
        if strike.size < 2:
            dstrike = np.nan*np.ones(strike.size)
        elif strike.size == 2:
            dstrike = np.hstack(((strike[1]-strike[0])/2, (strike[-1]-strike[-2])/2))  
        else:
            dstrike = np.hstack(((strike[1]-strike[0])/2, (strike[2:]-strike[:-2])/2 ,(strike[-1]-strike[-2])/2))
        
        # Determine the scalar spotprice
        if isinstance(spotprice, np.ndarray) and spotprice.size > 0:
            sp_val = spotprice[0]
        else:
            sp_val = spotprice

        vix[m]  = np.array([xvals[m], np.sum(dstrike*(strike**(-1))*mid)*(1/sp_val)])
    vix = vix[~np.isnan(vix).any(axis=1)]
    x_ = vix.T[0]
    y_ = vix.T[1]
    interpolated_vix = np.interp(xvals_, x_, y_)
    return(interpolated_vix)


def LVIX_bid(xvals_,pair,inputdata,spotprice):
    # compute the risk-free rates
    xvals = np.array(inputdata['maturity'].unique())
    vix = np.empty([xvals.size,2])
    for m in range(xvals.size):
    #take price info
        bid = np.array(inputdata[inputdata['maturity'] == xvals[m]]['best_bid']) 
        #ask = np.array(inputdata[inputdata['maturity'] == xvals[m]]['best_offer'])
        #mid = (bid+ask)/2
        strike = np.array(inputdata[inputdata['maturity'] == xvals[m]]['strike_price'])
        if strike.size < 2:
            dstrike = np.nan*np.ones(strike.size)
        elif strike.size == 2:
            dstrike = np.hstack(((strike[1]-strike[0])/2, (strike[-1]-strike[-2])/2))  
        else:
            dstrike = np.hstack(((strike[1]-strike[0])/2, (strike[2:]-strike[:-2])/2 ,(strike[-1]-strike[-2])/2))
        
        # Determine the scalar spotprice
        if isinstance(spotprice, np.ndarray) and spotprice.size > 0:
            sp_val = spotprice[0]
        else:
            sp_val = spotprice
            
        vix[m]  = np.array([xvals[m], np.sum(dstrike*(strike**(-1))*bid)*(1/sp_val)])
    vix = vix[~np.isnan(vix).any(axis=1)]
    x_ = vix.T[0]
    y_ = vix.T[1]
    interpolated_vix = np.interp(xvals_, x_, y_)
    return(interpolated_vix)

#-------------------------------------------------------------------------------
import os # this is to set filepath to currrent folder 
filepath_data = os.path.dirname(os.path.abspath(__file__))
filepath_compute = os.path.dirname(os.path.abspath(__file__))


#------------------------------------------------------------------------------
# load different data source

mthly_option = pd.read_csv(filepath_data + '/Pseudo_Data/Monthly_index_option_9619.csv' ) # index option prices
yieldcurve = pd.read_csv(filepath_data+'/Pseudo_Data/Monthly_yield_9619.csv') # term structure data
price_data = pd.read_csv(filepath_data+'/Pseudo_Data/Daily_index_price_9619.csv') # daily index price

# some cleaning of the option price dataset   
mthly_option = mthly_option[mthly_option['forward_price']>0]
mthly_option['strike_price'] = mthly_option['strike_price']/1000
mthly_option = mthly_option[((mthly_option['cp_flag']=='C')&\
                             (mthly_option['strike_price']>mthly_option['forward_price']))\
                            |((mthly_option['cp_flag']=='P')&\
                             (mthly_option['strike_price']<mthly_option['forward_price']))]
mthly_option['mid_price'] = mthly_option['best_bid']/2+mthly_option['best_offer']/2
mthly_option['maturity'] = (mthly_option['exdate'].apply(lambda x: dt.datetime.strptime(x,'%d/%m/%Y'))\
-mthly_option['date'].apply(lambda x: dt.datetime.strptime(x,'%d/%m/%Y'))).apply(lambda x: x.days)
mthly_option = mthly_option[mthly_option['maturity']>7]

#-------------------------------------------------------------------------------------------------------
# computing the LVIX at end of each month
maturities_we_want = np.array([30, 60, 90, 180, 360, 540, 720])
name_date_map = mthly_option[['secid','date']].drop_duplicates()
result = np.empty([1,2+maturities_we_want.size])
for pair in zip(name_date_map['secid'],name_date_map['date']):
         spotprice = np.array(price_data[(price_data['secid']==pair[0])&\
                                         (price_data['date']==pair[1])]['close'])                     
         inputdata = mthly_option[(mthly_option['secid']==pair[0])&\
                                         (mthly_option['date']==pair[1])][['best_bid',\
                                         'best_offer','mid_price','strike_price','maturity']]  
         inputdata = inputdata.sort_values(by = ['maturity','strike_price'])
         newline = np.hstack((np.array(pair),LVIX(maturities_we_want, pair, inputdata, spotprice)))
         result = np.vstack((result, newline))
result = pd.DataFrame(result[1:])
result.columns = ['secid','date','1mo','2mo','3mo','6mo','12mo','18mo','24mo']
result.to_csv(filepath_compute+'/Pseudo_results/LVIX9619.csv')
# write result into spreadsheet
lvix = result
LVIX_SPX100 = lvix[lvix.secid == '109764'][['date','1mo','2mo','3mo','6mo','12mo','18mo','24mo']]
LVIX_SPX500 = lvix[lvix['secid'] == '108105'][['date','1mo','2mo','3mo','6mo','12mo','18mo','24mo']]
LVIX_DOWJ   = lvix[lvix['secid'] == '102456'][['date','1mo','2mo','3mo','6mo','12mo','18mo','24mo']]
LVIX_NDQ    = lvix[lvix['secid'] == '102480'][['date','1mo','2mo','3mo','6mo','12mo','18mo','24mo']]
writer = pd.ExcelWriter(filepath_compute + '/Pseudo_results/LVIX_mthly.xlsx', engine='xlsxwriter')
LVIX_SPX100.to_excel(writer, sheet_name='SPX100')
LVIX_SPX500.to_excel(writer, sheet_name='SPX500')
LVIX_DOWJ.to_excel(writer, sheet_name='DOWJ')
LVIX_NDQ.to_excel(writer, sheet_name='NDQ100')
writer.close()

#------------------------------------------------------------------------------------------------------
# uding bid price repeat the process
result = np.empty([1,2+maturities_we_want.size])
for pair in zip(name_date_map['secid'],name_date_map['date']):
         spotprice = np.array(price_data[(price_data['secid']==pair[0])&\
                                         (price_data['date']==pair[1])]['close'])                     
         inputdata = mthly_option[(mthly_option['secid']==pair[0])&\
                                         (mthly_option['date']==pair[1])][['best_bid',\
                                         'best_offer','mid_price','strike_price','maturity']]  
         inputdata = inputdata.sort_values(by = ['maturity','strike_price'])
         newline = np.hstack((np.array(pair),LVIX_bid(maturities_we_want, pair, inputdata, spotprice)))
         result = np.vstack((result, newline))
result = pd.DataFrame(result[1:])
result.columns = ['secid','date','1mo','2mo','3mo','6mo','12mo','18mo','24mo']
#result.to_csv(filepath_compute+'/LVIX9619_bid_pseudo.csv')
# write result into spreadsheet
lvix = result
LVIX_SPX100 = lvix[lvix['secid'] == '109764'][['date','1mo','2mo','3mo','6mo','12mo','18mo','24mo']]
LVIX_SPX500 = lvix[lvix['secid'] == '108105'][['date','1mo','2mo','3mo','6mo','12mo','18mo','24mo']]
LVIX_DOWJ   = lvix[lvix['secid'] == '102456'][['date','1mo','2mo','3mo','6mo','12mo','18mo','24mo']]
LVIX_NDQ    = lvix[lvix['secid'] == '102480'][['date','1mo','2mo','3mo','6mo','12mo','18mo','24mo']]
writer = pd.ExcelWriter(filepath_compute + '/Pseudo_results/LVIX_mthly_bid.xlsx', engine='xlsxwriter')
LVIX_SPX100.to_excel(writer, sheet_name='SPX100')
LVIX_SPX500.to_excel(writer, sheet_name='SPX500')
LVIX_DOWJ.to_excel(writer, sheet_name='DOWJ')
LVIX_NDQ.to_excel(writer, sheet_name='NDQ100')
writer.close()

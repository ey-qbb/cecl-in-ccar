# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:17:06 2019

@author: chenke
"""


# dont run this if all warnings should be displayed

import warnings
warnings.filterwarnings("ignore")

import os
import requests
import pandas as pds
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import waterfall_chart
from statsmodels.formula.api import ols
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA

os.chdir(r'C:\Users\chenke\Desktop\EY Work\CECL Campaign\CECL in CCAR')

fred_api_key = "e471849abbbc87ca8ec647b8f8b4132a"
# =============================================================================
# quandl_api_key = "xo-Rpzsp47W1kPRG3C5y"
# https://www.quandl.com/api/v3/datasets/FED/FL075035503_Q.json?api_key=xo-Rpzsp47W1kPRG3C5y
# =============================================================================
"""
User inputs
"""

portfolio = input("Input the portfolio: (C&I, CRE, Credit Card, Resi Mortgage, Other Consumer): ")
bank_size = input("\nInput the group of the banks: (Top 100, Not in Top 100): ")
obs_start = input("\nInput the starting point of timeseries for modeling in yyyy-mm-dd (e.g.2000-01-01): ")

"""
- Pull data from the FRED website for historical MEV timeseries
- Four timeseries are pull: 
    GDP - real GDP annualized growth
    UR - natiaonl unemployment rate
    BaaSpread - Baa yield - 10yr treasury
    hpa - YoY change in CS HPI
"""


def get_mev (mev):
    if mev == "CSUSHPINSA":
        fred_api_url = "https://api.stlouisfed.org/fred/series/observations?series_id="+mev+"&+api_key="+fred_api_key+"&file_type=json&observation_start="+obs_start+"&frequency=q&units=pc1"
    else:
        fred_api_url = "https://api.stlouisfed.org/fred/series/observations?series_id="+mev+"&+api_key="+fred_api_key+"&file_type=json&observation_start="+obs_start+"&frequency=q"
    var = pds.DataFrame(requests.request('GET', fred_api_url).json()['observations'])[["date", "value"]]
    
    var.date[var.date.str[5:7] == '01'] = pds.to_numeric(var.date.str[:4])*100 + 1
    var.date[var.date.str[5:7] == '04'] = pds.to_numeric(var.date.str[:4])*100 + 2
    var.date[var.date.str[5:7] == '07'] = pds.to_numeric(var.date.str[:4])*100 + 3
    var.date[var.date.str[5:7] == '10'] = pds.to_numeric(var.date.str[:4])*100 + 4
    
    var = var[var.value != "."]
    
    var.date = pds.to_numeric(var.date)
    var.value = pds.to_numeric(var.value)
    
#    var.set_index(var.date, inplace = True)
    
    return var

gdp = get_mev("A191RL1Q225SBEA")
gdp.rename(columns={'value': 'gdp'}, inplace = True)
ur = get_mev("UNRATE")[["date", "value"]]
ur.rename(columns={'value': 'ur'}, inplace = True)
baa = get_mev("BAA10Y")[["date", "value"]]
baa.rename(columns={'value': 'baa'}, inplace = True)
hpa = get_mev("CSUSHPINSA")[["date", "value"]]
hpa.rename(columns={'value': 'hpa'}, inplace = True)
dpa = get_mev("A067RL1Q156SBEA")[["date", "value"]]
dpa.rename(columns={'value': 'dpa'}, inplace = True)
vix = get_mev("VIXCLS")[["date", "value"]]
vix.rename(columns={'value': 'vix'}, inplace = True)

"""
- Pull net charge-off rates separated by portfolio and bank size
"""

def get_nco (nco):
    
    fred_api_url = "https://api.stlouisfed.org/fred/series/observations?series_id="+nco+"&+api_key="+fred_api_key+"&file_type=json&observation_start="+obs_start+"&frequency=q"
    var = pds.DataFrame(requests.request('GET', fred_api_url).json()['observations'])[["date", "value"]]
    
    var.date[var.date.str[5:7] == '01'] = pds.to_numeric(var.date.str[:4])*100 + 1
    var.date[var.date.str[5:7] == '04'] = pds.to_numeric(var.date.str[:4])*100 + 2
    var.date[var.date.str[5:7] == '07'] = pds.to_numeric(var.date.str[:4])*100 + 3
    var.date[var.date.str[5:7] == '10'] = pds.to_numeric(var.date.str[:4])*100 + 4
    
    var = var[var.value != "."]
    
    var.date = pds.to_numeric(var.date)
    var.value = pds.to_numeric(var.value)
    var.value = var.value/400
    
#    var.set_index(var.date, inplace = True)
    
    return var

if portfolio == 'C&I' and bank_size == 'Top 100':
    nco = get_nco("CORBLT100S") 
    nco.rename(columns={'value': 'nco'}, inplace = True)
    model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
elif portfolio == 'C&I' and bank_size == 'Not in Top 100':
    nco = get_nco("CORBLOBS")   
    nco.rename(columns={'value': 'nco'}, inplace = True)
    model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
elif portfolio == 'CRE' and bank_size == 'Top 100':
    nco = get_nco("CORCREXFT100S") 
    nco.rename(columns={'value': 'nco'}, inplace = True)
    model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
elif portfolio == 'CRE' and bank_size == 'Not in Top 100':
    nco = get_nco("CORCREXFOBS") 
    nco.rename(columns={'value': 'nco'}, inplace = True)
    model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
elif portfolio == 'Credit Card' and bank_size == 'Top 100':
    nco = get_nco("CORCCT100S") 
    nco.rename(columns={'value': 'nco'}, inplace = True)
    model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
elif portfolio == 'Credit Card' and bank_size == 'Not in Top 100':
    nco = get_nco("CORCCOBS") 
    nco.rename(columns={'value': 'nco'}, inplace = True)
    model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
elif portfolio == 'Resi Mortgage' and bank_size == 'Top 100':
    nco = get_nco("CORSFRMT100S") 
    nco.rename(columns={'value': 'nco'}, inplace = True)
    model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
elif portfolio == 'Resi Mortgage' and bank_size == 'Not in Top 100':
    nco = get_nco("CORSFRMOBS") 
    nco.rename(columns={'value': 'nco'}, inplace = True)
    model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
elif portfolio == 'Other Consumer' and bank_size == 'Top 100':
    nco = get_nco("COROCLT100S") 
    nco.rename(columns={'value': 'nco'}, inplace = True)
    model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
elif portfolio == 'Other Consumer' and bank_size == 'Not in Top 100':
    nco = get_nco("COROCLOBS") 
    nco.rename(columns={'value': 'nco'}, inplace = True)
    model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
else: print ("Please endter the right portfolio name and bank size!")

"""
Plot the net charge-off and MEVs in the same charge

"""
model_data_dic = {1: model_data.nco.tolist(), 
                  2: model_data.gdp.tolist(),
                  3: model_data.baa.tolist(),
                  4: model_data.ur.tolist(),
                  5: model_data.hpa.tolist(),
                  6: model_data.dpa.tolist(),
                  7: model_data.vix.tolist()}

fig = plt.figure(tight_layout = True, figsize=(10, 10))
gs = gridspec.GridSpec(4,2)

ax = fig.add_subplot(gs[0, :])
ax.plot(np.arange(0, len(model_data.date),1),model_data.nco, color='C0')
ax.set_ylabel('Net Charge-off')
pos = 2
for i in range(1,4):
    for j in range(0,2):
        ax = fig.add_subplot(gs[i, j])
        ax.plot(np.arange(0, len(model_data.date),1),model_data_dic[pos], color='C1')
        if i==1 and j==0: ax.set_ylabel('GDP growth')
        elif i == 1 and j==1: ax.set_ylabel('Baa spread')
        elif i == 2 and j==0: ax.set_ylabel('Unemployment Rate')
        elif i == 2 and j==1:  ax.set_ylabel('Housing Price Appreciation')
        elif i == 3 and j==0:  ax.set_ylabel('Disposable Income Growth')
        else: ax.set_ylabel('VIX')
        pos += 1
plt.show()



"""
Loss model prediction
"""

while True:
    var1 = input("Input the first MEV for model development (gdp, ur, baa, hpa, dpa, vix): ")
    var2 = input("\nInput the second MEV for model development (gdp, ur, baa, hpa, dpa, vix): ")
    ccar_cycle = input("\nInput the CCAR cycle for performance assessment (2018, 2019) ")
    
    mod_ols = ols(formula="nco~"+var1+"+"+var2, data = model_data).fit()
    #mod_logit = sm.Logit(model_data[['nco']].values, model_data[['gdp', 'ur']].values).fit()
    
    print(mod_ols.summary())
    #print(mod_logit.summary())
    
    pred = pds.DataFrame({'date':model_data.date, 'nco_pred':mod_ols.predict(model_data)})

    ccar = pds.read_excel('CCAR_scenario_input.xlsx', sheet_name='Base_'+ccar_cycle)
    base = pds.DataFrame({'date':ccar.date, 'nco_base':mod_ols.predict(ccar)}).merge(ccar,on = 'date')
    
    ccar = pds.read_excel('CCAR_scenario_input.xlsx', sheet_name='Adverse_'+ccar_cycle)
    adverse = pds.DataFrame({'date':ccar.date, 'nco_adverse':mod_ols.predict(ccar)}).merge(ccar,on = 'date')
    
    ccar = pds.read_excel('CCAR_scenario_input.xlsx', sheet_name='Severely_'+ccar_cycle)
    severely = pds.DataFrame({'date':ccar.date, 'nco_severely':mod_ols.predict(ccar)}).merge(ccar,on = 'date')

    model_data_pred = model_data.append(base).merge(adverse[['date', 'nco_adverse']], on='date', how='left').merge(severely[['date', 'nco_severely']], on='date', how='left').merge(pred[['date', 'nco_pred']], on='date', how='left')
    
    model_data_pred.nco_base[model_data_pred.date.shift(-1)==1] = model_data_pred.nco_pred
    model_data_pred.nco_adverse[model_data_pred.date.shift(-1)==1] = model_data_pred.nco_pred
    model_data_pred.nco_severely[model_data_pred.date.shift(-1)==1] = model_data_pred.nco_pred

    plt.figure(figsize = (10, 6))
    plt.plot(np.arange(0, len(model_data_pred),1), model_data_pred.nco.tolist())
    plt.plot(np.arange(0, len(model_data_pred),1), model_data_pred.nco_pred.tolist())
    plt.plot(np.arange(0, len(model_data_pred),1), model_data_pred.nco_base.tolist())
    plt.plot(np.arange(0, len(model_data_pred),1), model_data_pred.nco_adverse.tolist())
    plt.plot(np.arange(0, len(model_data_pred),1), model_data_pred.nco_severely.tolist())
    plt.show()
    
    if input("Repeat? (Y=Yes; N=No) ") == 'N': 
        print(f'\n Model is successfully developed with MEVs: {var1} and {var2}')
        break




"""
Perfect vs. Imperfect Foresight (AR vs. MA models)
"""

while True:
    ccar_cycle2 = input("\nInput the CCAR cycle for CECL forecasting (2018, 2019): ")
    scenario = input("\nInput the CCAR scenario (Base, Adverse, Severely): ")
    ccar = pds.read_excel('CCAR_scenario_input.xlsx', sheet_name=scenario+'_'+ccar_cycle2)
    
    while True:
        
        macro = model_data[['date', var1, var2]].append(ccar[['date', var1, var2]], sort=False)
        macro_forecast = ccar.append([ccar[ccar.date == 13]]*120, ignore_index=True)
        macro_forecast.date = range(1,len(macro_forecast)+1)
        
        pf = input("\nP - Perfect foresight; IP - Imperfect foresight: ")
        if pf == 'IP': 
            while True:
                ip_model_var1 = input(f"For {var1}, Autoregressive model - AR; Moving average model - MA: ")
                macro_forecast_var1 = macro_forecast[['date', var1]]
                if ip_model_var1 == 'AR':
                    
                    for i in range(1,15):
                        mev_ar = ARIMA(macro[[var1]][(macro.date>13)|(macro.date<i)], order = (2,0,0)).fit()
                        pred = mev_ar.forecast(steps = 120)[0]
                        tmp = pds.DataFrame({'date': range(i,121+i-1), var1+str(i): pred})
                        macro_forecast_var1 = macro_forecast_var1.merge(tmp, on='date', how='left')
                    
                    plt.figure(figsize = (10, 6))
                    plt.plot(macro_forecast_var1[macro_forecast_var1.date<=13].date.tolist(), macro_forecast_var1[macro_forecast_var1.date<=13][var1].tolist())
                    plt.plot(macro_forecast_var1[macro_forecast_var1.date<=13].date.tolist(), macro_forecast_var1[macro_forecast_var1.date<=13][var1+'2'].tolist())
                    plt.plot(macro_forecast_var1[macro_forecast_var1.date<=13].date.tolist(), macro_forecast_var1[macro_forecast_var1.date<=13][var1+'7'].tolist())
                    plt.plot(macro_forecast_var1[macro_forecast_var1.date<=13].date.tolist(), macro_forecast_var1[macro_forecast_var1.date<=13][var1+'9'].tolist())
                    plt.show()
                else:
                    window = input("\nPlease input the length of the moving window: ")
                    for i in range(1,15):
                        avg = []
                        history = [macro[var1].tolist()[j] for j in range(-13+(i-1)-int(window),-13+(i-1))]
                        for k in range(120):
                            avg.append(np.mean([history[l] for l in range(k,k+int(window))]))
                            history.append(np.mean([history[l] for l in range(k,k+int(window))]))
                        tmp = pds.DataFrame({'date': range(i,121+i-1), var1+str(i): avg})
                        macro_forecast_var1 = macro_forecast_var1.merge(tmp, on='date', how='left')
                    plt.figure(figsize = (10, 6))
                    plt.plot(macro_forecast_var1[macro_forecast_var1.date<=13].date.tolist(), macro_forecast_var1[macro_forecast_var1.date<=13][var1].tolist())
                    plt.plot(macro_forecast_var1[macro_forecast_var1.date<=13].date.tolist(), macro_forecast_var1[macro_forecast_var1.date<=13][var1+'2'].tolist())
                    plt.plot(macro_forecast_var1[macro_forecast_var1.date<=13].date.tolist(), macro_forecast_var1[macro_forecast_var1.date<=13][var1+'7'].tolist())
                    plt.plot(macro_forecast_var1[macro_forecast_var1.date<=13].date.tolist(), macro_forecast_var1[macro_forecast_var1.date<=13][var1+'9'].tolist())
                    plt.show()
                if input("Repeat? (Y=Yes; N=No) ") == 'N': 
                    break
            while True:
                ip_model_var2 = input(f"For {var2}, Autoregressive model - AR; Moving average model - MA: ")
                macro_forecast_var2 = macro_forecast[['date', var2]]
                if ip_model_var2 == 'AR':
                    
                    for i in range(1,15):
                        mev_ar = ARIMA(macro[[var2]][(macro.date>13)|(macro.date<i)], order = (2,0,0)).fit()
                        pred = mev_ar.forecast(steps = 120)[0]
                        tmp = pds.DataFrame({'date': range(i,121+i-1), var2+str(i): pred})
                        macro_forecast_var2 = macro_forecast_var2.merge(tmp, on='date', how='left')
                    
                    plt.figure(figsize = (10, 6))
                    plt.plot(macro_forecast_var2[macro_forecast_var2.date<=13].date.tolist(), macro_forecast_var2[macro_forecast_var2.date<=13][var2].tolist())
                    plt.plot(macro_forecast_var2[macro_forecast_var2.date<=13].date.tolist(), macro_forecast_var2[macro_forecast_var2.date<=13][var2+'2'].tolist())
                    plt.plot(macro_forecast_var2[macro_forecast_var2.date<=13].date.tolist(), macro_forecast_var2[macro_forecast_var2.date<=13][var2+'7'].tolist())
                    plt.plot(macro_forecast_var2[macro_forecast_var2.date<=13].date.tolist(), macro_forecast_var2[macro_forecast_var2.date<=13][var2+'9'].tolist())
                    plt.show()
                else:
                    window = input("Please input the length of the moving window: ")
                    for i in range(1,15):
                        avg = []
                        history = [macro[var2].tolist()[j] for j in range(-13+(i-1)-int(window),-13+(i-1))]
                        for k in range(120):
                            avg.append(np.mean([history[l] for l in range(k,k+int(window))]))
                            history.append(np.mean([history[l] for l in range(k,k+int(window))]))
                        tmp = pds.DataFrame({'date': range(i,121+i-1), var2+str(i): avg})
                        macro_forecast_var2 = macro_forecast_var2.merge(tmp, on='date', how='left')
                    plt.figure(figsize = (10, 6))
                    plt.plot(macro_forecast_var2[macro_forecast_var2.date<=13].date.tolist(), macro_forecast_var2[macro_forecast_var2.date<=13][var2].tolist())
                    plt.plot(macro_forecast_var2[macro_forecast_var2.date<=13].date.tolist(), macro_forecast_var2[macro_forecast_var2.date<=13][var2+'2'].tolist())
                    plt.plot(macro_forecast_var2[macro_forecast_var2.date<=13].date.tolist(), macro_forecast_var2[macro_forecast_var2.date<=13][var2+'6'].tolist())
                    plt.plot(macro_forecast_var2[macro_forecast_var2.date<=13].date.tolist(), macro_forecast_var2[macro_forecast_var2.date<=13][var2+'9'].tolist())
                    plt.show()
                if input("Repeat? (Y=Yes; N=No) ") == 'N': 
                    break
        if input("Repeat the overall forecast selection? (Y=Yes; N=No) ") == 'N': 
            if pf == 'P': 
                macro_forecast_final = macro_forecast[['date', var1, var2]]
                print("Perfect foresight assumption is used ")
                nco_fct = pds.DataFrame({'date': macro_forecast_final['date'], 'nco': mod_ols.predict(macro_forecast_final)}).clip_lower(0.00005)
                
                plt.figure(figsize = (10, 6))
                plt.plot(nco_fct[nco_fct.date<=13].date.tolist(), nco_fct[nco_fct.date<=13]['nco'].tolist())
                plt.ylim(bottom = 0)
                plt.show()
            else: 
                macro_forecast_final = macro_forecast_var1.merge(macro_forecast_var2, on='date', how='outer')
                
                for i in range(10):
                    tmp = pds.DataFrame({'date': macro_forecast_final['date'], var1: macro_forecast_final[var1+str(i+1)], var2: macro_forecast_final[var2+str(i+1)]})
                    if i== 0: nco_fct = pds.DataFrame({'date': macro_forecast_final['date'], 'nco_'+str(i+1): mod_ols.predict(tmp)}).clip_lower(0.00005)
                    else: nco_fct = nco_fct.merge(pds.DataFrame({'date': macro_forecast_final['date'], 'nco_'+str(i+1): mod_ols.predict(tmp)}), on='date').clip_lower(0.00005)
                print(f'Imperfect foresight assumption is used; {var1} is used {ip_model_var1} model and {var2} is used {ip_model_var2} model.')
                
                plt.figure(figsize = (10, 6))
                for i in range(10):
                    plt.plot(nco_fct[nco_fct.date<=13].date.tolist(), nco_fct[nco_fct.date<=13]['nco_'+str(i+1)].tolist())
                plt.ylim(bottom = 0)
                plt.show()
            break
    
    
    """
    Scoring the NCO rate and calculate the CECL forecast
    """
    def balance_calc (life, new_loan_assumption, new_loan_pct, amortization_type, ir):
        balance_t0 = 1000000000
        balance = {1: [],
                   2: [],
                   3: [],
                   4: [],
                   5: [],
                   6: [],
                   7: [],
                   8: [],
                   9: []}
        for index in balance:
            for i in range(int(life)+8):
                if index == 1:
                    if i == 0: balance[index].append(balance_t0)
                    else: 
                        if amortization_type == 'linear': balance[index].append(max(balance[index][i-1] - balance[index][0] / int(life),0))
                        else: balance[index].append(max(balance[index][i-1]+np.ppmt(float(ir)/400, i, int(life), balance[index][0]),0)) 
                elif index == 2:
                    if i < 1: balance[index].append(0)
                    elif i == 1: 
                        if new_loan_assumption == 'ME': balance[index].append(balance_t0 - balance[index-1][i])
                        else: balance[index].append(balance_t0 * float(new_loan_pct)/100) 
                    else: 
                        if amortization_type == 'linear': balance[index].append(max(balance[index][i-1] - balance[index][1] / int(life),0))
                        else: balance[index].append(max(balance[index][i-1]+np.ppmt(float(ir)/400, i-1, int(life), balance[index][1]),0))
                elif index == 3:
                    if i < 2: balance[index].append(0)
                    elif i == 2: 
                        if new_loan_assumption == 'ME': balance[index].append(balance_t0 - balance[index-1][i]- balance[index-2][i])
                        else: balance[index].append(balance_t0 * float(new_loan_pct)/100) 
                    else: 
                        if amortization_type == 'linear': balance[index].append(max(balance[index][i-1] - balance[index][2] / int(life),0))
                        else: balance[index].append(max(balance[index][i-1]+np.ppmt(float(ir)/400, i-2, int(life), balance[index][2]),0))
                elif index == 4:
                    if i < 3: balance[index].append(0)
                    elif i == 3: 
                        if new_loan_assumption == 'ME': balance[index].append(balance_t0 - balance[index-1][i]- balance[index-2][i]- balance[index-3][i])
                        else: balance[index].append(balance_t0 * float(new_loan_pct)/100) 
                    else: 
                        if amortization_type == 'linear': balance[index].append(max(balance[index][i-1] - balance[index][3] / int(life),0))
                        else: balance[index].append(max(balance[index][i-1]+np.ppmt(float(ir)/400, i-3, int(life), balance[index][3]),0))
                elif index == 5:
                    if i < 4: balance[index].append(0)
                    elif i == 4: 
                        if new_loan_assumption == 'ME': balance[index].append(balance_t0 - balance[index-1][i]- balance[index-2][i]- balance[index-3][i]- balance[index-4][i])
                        else: balance[index].append(balance_t0 * float(new_loan_pct)/100) 
                    else: 
                        if amortization_type == 'linear': balance[index].append(max(balance[index][i-1] - balance[index][4] / int(life),0))
                        else: balance[index].append(max(balance[index][i-1]+np.ppmt(int(ir)/400, i-4, int(life), balance[index][4]),0))
                elif index == 6:
                    if i < 5: balance[index].append(0)
                    elif i == 5: 
                        if new_loan_assumption == 'ME': balance[index].append(balance_t0 - balance[index-1][i]- balance[index-2][i]- balance[index-3][i]- balance[index-4][i]- balance[index-5][i])
                        else: balance[index].append(balance_t0 * float(new_loan_pct)/100) 
                    else: 
                        if amortization_type == 'linear': balance[index].append(max(balance[index][i-1] - balance[index][5] / int(life),0))
                        else: balance[index].append(max(balance[index][i-1]+np.ppmt(float(ir)/400, i-5, int(life), balance[index][5]),0))
                elif index == 7:
                    if i < 6: balance[index].append(0)
                    elif i == 6: 
                        if new_loan_assumption == 'ME': balance[index].append(balance_t0 - balance[index-1][i]- balance[index-2][i]- balance[index-3][i]- balance[index-4][i]- balance[index-5][i]- balance[index-6][i])
                        else: balance[index].append(balance_t0 * float(new_loan_pct)/100)
                    else: 
                        if amortization_type == 'linear': balance[index].append(max(balance[index][i-1] - balance[index][6] / int(life),0))
                        else: balance[index].append(max(balance[index][i-1]+np.ppmt(float(ir)/400, i-6, int(life), balance[index][6]),0))
                elif index == 8:
                    if i < 7: balance[index].append(0)
                    elif i == 7: 
                        if new_loan_assumption == 'ME': balance[index].append(balance_t0 - balance[index-1][i]- balance[index-2][i]- balance[index-3][i]- balance[index-4][i]- balance[index-5][i]- balance[index-6][i]- balance[index-7][i])
                        else: balance[index].append(balance_t0 * float(new_loan_pct)/100)
                    else: 
                        if amortization_type == 'linear': balance[index].append(max(balance[index][i-1] - balance[index][7] / int(life),0))
                        else: balance[index].append(max(balance[index][i-1]+np.ppmt(float(ir)/400, i-7, int(life), balance[index][7]),0))
                else:
                    if i < 8: balance[index].append(0)
                    elif i == 8: 
                        if new_loan_assumption == 'ME': balance[index].append(balance_t0 - balance[index-1][i]- balance[index-2][i]- balance[index-3][i]- balance[index-4][i]- balance[index-5][i]- balance[index-6][i]- balance[index-7][i]- balance[index-8][i])
                        else: balance[index].append(balance_t0 * float(new_loan_pct)/100)
                    else: 
                        if amortization_type == 'linear': balance[index].append(max(balance[index][i-1] - balance[index][8] / int(life),0))
                        else: balance[index].append(max(balance[index][i-1]+np.ppmt(float(ir)/400, i-8, int(life), balance[index][8]),0))
        return balance
    
    
    life = input("Input the life of the existing and new loans (in quarters): ")
    
    amortization_type = input("\nInput the amortization type for both the existing and new loans (linear, regular): ")
    if amortization_type != 'linear': ir = input("\nInput the interest rate assumption (in integer - 5 = 5%): ")
    else: ir = 0
    
    new_loan_assumption = input("\nInput the assumption for new loan balance (Make Even - ME; Percentage of Existing - PE): ") 
    if new_loan_assumption != 'ME': new_loan_pct = input("\nInput the percentage of the existing portfolio at T0 (in integer - 1 = 1%): ")
    else: new_loan_pct = 0
    
    bal = pds.DataFrame(balance_calc(life, new_loan_assumption, new_loan_pct, amortization_type, ir))
    bal.rename(columns = {1:'bal_1',2:'bal_2',3:'bal_3',4:'bal_4',5:'bal_5',6:'bal_6',7:'bal_7',8:'bal_8',9:'bal_9'}, inplace = True)
    bal['date'] = range(1,int(life)+9) 
    
    bal_print = pds.DataFrame({'date':bal['date'], 'existing': bal['bal_1'], 'new': sum([bal['bal_'+str(i)] for i in range(2,10)])})
            
    #bal_print[['existing', 'new']].plot.area()
    plt.figure(figsize = (10, 6))
    plt.stackplot(range(1,int(life)+9), bal_print['existing'], bal_print['new'], labels=['Existing','New'])
    plt.legend(loc='upper right')
    plt.show()
    
    
    """
    Start of CECL forecast
    """
    result = {'date': list(range(10)),
              'cecl': [],
              'nco':[],
              'build':[],
              'provision':[]}
    
    rs_period = input("Input the Reasonable and Supportable period (N - no R&S period; # - number of quarters): ")
    
    """
    T0 NCO forecast
    """
    baseline = pds.read_excel('CCAR_scenario_input.xlsx', sheet_name='Base_'+ccar_cycle2)
    baseline = baseline.append([baseline[baseline.date == 13]]*120, ignore_index=True)
    baseline.date = range(1,len(baseline)+1)
    nco_fct_t0 = pds.DataFrame({'date': baseline['date'], 'nco': mod_ols.predict(baseline)}).clip_lower(0.00005)
          
    
    """
    CECL forecast
    """    
    if pf == "IP":                                                 
        if rs_period == 'N':
            
            """
            T0
            """
            tmp = pds.DataFrame({'date': bal.date, 'bal': bal.bal_1}).merge(nco_fct_t0[['date', 'nco']], on='date', how = 'inner')
            result['cecl'].append(np.sum(tmp['bal']*tmp['nco']))
            
            """
            T1 - T9
            """
            for i in range (9):
                if i == 8: tmp = pds.DataFrame({'date': bal.date, 'bal': sum([bal['bal_'+str(j)] for j in range(1,i+2)])}).merge(nco_fct[['date', 'nco_'+str(i+2)]], on='date', how = 'inner') 
                else: tmp = pds.DataFrame({'date': bal.date, 'bal': sum([bal['bal_'+str(j)] for j in range(1,i+3)])}).merge(nco_fct[['date', 'nco_'+str(i+2)]], on='date', how = 'inner')
                result['cecl'].append(np.sum(tmp['bal']*tmp['nco_'+str(i+2)]))
        else:
            downturn_in_yrs = input("\nDefine the TTC measures by inputting number of years a downturn is observed (in yrs): ")
            nco_data = model_data[['date', 'nco']]
            nco_data['weight'] = 1
            if portfolio == 'C&I' or portfolio == 'CRE':
                if nco_data.iloc[0].date > 200204: nco_data.weight[(200801>nco_data.date) | (nco_data.date>200904)] = (int(downturn_in_yrs)*4-8)/(len(nco_data)-8)
                else: nco_data.weight[((200801>nco_data.date) & (nco_data.date>200204)) | (200101>nco_data.date) | (nco_data.date>200904)] = (int(downturn_in_yrs)*4-16)/(len(nco_data)-16)
            else: nco_data.weight[(200801>nco_data.date) | (nco_data.date>200904)] = (int(downturn_in_yrs)*4-8)/(len(nco_data)-8)
                
            ttc_nco = np.sum(nco_data['nco']*nco_data['weight']) / np.sum(nco_data['weight'])
            
            print(f"\nThe simple average of NCO using the historical data is {round(np.average(nco_data['nco'])*100,4)}%. The adjusted TTC NCO used in the final calculation is {round(ttc_nco*100,4)}%. ")
            
            """
            T0
            """
            tmp = pds.DataFrame({'date': bal.date, 'bal': bal.bal_1}).merge(nco_fct_t0[['date', 'nco']], on='date', how = 'inner')
            tmp['nco'][(tmp.date>int(rs_period))] = ttc_nco
            result['cecl'].append(np.sum(tmp['bal']*tmp['nco']))
            
            """
            T1 - T9
            """
            
            for i in range (9):
                if i == 8: tmp = pds.DataFrame({'date': bal.date, 'bal': sum([bal['bal_'+str(j)] for j in range(1,i+2)])}).merge(nco_fct[['date', 'nco_'+str(i+2)]], on='date', how = 'inner') 
                else: tmp = pds.DataFrame({'date': bal.date, 'bal': sum([bal['bal_'+str(j)] for j in range(1,i+3)])}).merge(nco_fct[['date', 'nco_'+str(i+2)]], on='date', how = 'inner')
                
                tmp['nco_'+str(i+2)][(tmp.date>=i+2+int(rs_period))] = ttc_nco
                result['cecl'].append(np.sum(tmp['bal']*tmp['nco_'+str(i+2)]))
    else:
        if rs_period == 'N':
            
            """
            T0
            """
            tmp = pds.DataFrame({'date': bal.date, 'bal': bal.bal_1}).merge(nco_fct_t0[['date', 'nco']], on='date', how = 'inner')
            result['cecl'].append(np.sum(tmp['bal']*tmp['nco']))
            
            """
            T1 - T9
            """
            for i in range (9):
                if i == 8: tmp = pds.DataFrame({'date': bal.date, 'bal': sum([bal['bal_'+str(j)] for j in range(1,i+2)])}).merge(nco_fct[['date', 'nco']], on='date', how = 'inner')
                else: tmp = pds.DataFrame({'date': bal.date, 'bal': sum([bal['bal_'+str(j)] for j in range(1,i+3)])}).merge(nco_fct[['date', 'nco']], on='date', how = 'inner')
                tmp['bal'][(tmp.date < i+2)] = 0
                result['cecl'].append(np.sum(tmp['bal']*tmp['nco']))
        else:
            downturn_in_yrs = input("\nDefine the TTC measures by inputting number of years a downturn is observed (in yrs): ")
            nco_data = model_data[['date', 'nco']]
            nco_data['weight'] = 1
            if portfolio == 'C&I' or portfolio == 'CRE':
                if nco_data.iloc[0].date > 200204: nco_data.weight[(200801>nco_data.date) | (nco_data.date>200904)] = (int(downturn_in_yrs)*4-8)/(len(nco_data)-8)
                else: nco_data.weight[((200801>nco_data.date) & (nco_data.date>200204)) | (200101>nco_data.date) | (nco_data.date>200904)] = (int(downturn_in_yrs)*4-16)/(len(nco_data)-16)
            else: nco_data.weight[(200801>nco_data.date) | (nco_data.date>200904)] = (int(downturn_in_yrs)*4-8)/(len(nco_data)-8)
                
            ttc_nco = np.sum(nco_data['nco']*nco_data['weight']) / np.sum(nco_data['weight'])
            
            print(f"\nThe simple average of NCO using the historical data is {round(np.average(nco_data['nco'])*100,4)}%. The adjusted TTC NCO used in the final calculation is {round(ttc_nco*100,4)}%. ")
            
            """
            T0
            """
            tmp = pds.DataFrame({'date': bal.date, 'bal': bal.bal_1}).merge(nco_fct_t0[['date', 'nco']], on='date', how = 'inner')
            tmp['nco'][(tmp.date>int(rs_period))] = ttc_nco
            result['cecl'].append(np.sum(tmp['bal']*tmp['nco']))
            
            """
            T1 - T9
            """
            
            for i in range (9):
                if i == 8: tmp = pds.DataFrame({'date': bal.date, 'bal': sum([bal['bal_'+str(j)] for j in range(1,i+2)])}).merge(nco_fct[['date', 'nco']], on='date', how = 'inner') 
                else: tmp = pds.DataFrame({'date': bal.date, 'bal': sum([bal['bal_'+str(j)] for j in range(1,i+3)])}).merge(nco_fct[['date', 'nco']], on='date', how = 'inner')
                
                tmp['bal'][(tmp.date < i+2)] = 0
                tmp['nco'][(tmp.date>=i+2+int(rs_period))] = ttc_nco
                result['cecl'].append(np.sum(tmp['bal']*tmp['nco']))
    
    """
    calculate NCO
    """
    
    scenario2 = pds.read_excel('CCAR_scenario_input.xlsx', sheet_name= scenario+'_'+ccar_cycle2)
    scenario2 = scenario2.append([scenario2[scenario2.date == 13]]*120, ignore_index=True)
    scenario2.date = range(1,len(scenario2)+1)
    nco_loss = pds.DataFrame({'date': scenario2['date'], 'nco': mod_ols.predict(scenario2)}).clip_lower(0.00005)
          
    tmp = pds.DataFrame({'date': bal.date, 'bal': sum([bal['bal_'+str(j)] for j in range(1,10)])}).merge(nco_loss[['date', 'nco']], on='date', how = 'inner')
    tmp['nco_loss'] = tmp['bal']*tmp['nco']
    
    result['nco'] = tmp['nco_loss'][(tmp.date <= 9)].tolist()
    
    
    """
    calculate reserve build and provision
    """
    result['build'] = [(result['cecl'][i] - result['cecl'][i-1]) for i in range(1,10)]
    result['build'].insert(0, result['cecl'][0])
    result['provision'] = [(result['build'][i+1] + result['nco'][i]) for i in range(9)]
    #result['provision'].insert(0, NaN)
    
    
    """
    Plot reserve build and provision
    """
    
    waterfall_chart.plot(result['date'], result['build'], net_label = 'Q9 CECL', figsize = (10,6))                                                  
    
    cum_prov = []
    for i in range (9):
        cum_prov.append(np.sum([result['provision'][j] for j in range (0, i+1)]))
    
    plt.figure(figsize = (10, 6))
    plt.bar(np.arange(1, len(cum_prov)+1,1), np.array(cum_prov)/1000000)
    plt.plot(np.arange(1, len(cum_prov)+1,1), np.array(result['provision'])/1000000, color='C2')
    plt.show()
    
    if input('Calculate again? (Y-Yes, N - No): ') == 'N': break

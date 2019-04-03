# -*- coding: utf-8 -*-
"""
Created on Tues Mar 12

@author: riverga
"""

import requests
import pandas as pds
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from statsmodels.formula.api import ols
#from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools


def get_mev (mev, key, sdate):
    if mev == "CSUSHPINSA":
        fred_api_url = "https://api.stlouisfed.org/fred/series/observations?series_id="+mev+"&+api_key="+key+"&file_type=json&observation_start="+sdate+"&frequency=q&units=pc1"
    else:
        fred_api_url = "https://api.stlouisfed.org/fred/series/observations?series_id="+mev+"&+api_key="+key+"&file_type=json&observation_start="+sdate+"&frequency=q"
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


def get_nco (nco,key,sdate):
     
    fred_api_url = "https://api.stlouisfed.org/fred/series/observations?series_id="+nco+"&+api_key="+key+"&file_type=json&observation_start="+sdate+"&frequency=q"
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


def pull_data (port,size,sdate):

    portfolio = port
    bank_size = size

    key = "e471849abbbc87ca8ec647b8f8b4132a"

    gdp = get_mev("A191RL1Q225SBEA", key, sdate)
    gdp.rename(columns={'value': 'gdp'}, inplace = True)
    ur = get_mev("UNRATE",key, sdate)[["date", "value"]]
    ur.rename(columns={'value': 'ur'}, inplace = True)
    baa = get_mev("BAA10Y",key, sdate)[["date", "value"]]
    baa.rename(columns={'value': 'baa'}, inplace = True)
    hpa = get_mev("CSUSHPINSA",key, sdate)[["date", "value"]]
    hpa.rename(columns={'value': 'hpa'}, inplace = True)
    dpa = get_mev("A067RL1Q156SBEA",key, sdate)[["date", "value"]]
    dpa.rename(columns={'value': 'dpa'}, inplace = True)
    vix = get_mev("VIXCLS",key, sdate)[["date", "value"]]
    vix.rename(columns={'value': 'vix'}, inplace = True)

    if portfolio == 'C&I' and bank_size == 'Top 100':
        nco = get_nco("CORBLT100S",key,sdate) 
        nco.rename(columns={'value': 'nco'}, inplace = True)
        model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
    elif portfolio == 'C&I' and bank_size == 'Not in Top 100':
        nco = get_nco("CORBLOBS",key,sdate)   
        nco.rename(columns={'value': 'nco'}, inplace = True)
        model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
    elif portfolio == 'CRE' and bank_size == 'Top 100':
        nco = get_nco("CORCREXFT100S",key,sdate) 
        nco.rename(columns={'value': 'nco'}, inplace = True)
        model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
    elif portfolio == 'CRE' and bank_size == 'Not in Top 100':
        nco = get_nco("CORCREXFOBS",key,sdate) 
        nco.rename(columns={'value': 'nco'}, inplace = True)
        model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
    elif portfolio == 'Credit Card' and bank_size == 'Top 100':
        nco = get_nco("CORCCT100S",key,sdate) 
        nco.rename(columns={'value': 'nco'}, inplace = True)
        model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
    elif portfolio == 'Credit Card' and bank_size == 'Not in Top 100':
        nco = get_nco("CORCCOBS",key,sdate) 
        nco.rename(columns={'value': 'nco'}, inplace = True)
        model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
    elif portfolio == 'Resi Mortgage' and bank_size == 'Top 100':
        nco = get_nco("CORSFRMT100S",key,sdate) 
        nco.rename(columns={'value': 'nco'}, inplace = True)
        model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
    elif portfolio == 'Resi Mortgage' and bank_size == 'Not in Top 100':
        nco = get_nco("CORSFRMOBS",key,sdate) 
        nco.rename(columns={'value': 'nco'}, inplace = True)
        model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
    elif portfolio == 'Other Consumer' and bank_size == 'Top 100':
        nco = get_nco("COROCLT100S",key,sdate) 
        nco.rename(columns={'value': 'nco'}, inplace = True)
        model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
    elif portfolio == 'Other Consumer' and bank_size == 'Not in Top 100':
        nco = get_nco("COROCLOBS",key,sdate) 
        nco.rename(columns={'value': 'nco'}, inplace = True)
        model_data = nco.merge(gdp, on='date').merge(ur, on='date').merge(baa, on='date').merge(hpa, on='date').merge(dpa, on='date').merge(vix, on='date')
    else: print ("Please enter the right portfolio name and bank size!")

    model_data['date2'] = ''

    for i in range(0,len(model_data.date)):
        tmp = str(model_data.date[i])
        tmp = tmp[0:4]+'Q'+tmp[5:7]
        model_data['date2'][i] = tmp

    return model_data

def model_dict(data):

    model_data = data
    model_data_dic = {1: model_data.nco.tolist(), 
                  2: model_data.gdp.tolist(),
                  3: model_data.baa.tolist(),
                  4: model_data.ur.tolist(),
                  5: model_data.hpa.tolist(),
                  6: model_data.dpa.tolist(),
                  7: model_data.vix.tolist()}

    return model_data_dic
    

#def plot_nco(data, data_dict):
#    model_data = data
#    model_data_dic = data_dict
#    
#    fig = plt.figure(tight_layout = True, figsize=(10, 10))
#    gs = gridspec.GridSpec(4,2)
#
#    ax = fig.add_subplot(gs[0, :])
#    ax.plot(np.arange(0, len(model_data.date),1),model_data.nco, color='#188CE5')
#    ax.set_ylabel('Net Charge-off')
#    pos = 2
#    for i in range(1,4):
#        for j in range(0,2):
#            ax = fig.add_subplot(gs[i, j])
#            with plt.rc_context({'axes.edgecolor':'white'}):
#                ax.plot(np.arange(0, len(model_data.date),1),model_data_dic[pos], color='#2DB757')
#            if i==1 and j==0: ax.set_ylabel('GDP growth')
#            elif i == 1 and j==1: ax.set_ylabel('Baa spread')
#            elif i == 2 and j==0: ax.set_ylabel('Unemployment Rate')
#            elif i == 2 and j==1:  ax.set_ylabel('Housing Price Appreciation')
#            elif i == 3 and j==0:  ax.set_ylabel('Disposable Income Growth')
#            else: ax.set_ylabel('VIX')
#            ax.spines['bottom'].set_edgecolor('white')
#            pos += 1
#    #plt.show()
#    plt.close(fig)
#    
#    return fig

def plotly_nco(data,data_dict):

    fig = tools.make_subplots(rows=4, cols=2,specs=
                              [[{'colspan':2},None],
                             [{},{}],
                             [{},{}],
                             [{},{}]],
                             subplot_titles=('Net Charge-off',
                             'Gross Domestic Product',
                             'Baa Credit Spread',
                             'US Unemployment Rate',
                             'Housing Price Appreciation',
                             'Disposable Income Growth',
                             'Volatility Index'))
    
    fig.add_scatter(x=data['date2'],y=data['nco'],row=1,col=1,line=dict(color='#188CE5'))
    fig.add_scatter(x=data['date2'],y=data['gdp'],row=2,col=1,line=dict(color='#FF4136'))
    fig.add_scatter(x=data['date2'],y=data['baa'],row=2,col=2,line=dict(color='#2DB757'))
    fig.add_scatter(x=data['date2'],y=data['ur'],row=3,col=1,line=dict(color='#27ACAA'))
    fig.add_scatter(x=data['date2'],y=data['hpa'],row=3,col=2,line=dict(color='#9C82D4'))
    fig.add_scatter(x=data['date2'],y=data['dpa'],row=4,col=1,line=dict(color='#C981B2'))
    fig.add_scatter(x=data['date2'],y=data['vix'],row=4,col=2,line=dict(color='#FF6D00'))
    
    fig['layout'].update(height=1000, width=800,margin={'t':0})
    fig['layout'].update({
        'plot_bgcolor':'rgba(0,0,0,0)',
        'paper_bgcolor':'rgba(0,0,0,0)',
        'showlegend':False,
        })
    fig['layout']['yaxis1'].update(title='NCO')
    fig['layout']['yaxis2'].update(title='GDP')
    fig['layout']['yaxis3'].update(title='BAA')
    fig['layout']['yaxis4'].update(title='UR')
    fig['layout']['yaxis5'].update(title='HPA')
    fig['layout']['yaxis6'].update(title='DPA')
    fig['layout']['yaxis7'].update(title='VIX')
    
    fig.layout.template = 'plotly_dark'
    
    return fig


def mod_ols(var1,var2,data):
    
    model_data = data
    mod_ols = ols(formula="nco~"+var1+"+"+var2, data = model_data).fit()
    
    return mod_ols


def load_ccar_data(cycle,var1,var2,data,mod):

    #var1 = input("Input the first MEV for model development (gdp, ur, baa, hpa, dpa, vix): ")
    #var2 = input("\nInput the second MEV for model development (gdp, ur, baa, hpa, dpa, vix): ")
    #ccar_cycle = input("\nInput the CCAR cycle for performance assessment (2018, 2019) ")
    ccar_cycle = cycle
    model_data = data
    mod_ols = mod

    #mod_logit = sm.Logit(model_data[['nco']].values, model_data[['gdp', 'ur']].values).fit()
    
    #print(mod_ols.summary())
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

    return model_data_pred


def plot_loss_pred(var1, var2, ccar_data):

    model_data_pred = ccar_data

    fig = plt.figure(figsize = (10, 6))
    plt.plot(np.arange(0, len(model_data_pred),1), model_data_pred.nco.tolist())
    plt.plot(np.arange(0, len(model_data_pred),1), model_data_pred.nco_pred.tolist())
    plt.plot(np.arange(0, len(model_data_pred),1), model_data_pred.nco_base.tolist())
    plt.plot(np.arange(0, len(model_data_pred),1), model_data_pred.nco_adverse.tolist())
    plt.plot(np.arange(0, len(model_data_pred),1), model_data_pred.nco_severely.tolist())
    #plt.show()
    plt.close(fig)

    return fig

def macro_data(var1,var2,data,ccar_data):
    
    model_data = data
    ccar = ccar_data
    
    macro = model_data[['date', var1, var2]].append(ccar[['date', var1, var2]], sort=False)
    
    return macro


def macro_forecast(ccar_data,var1,var2):

    ccar = ccar_data

    macro_forecast = ccar.append([ccar[ccar.date == 13]]*120, ignore_index=True)
    macro_forecast.date = range(1,len(macro_forecast)+1)   

    return macro_forecast


def gen_cast_var(var,mod_type,ma_len,mac_data,mac_cast,data,ccar_data):

    #ip_model_var1 = input(f"For {var1}, Autoregressive model - AR; Moving average model - MA: ")
    ip_model_var = mod_type
    macro = mac_data
    macro_forecast = mac_cast
    var=var
    
    macro_forecast_var = macro_forecast[['date', var]]
    
    
    if ip_model_var == 'AR':
        
        for i in range(1,15):
            mev_ar = ARIMA(macro[[var]][(macro.date>13)|(macro.date<i)], order = (2,0,0)).fit()
            pred = mev_ar.forecast(steps = 120)[0]
            tmp = pds.DataFrame({'date': range(i,121+i-1), var+str(i): pred})
            macro_forecast_var = macro_forecast_var.merge(tmp, on='date', how='left')
        
    else:
        #window = input("\nPlease input the length of the moving window: ")
        window = ma_len
        for i in range(1,15):
            avg = []
            history = [macro[var].tolist()[j] for j in range(-13+(i-1)-int(window),-13+(i-1))]
            for k in range(120):
                avg.append(np.mean([history[l] for l in range(k,k+int(window))]))
                history.append(np.mean([history[l] for l in range(k,k+int(window))]))
            tmp = pds.DataFrame({'date': range(i,121+i-1), var+str(i): avg})
            macro_forecast_var = macro_forecast_var.merge(tmp, on='date', how='left')

    return macro_forecast_var

def plot_ip(cast_var,var):
    macro_forecast_var = cast_var
    
    fig = plt.figure(figsize = (10, 6))
    plt.plot(macro_forecast_var[macro_forecast_var.date<=13].date.tolist(), macro_forecast_var[macro_forecast_var.date<=13][var].tolist())
    plt.plot(macro_forecast_var[macro_forecast_var.date<=13].date.tolist(), macro_forecast_var[macro_forecast_var.date<=13][var+'2'].tolist())
    plt.plot(macro_forecast_var[macro_forecast_var.date<=13].date.tolist(), macro_forecast_var[macro_forecast_var.date<=13][var+'7'].tolist())
    plt.plot(macro_forecast_var[macro_forecast_var.date<=13].date.tolist(), macro_forecast_var[macro_forecast_var.date<=13][var+'9'].tolist())
    #plt.show()    
    
    plt.close(fig)
    
    return fig


def nco_cast(mac_cast,var1,var2,mod,choice,cast_var1,cast_var2):
    macro_forecast = mac_cast
    mod_ols = mod
    pf = choice
    macro_forecast_var1 = cast_var1
    macro_forecast_var2 = cast_var2

    if pf == 'P': 
        macro_forecast_final = macro_forecast[['date', 'date2', var1, var2]]
        #print("Perfect foresight assumption is used ")
        nco_fct = pds.DataFrame({'date': macro_forecast_final['date'],'nco': mod_ols.predict(macro_forecast_final)}).clip_lower(0.00005)
        nco_fct['date2'] = macro_forecast['date2']
    else: 
        macro_forecast_final = macro_forecast_var1.merge(macro_forecast_var2, on='date', how='outer')
        
        for i in range(10):
            tmp = pds.DataFrame({'date': macro_forecast_final['date'],var1: macro_forecast_final[var1+str(i+1)], var2: macro_forecast_final[var2+str(i+1)]})
            if i== 0: 
                nco_fct = pds.DataFrame({'date': macro_forecast_final['date'], 'nco_'+str(i+1): mod_ols.predict(tmp)}).clip_lower(0.00005)
            else: 
                nco_fct = nco_fct.merge(pds.DataFrame({'date': macro_forecast_final['date'], 'nco_'+str(i+1): mod_ols.predict(tmp)}), on='date').clip_lower(0.00005)
                
            #print(f'Imperfect foresight assumption is used; {var1} is used {ip_model_var1} model and {var2} is used {ip_model_var2} model.')
            
    nco_fct['date2'] = macro_forecast['date2']
    return nco_fct


#def plot_pf(nco_fct,choice):
#
#    #ccar_cycle2 = cycle
#    #scenario = scen
#    pf = choice
#    
#    if pf == 'P':
#        fig = plt.figure(figsize = (10, 6))
#        plt.plot(nco_fct[nco_fct.date<=13].date.tolist(), nco_fct[nco_fct.date<=13]['nco'].tolist())
#        plt.ylim(bottom = 0)
#        #plt.show()
#    else:
#        fig = plt.figure(figsize = (10, 6))
#        for i in range(10):
#            plt.plot(nco_fct[nco_fct.date<=13].date.tolist(), nco_fct[nco_fct.date<=13]['nco_'+str(i+1)].tolist())
#        plt.ylim(bottom = 0)
#        #plt.show()
#
#    plt.close(fig)
#    return fig

def plotly_pf(nco_fct,choice):
    tmp = nco_fct[nco_fct.date<=13]
    
    fig = go.Figure()
    
    if choice == 'P':
        fig.add_scatter(x=tmp['date2'],y=tmp['nco'],line={'color':'#188CE5'})
    else:
        for i in range(10):
            fig.add_scatter(x=tmp['date2'],y=tmp['nco_'+str(i+1)])
        
    fig['layout'].update(height=400,width=800,margin={'t':0})
    fig['layout'].update({
        'plot_bgcolor':'rgba(0,0,0,0)',
        'paper_bgcolor':'rgba(0,0,0,0)',
        'showlegend':False
        })
    fig['layout']['yaxis1'].update(title='NCO')
    fig['layout']['xaxis1'].update(title=None)
    
    fig.layout.template = 'plotly_dark'
    
    
    return fig


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


def amort_bal(life, grow_assump, grow_pct, amort_type, rate):
    new_loan_assumption = grow_assump
    new_loan_pct = grow_pct
    amortization_type = amort_type
    ir = rate
    
    bal = pds.DataFrame(balance_calc(life, new_loan_assumption, new_loan_pct, amortization_type, ir))
    bal.rename(columns = {1:'bal_1',2:'bal_2',3:'bal_3',4:'bal_4',5:'bal_5',6:'bal_6',7:'bal_7',8:'bal_8',9:'bal_9'}, inplace = True)
    bal['date'] = range(1,int(life)+9)    

    return bal


def plot_bal(bal,life):

    bal_print = pds.DataFrame({'date':bal['date'], 'existing': bal['bal_1'], 'new': sum([bal['bal_'+str(i)] for i in range(2,10)])})
            
    #bal_print[['existing', 'new']].plot.area()
    fig = plt.figure(figsize = (10, 6))
    plt.stackplot(range(1,int(life)+9), bal_print['existing'], bal_print['new'], labels=['Existing','New'])
    plt.legend(loc='upper right')
    #plt.show()    

    plt.close(fig)

    return fig

def plotly_bal(bal,life):
    
    bal_print = pds.DataFrame({'date':bal['date'], 'existing': bal['bal_1'], 'new': sum([bal['bal_'+str(i)] for i in range(2,10)])})
    
    fig = go.Figure()
    fig.add_scatter(x=np.arange(1, int(life)+9,1),y=bal_print['existing'],stackgroup='one',line={'color':'#9C82D4'})
    fig.add_scatter(x=np.arange(1, int(life)+9,1),y=bal_print['new'],stackgroup='one',line={'color':'#93F0E6'})
    
    fig['layout'].update(width=800,margin={'t':0})
    fig['layout'].update({
        'plot_bgcolor':'rgba(0,0,0,0)',
        'paper_bgcolor':'rgba(0,0,0,0)',
        'showlegend':True
        })
#    fig['layout']['yaxis1'].update(title='Provision ($m)')
#    fig['layout']['xaxis1'].update(title='Quarters')
    
    fig.layout.template = 'plotly_dark'        
    
    return fig


def cecl_calc(rs_len,choice,bal,dt_len,data,port,mod,nco_fct,cycle,):
    rs_period = rs_len
    downturn_in_yrs = dt_len
    model_data = data
    portfolio = port
    pf = choice

    result = {'date': list(range(10)),
              'cecl': [],
              'nco':[],
              'build':[],
              'provision':[]}

    ccar_cycle2 = cycle
    mod_ols = mod

    baseline = pds.read_excel('CCAR_scenario_input.xlsx', sheet_name='Base_'+ccar_cycle2)
    baseline = baseline.append([baseline[baseline.date == 13]]*120, ignore_index=True)
    baseline.date = range(1,len(baseline)+1)
    nco_fct_t0 = pds.DataFrame({'date': baseline['date'], 'nco': mod_ols.predict(baseline)}).clip_lower(0.00005)

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
            #downturn_in_yrs = input("\nDefine the TTC measures by inputting number of years a downturn is observed (in yrs): ")
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
            #downturn_in_yrs = dt_len
            #input("\nDefine the TTC measures by inputting number of years a downturn is observed (in yrs): ")
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

    return result


def nco_calc(scen,cycle,mod,result,bal):
    ccar_cycle2 = cycle
    scenario = scen
    mod_ols = mod

    scenario2 = pds.read_excel('CCAR_scenario_input.xlsx', sheet_name= scenario+'_'+ccar_cycle2)
    scenario2 = scenario2.append([scenario2[scenario2.date == 13]]*120, ignore_index=True)
    scenario2.date = range(1,len(scenario2)+1)
    nco_loss = pds.DataFrame({'date': scenario2['date'], 'nco': mod_ols.predict(scenario2)}).clip_lower(0.00005)
          
    tmp = pds.DataFrame({'date': bal.date, 'bal': sum([bal['bal_'+str(j)] for j in range(1,10)])}).merge(nco_loss[['date', 'nco']], on='date', how = 'inner')
    tmp['nco_loss'] = tmp['bal']*tmp['nco']
    
    result['nco'] = tmp['nco_loss'][(tmp.date <= 9)].tolist()    

    return result


def provision_calc(result):

    result['build'] = [(result['cecl'][i] - result['cecl'][i-1]) for i in range(1,10)]
    result['build'].insert(0, result['cecl'][0])
    result['provision'] = [(result['build'][i+1] + result['nco'][i]) for i in range(9)]
    #result['provision'].insert(0, NaN)    

    return result


def waterfall_plotly(result):
    
    base = [0 for x in range(len(result['provision'])+1)]   
    pos = [0 for x in range(len(result['provision'])+1)]  
    neg = [0 for x in range(len(result['provision'])+1)]  
    anchor = [0 for x in range(len(result['provision'])+1)]    

    for i in range(len(result['provision'])):
        if i==0:
            anchor[i] = result['provision'][i]
            base[i] = result['provision'][i]
        else: 
            if result['provision'][i]<0:
                neg[i] = result['provision'][i]
                base[i] = base[i-1]+neg[i-1]+pos[i-1]
            else:
                pos[i] = result['provision'][i]
                base[i] = base[i-1]+neg[i-1]+pos[i-1]
    ix = len(result['provision'])   
    anchor[ix] = base[ix-1]+neg[ix-1]+pos[ix-1]
                
    fig = go.Figure()
    fig.add_bar(x=result['date'],y=anchor,marker={'color':'#747480'})
    fig.add_bar(x=result['date'],y=base,marker={'opacity':0})
    fig.add_bar(x=result['date'],y=pos,marker={'color':'#2DB757'})
    fig.add_bar(x=result['date'],y=neg,marker={'color':'#FF4136'})
    
    fig['layout'].update(width=800,margin={'t':0})
    fig['layout'].update({
        'plot_bgcolor':'rgba(0,0,0,0)',
        'paper_bgcolor':'rgba(0,0,0,0)',
        'showlegend':False
        })    
    
    fig['layout'].update(barmode='stack')  
    fig['layout']['yaxis1'].update(title='Provision ($m)')
    fig['layout']['xaxis1'].update(title='Quarters')
    
    fig.layout.template = 'plotly_dark'                                                    
    
    return fig
    

#def provision_plot(result):
#    cum_prov = []
#    for i in range (9):
#        cum_prov.append(np.sum([result['provision'][j] for j in range (0, i+1)]))
#    
#    fig = plt.figure(figsize = (10, 6))
#    plt.bar(np.arange(1, len(cum_prov)+1,1), np.array(cum_prov)/1000000)
#    plt.plot(np.arange(1, len(cum_prov)+1,1), np.array(result['provision'])/1000000, color='C2')
#    #plt.show()
#    
#    plt.close(fig)
#    
#    return fig

def provision_plotly(result):
    
    cum_prov = []
    for i in range (9):
        cum_prov.append(np.sum([result['provision'][j] for j in range (0, i+1)]))    
    
    fig = go.Figure()
    fig.add_bar(x=np.arange(1, len(cum_prov)+1,1),y=np.array(cum_prov)/1000000,marker={'color':'#747480'})
    fig.add_scatter(x=np.arange(1, len(cum_prov)+1,1),y=np.array(result['provision'])/1000000,line={'color':'#FFE600'})
    
    fig['layout'].update(width=800,margin={'t':0})
    fig['layout'].update({
        'plot_bgcolor':'rgba(0,0,0,0)',
        'paper_bgcolor':'rgba(0,0,0,0)',
        'showlegend':False
        })
    fig['layout']['yaxis1'].update(title='Provision ($m)')
    fig['layout']['xaxis1'].update(title='Quarters')
    
    fig.layout.template = 'plotly_dark'    
    
    return fig



def main():
    
    port = 'Resi Mortgage'
    size = 'Top 100'
    sdate = '2000-01-01'
    var1 = 'ur'
    var2 = 'hpa'
    cycle = '2018'
    choice = 'P'
    mod_type1 = 'MA'
    mod_type2 = 'MA'
    ma_len1 = 2
    ma_len2 = 2
    scen = "Base"
    life = 40
    amort_type = 'linear'
    rate = 5
    grow_assump = 'ME'
    grow_pct = 1
    rs_len = 12
    dt_len = 25

    # Pull data from API
    data = pull_data(port, size, sdate)
    data_dict = model_dict(data)
    
    # Fit model
    mod1 = mod_ols(var1,var2,data)
    
    # Prepare CCAR data
    ccar_data = load_ccar_data(cycle,var1,var2,data,mod1)    

    # Prepare macro variable data
    mac_data = macro_data(var1,var2,data,ccar_data)
    mac_cast = macro_forecast(ccar_data,var1,var2)  
    
    # Macrovar forecast
    cast_var1 = gen_cast_var(var1,mod_type1,ma_len1,mac_data,mac_cast,data,ccar_data)
    cast_var2 = gen_cast_var(var2,mod_type2,ma_len2,mac_data,mac_cast,data,ccar_data)
  
    # NCO forecast
    nco_fct = nco_cast(mac_cast,var1,var2,mod1,choice,cast_var1,cast_var2)
    
    # Balance runoff
    bal = amort_bal(life, grow_assump, grow_pct, amort_type, rate)
    
    # Generate results
    result = cecl_calc(rs_len,choice,bal,dt_len,data,port,mod1,nco_fct,cycle)
    result = nco_calc(scen, cycle, mod1,result,bal)
    result = provision_calc(result)    

    # Plots
#    nco_plot = plotly_nco(data, data_dict)
#    loss_plot = plot_loss_pred(var1,var2,ccar_data)    
#    m1_ip_plot = plot_ip(cast_var1,var1)
#    m2_ip_plot = plot_ip(cast_var2,var2)
#    pf_plot = plot_pf(nco_fct,choice)
#    bal_plot = plot_bal(bal,life)    
#    walk_plot = waterfall_plot(result)
#    prov_plot = provision_plot(result)
    
    # Show all plots
#    nco_plot
#    loss_plot
#    m1_ip_plot
#    m2_ip_plot
#    pf_plot
#    bal_plot
#    walk_plot
#    prov_plot

    return

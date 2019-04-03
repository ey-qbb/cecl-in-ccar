# -*- coding: utf-8 -*-
"""
Created on Mon Mar 1

@author: riverga
"""
import os
import pandas as pd
import dash
import dash_core_components as dcc 
import dash_html_components as html 
from dash.dependencies import Input, Output, State
import plotly.tools as tls
import plotly.plotly as py

#tls.set_credentials_file(username='griv1012', api_key='T7cd1fUn2qwcQ8dSRHhP')

#os.chdir('c:/users/riverga/desktop/cecl-in-ccar/')

#os.chdir('./assets/')

import cecl_walk as cecl

external_stylesheets = ['https://codepen.io/chriddyp/pen/brPBPO.css']

app = dash.Dash(__name__,static_file='assets',external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
        
    html.Div(
            className = 'twelve columns',
            children=[
                    html.Div('CECL in CCAR Analytics Tool',
                             style={'font-size':40,'color':'#FFE600','margin-left':'2%','font-weight':'bold'},
                             className = 'nine columns'),
                    html.Div(html.Img(src=app.get_asset_url('EY_Logo_Beam_RGB_White_Yellow.png'),
                                      style={'height':60}),
                             style={'height':70,'line-height':70,'text-align':'right','padding-right':0,'margin-left':0},
                             className = 'three columns')                                    
                    ], style={'display':'inline-block','height':70,'line-height':70,'margin-bottom':50,'margin-top':'1%'}
            ),
    
    html.Div(
            className='five columns',
            children=[
                    html.Div(children=[
                            html.H5('Portfolio Characteristics',
                                    style={'border-bottom':'4px solid #C4C4CD'})]),
                    html.Div(children=[
                                html.H6('Portfolio type:',
                                        style={'width':200,'display':'inline-block','vertical-align':'top'}),
                                dcc.Dropdown(
                                            id='port',
                                            options=[
                                                {'label':'C&I', 'value':'C&I'},
                                                {'label':'CRE', 'value':'CRE'},
                                                {'label':'Credit Card', 'value':'Credit Card'},
                                                {'label':'Resi Mortgage', 'value':'Resi Mortgage'},
                                                {'label':'Other Consumer', 'value':'Other Consumer'}
                                            ],
                                            value = 'C&I',style={'width':200,'display':'inline-block'})
                                ],
                            ),
                    html.Div(children=[
                                html.H6('Bank size:',
                                        style={'width':200,'display':'inline-block','vertical-align':'top'}),
                                dcc.Dropdown(
                                            id='size',
                                            options=[
                                                {'label':'Top 100', 'value':'Top 100'},
                                                {'label':'Not in Top 100', 'value':'Not in Top 100'},
                                            ],
                                            value='Top 100',style={'width':200,'display':'inline-block'})
                                ]
                                ),
                    html.Div(children=[
                                html.H6('Start date:',
                                        style={'width':200,'display':'inline-block','vertical-align':'top'}),
                                dcc.Input(id='sdate', value='2000-01-01', type='text', 
                                          style={'width':200,'display':'inline-block'}) 
                            ], style={'display':'inline-block'}),
                    html.Div(children=[
                                html.H5('CCAR Inputs',
                                        style={'border-bottom':'4px solid #C4C4CD'})],
                                               style={'padding-top':15}),
                    html.Div(children=[
                                html.H6('CCAR cycle:',
                                        style={'width':200,'display':'inline-block','vertical-align':'top'}),
                                dcc.Dropdown(
                                            id='ccar-cyc',
                                            options=[
                                                {'label':'2018', 'value':'2018'},
                                                {'label':'2019', 'value':'2019'}
                                    ],
                                    value='2018',style={'width':200,'display':'inline-block'})                                
                                ]
                                ),
                    html.Div(children=[
                                html.H6('CCAR scenario:',
                                        style={'width':200,'display':'inline-block','vertical-align':'top'}),
                                dcc.Dropdown(
                                            id='ccar-scen',
                                            options=[
                                                {'label':'Base', 'value':'Base'},
                                                {'label':'Adverse', 'value':'Adverse'},
                                                {'label':'Severely Adverse', 'value':'Severely'}
                                    ],
                                    value='Base',style={'width':200,'display':'inline-block'})                                
                                ]
                                ),
                    html.Div(children=[
                                html.H5('Forecast Model Parameters:',
                                        style={'border-bottom':'4px solid #C4C4CD'})],
                                               style={'padding-top':15}),
                    html.Div(children=[
                                html.H6('Perfect/Imperfect foresight:',
                                        style={'width':200,'display':'inline-block','vertical-align':'top'}),
                                dcc.Dropdown(
                                    id='fs-opt',
                                        options=[
                                            {'label':'Perfect', 'value':'P'},
                                            {'label':'Imperfect', 'value':'IP'}
                                        ],
                                    value='P',style={'width':200,'display':'inline-block'})                                
                            ],style={'border-bottom':'1px solid #C4C4CD'}),
                    html.Div(children=[
                                html.H6('Macroeconomic variable #1:',
                                        style={'width':200,'display':'inline-block','vertical-align':'top'}),
                                dcc.Dropdown(
                                            id='var1',
                                            options=[
                                                {'label':'US GDP', 'value':'gdp'},
                                                {'label':'US Unemployment Rate', 'value':'ur'},
                                                {'label':'Baa Spread', 'value':'baa'},
                                                {'label':'Housing Price Appr.', 'value':'hpa'},
                                                {'label':'US Disposable Income', 'value':'dpa'},
                                                {'label':'VIX', 'value':'vix'}
                                    ],
                                    value='ur',style={'width':200,'display':'inline-block'}),                                
                                ],style={'padding-top':10}),
                    html.Div(children=[
                            html.H6('Variable #1 model type:',
                                    style={'width':200,'display':'inline-block','vertical-align':'top'}),
                            dcc.Dropdown(
                                id='m1-model',
                                options=[
                                                {'label':'Autoregressive', 'value':'AR'},
                                                {'label':'Moving Average', 'value':'MA'}
                                ],
                                value='AR',style={'width':200,'display':'inline-block'}),                            
                            ]),
                    html.Div(children=[
                            html.H6('Window length (if MA):',
                                    style={'width':200,'display':'inline-block','vertical-align':'top'}),
                            dcc.Input(id='ma-len1', value=2, type='numeric',style={'width':200,'display':'inline-block'})
                            ],style={'border-bottom':'1px solid #C4C4CD'}),
                    html.Div(children=[
                                html.H6('Macroeconomic variable #2:',
                                        style={'width':200,'display':'inline-block','vertical-align':'top'}),
                                dcc.Dropdown(
                                            id='var2',
                                            options=[
                                                {'label':'US GDP', 'value':'gdp'},
                                                {'label':'US Unemployment Rate', 'value':'ur'},
                                                {'label':'Baa Spread', 'value':'baa'},
                                                {'label':'Housing Price Appr.', 'value':'hpa'},
                                                {'label':'US Disposable Income', 'value':'dpa'},
                                                {'label':'VIX', 'value':'vix'}
                                    ],
                                    value='gdp',style={'width':200,'display':'inline-block'}),                                
                                ],style={'padding-top':10}),
                    html.Div(children=[
                            html.H6('Variable #2 model type:',
                                    style={'width':200,'display':'inline-block','vertical-align':'top'}),
                            dcc.Dropdown(
                                id='m2-model',
                                options=[
                                                {'label':'Autoregressive', 'value':'AR'},
                                                {'label':'Moving Average', 'value':'MA'}
                                ],
                                value='AR',style={'width':200,'display':'inline-block'}),                            
                            ]),
                    html.Div(children=[
                            html.H6('Window length (if MA):',
                                    style={'width':200,'display':'inline-block','vertical-align':'top'}),
                            dcc.Input(id='ma-len2', value=2, type='numeric',style={'width':200,'display':'inline-block'})
                            ]), 
                    html.Div(children=[
                            html.H5('Loan Characteristics',
                                    style={'border-bottom':'4px solid #C4C4CD'})],
                                           style={'padding-top':15}),
                    html.Div(children=[
                            html.H6('Amortization type:',
                                    style={'width':200,'display':'inline-block','vertical-align':'top'}),
                            dcc.Dropdown(
                                    id='amort-type',
                                    options=[
                                        {'label':'Linear', 'value':'Linear'},
                                        {'label':'Regular', 'value':'Regular'}
                                    ],
                                value='Linear',style={'width':200,'display':'inline-block'}),                            
                            ]),
                    html.Div(children=[
                            html.H6('New orig. growth assumption:',
                                    style={'width':200,'display':'inline-block','vertical-align':'top'}),
                            dcc.Dropdown(
                                    id='grow-assump',
                                    options=[
                                        {'label':'Make Even', 'value':'ME'},
                                        {'label':'Percentage of Existing', 'value':'PE'}
                                ],
                                value='PE',style={'width':200,'display':'inline-block'})
                            ]),
                    html.Div(children=[
                            html.H6('If % assump., enter %:',
                                    style={'width':200,'display':'inline-block'}),
                            dcc.Input(id='grow-pct', value=1, type='numeric',
                                      style={'width':200,'display':'inline-block'}),
                            ]),
                    html.Div(children=[
                            html.H6('Life of loan (Q):',
                                    style={'width':200,'display':'inline-block'}),
                            dcc.Input(id='life-loan', value=20, type='numeric',
                                      style={'width':200,'display':'inline-block'})
                            ]),
                    html.Div(children=[
                            html.H6('Interest rate:',
                                    style={'width':200,'display':'inline-block'}),
                            dcc.Input(id='rate',value=5,type='numeric',
                                      style={'width':200,'display':'inline-block'})
                            ]),
                    html.Div(children=[
                            html.H6('Length of R&S horizon (Q):',
                                    style={'width':200,'display':'inline-block'}),
                            dcc.Input(id='rs-len', value=12, type='numeric',
                                      style={'width':200,'display':'inline-block'})
                            ]),
                    html.Div(children=[
                            html.H6('Window for 1 downturn (Q):',
                                    style={'width':200,'display':'inline-block'}),
                            dcc.Input(id='dt-win', value=25, type='numeric',
                                      style={'width':200,'display':'inline-block'})
                            ]),
                    html.Div(children=[
                            html.Button(id='submit-btn', n_clicks=0, children='Submit',style={'background-color':'#FFE600'}),
                            ],style={'padding-top':15,'padding-bottom':15}),
                    html.Div(children=[
                            html.A("Link to interactive sensitivity analysis", href='https://kennethchen814.github.io/', target="_blank")
                            ])
                                
                                                             
                                ], style={'margin-left':'2%'}
                                ),
    html.Div(
            className='seven columns',
            children=[
                    html.Div(children=[
                            html.H5('NCO & Macroeconomic Variables:'),
                            dcc.Graph(id = 'nco_plot')
                            ]),        
                    html.Div(children=[
                            html.H5('Balance Run-off:'),
                            dcc.Graph(id = 'bal_plot')
                            ],style={'padding-top':15}),            
                    html.Div(children=[
                            html.H5('Perfect/Imperfect Foresight:'),
                            dcc.Graph(id = 'pf_plot')
                            ]),
                    html.Div(children=[
                            html.H5('Walk from Prior Quarter:'),
                            dcc.Graph(id = 'wfall_plot')
                            ],style={'padding-top':15}),
                    html.Div(children=[
                            html.H5('CECL Provision:'),
                            dcc.Graph(id = 'prov_plot')
                            ],style={'padding-top':15})                    
                    ], style={'display':'inline-block','width':800}
                ),
    
    html.Div(id='data_store',style={'display':'none'}),
    html.Div(id='ccar_data',style={'display':'none'})
    
    ])

                    
@app.callback(Output('data_store','children'),
              [Input('port','value'),Input('size','value'),Input('sdate','value')])
def pull_data(port,size,sdate):
    
    data = cecl.pull_data(port, size, sdate)
    
    return data.to_json()
                    

@app.callback(Output('nco_plot','figure'), [Input('submit-btn','n_clicks'),Input('data_store','children')])
def nco_plot(n_clicks,data_store):
    
    data = pd.read_json(data_store)
    data = data.sort_values('date')
    data_dict = cecl.model_dict(data)
    nco_plot = cecl.plotly_nco(data, data_dict)
    
    return nco_plot


@app.callback(Output('bal_plot','figure'), [Input('submit-btn','n_clicks')],
              [State('grow-assump','value'),State('amort-type','value'),State('grow-pct','value'),
               State('rate','value'),State('life-loan','value')])
def bal_plot(n_clicks,grow_assump,amort_type,grow_pct,rate,life):
    
    bal = cecl.amort_bal(life, grow_assump, grow_pct, amort_type, rate)    
    fig = cecl.plotly_bal(bal,life)
    
    return fig
  

@app.callback(Output('pf_plot','figure'), [Input('submit-btn','n_clicks'),Input('data_store','children')],
              [State('var1','value'),State('var2','value'),State('ccar-cyc','value'),
               State('m1-model','value'),State('m2-model','value'),State('ma-len1','value'),
               State('ma-len2','value'),State('fs-opt','value')])
def pf_plot(n_clicks,data_store,var1,var2,cycle,mod_type1,mod_type2,ma_len1,ma_len2,choice):
   
    data = pd.read_json(data_store)
    data = data.sort_values('date')
    
    mod = cecl.mod_ols(var1,var2,data)
    ccar_data = cecl.load_ccar_data(cycle,var1,var2,data,mod) 
    
    mac_data = cecl.macro_data(var1,var2,data,ccar_data)
    mac_cast = cecl.macro_forecast(ccar_data,var1,var2)  
    
    cast_var1 = cecl.gen_cast_var(var1,mod_type1,ma_len1,mac_data,mac_cast,data,ccar_data)
    cast_var2 = cecl.gen_cast_var(var2,mod_type2,ma_len2,mac_data,mac_cast,data,ccar_data)
  
    nco_fct = cecl.nco_cast(mac_cast,var1,var2,mod,choice,cast_var1,cast_var2)

    fig = cecl.plotly_pf(nco_fct,choice)
    
    return fig


@app.callback(Output('wfall_plot','figure'), [Input('submit-btn','n_clicks'),Input('data_store','children')],
              [State('var1','value'),State('var2','value'),State('ccar-cyc','value'),
               State('m1-model','value'),State('m2-model','value'),State('ma-len1','value'),
               State('ma-len2','value'),State('fs-opt','value'),State('grow-assump','value'),
               State('amort-type','value'),State('grow-pct','value'),State('rate','value'),
               State('rs-len','value'),State('dt-win','value'),State('ccar-scen','value'),
               State('port','value'),State('life-loan','value')])
def wfall_plot(n_clicks,data_store,var1,var2,cycle,mod_type1,mod_type2,ma_len1,
              ma_len2,choice,grow_assump,amort_type,grow_pct,rate,rs_len,dt_len,scen,port,life):
    
    data = pd.read_json(data_store)
    data = data.sort_values('date')
    
    mod = cecl.mod_ols(var1,var2,data)
    
    ccar_data = cecl.load_ccar_data(cycle,var1,var2,data,mod)    

    mac_data = cecl.macro_data(var1,var2,data,ccar_data)
    mac_cast = cecl.macro_forecast(ccar_data,var1,var2)  

    cast_var1 = cecl.gen_cast_var(var1,mod_type1,ma_len1,mac_data,mac_cast,data,ccar_data)
    cast_var2 = cecl.gen_cast_var(var2,mod_type2,ma_len2,mac_data,mac_cast,data,ccar_data)

    nco_fct = cecl.nco_cast(mac_cast,var1,var2,mod,choice,cast_var1,cast_var2)

    bal = cecl.amort_bal(life, grow_assump, grow_pct, amort_type, rate)
    
    result = cecl.cecl_calc(rs_len,choice,bal,dt_len,data,port,mod,nco_fct,cycle)
    result = cecl.nco_calc(scen, cycle, mod,result,bal)
    result = cecl.provision_calc(result)  

    waterfall = cecl.waterfall_plotly(result)
    
    return waterfall


@app.callback(Output('prov_plot','figure'), [Input('submit-btn','n_clicks'),Input('data_store','children')],
              [State('var1','value'),State('var2','value'),State('ccar-cyc','value'),
               State('m1-model','value'),State('m2-model','value'),State('ma-len1','value'),
               State('ma-len2','value'),State('fs-opt','value'),State('grow-assump','value'),
               State('amort-type','value'),State('grow-pct','value'),State('rate','value'),
               State('rs-len','value'),State('dt-win','value'),State('ccar-scen','value'),
               State('port','value'),State('life-loan','value')])
def prov_plot(n_clicks,data_store,var1,var2,cycle,mod_type1,mod_type2,ma_len1,
              ma_len2,choice,grow_assump,amort_type,grow_pct,rate,rs_len,dt_len,scen,port,life):
    
    data = pd.read_json(data_store)
    data = data.sort_values('date')
    
    mod = cecl.mod_ols(var1,var2,data)
    
    ccar_data = cecl.load_ccar_data(cycle,var1,var2,data,mod)    

    mac_data = cecl.macro_data(var1,var2,data,ccar_data)
    mac_cast = cecl.macro_forecast(ccar_data,var1,var2)  

    cast_var1 = cecl.gen_cast_var(var1,mod_type1,ma_len1,mac_data,mac_cast,data,ccar_data)
    cast_var2 = cecl.gen_cast_var(var2,mod_type2,ma_len2,mac_data,mac_cast,data,ccar_data)

    nco_fct = cecl.nco_cast(mac_cast,var1,var2,mod,choice,cast_var1,cast_var2)

    bal = cecl.amort_bal(life, grow_assump, grow_pct, amort_type, rate)
    
    result = cecl.cecl_calc(rs_len,choice,bal,dt_len,data,port,mod,nco_fct,cycle)
    result = cecl.nco_calc(scen, cycle, mod,result,bal)
    result = cecl.provision_calc(result)  

    prov_plot = cecl.provision_plotly(result)
    
    return prov_plot

if __name__ == '__main__':
    app.run_server()
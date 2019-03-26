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
import waterfall_chart

#tls.set_credentials_file(username='griv1012', api_key='T7cd1fUn2qwcQ8dSRHhP')

#os.chdir('c:/users/riverga/desktop/cecl-in-ccar/')

os.chdir('./assets/')

import cecl_walk as cecl

output = {}

#external_stylesheets = ['gui-style.css']

app = dash.Dash(__name__)

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
                            ],style={'padding-top':15,'padding-bottom':15})
                                
                                                             
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
                            html.H5('Perfect/Imperfect Foresight:'),
                            dcc.Graph(id = 'pf_plot')
                            ]),
                    html.Div(children=[
                            html.H5('Balance Run-off:'),
                            dcc.Graph(id = 'bal_plot')
                            ],style={'padding-top':15}),
                    html.Div(children=[
                            html.H5('CECL Provision:'),
                            dcc.Graph(id = 'prov_plot')
                            ],style={'padding-top':15})                    
                    ], style={'display':'inline-block','width':800}
                ),


    html.Div(id='port_val',style={'display':'none'}),
    html.Div(id='size_val',style={'display':'none'}),
    html.Div(id='sdate_val',style={'display':'none'}),
    html.Div(id='v1_val',style={'display':'none'}),
    html.Div(id='v2_val',style={'display':'none'}),
    html.Div(id='cycle_val',style={'display':'none'}),
    html.Div(id='scen_val',style={'display':'none'}),
    html.Div(id='fs_val',style={'display':'none'}),
    html.Div(id='m1_val',style={'display':'none'}),
    html.Div(id='m2_val',style={'display':'none'}),
    html.Div(id='amort_val',style={'display':'none'}),
    html.Div(id='grow_val',style={'display':'none'}),
    html.Div(id='grow_pct_val',style={'display':'none'}),
    html.Div(id='life_val',style={'display':'none'}),
    html.Div(id='rs_val',style={'display':'none'}),
    html.Div(id='win_val',style={'display':'none'}),
    html.Div(id='ma_val1',style={'display':'none'}),
    html.Div(id='ma_val2',style={'display':'none'}),
    html.Div(id='rate_val',style={'display':'none'})
    
    ])
#    
#    #dcc.Graph(id = 'nco_plot'),
#    #dcc.Graph(id = 'loss_plot'),
#    dcc.Graph(id = 'pf_plot'),
#    dcc.Graph(id = 'bal_plot'),
#    dcc.Graph(id = 'prov_plot')
    
    
##   ], style={'columnCount': 1})
#    ])
#
#
@app.callback(Output('port_val','children'), [Input('port','value')])
def port_val(value):
    output['portfolio'] = value
    return 

@app.callback(Output('size_val','children'), [Input('size','value')])
def size_val(value):
    output['size'] = value
    return 

@app.callback(Output('sdate_val','children'), [Input('sdate','value')])
def sdate_val(value):
    output['start'] = value
    return 

@app.callback(Output('v1_val','children'), [Input('var1','value')])
def v1_val(value):
    output['v1'] = value
    return 

@app.callback(Output('v2_val','children'), [Input('var2','value')])
def v2_val(value):
    output['v2'] = value
    return 

@app.callback(Output('cycle_val','children'), [Input('ccar-cyc','value')])
def cycle_val(value):
    output['cycle'] = value
    return 

@app.callback(Output('scen_val','children'), [Input('ccar-scen','value')])
def scen_val(value):
    output['scenario'] = value
    return 

@app.callback(Output('fs_val','children'), [Input('fs-opt','value')])
def fs_val(value):
    output['foresight'] = value
    return 

@app.callback(Output('m1_val','children'), [Input('m1-model','value')])
def m1_val(value):
    output['model1'] = value
    return 

@app.callback(Output('m2_val','children'), [Input('m2-model','value')])
def m2_val(value):
    output['model2'] = value
    return 

@app.callback(Output('ma_val1','children'), [Input('ma-len1','value')])
def ma_val1(value):
    output['ma_val1'] = value
    return 

@app.callback(Output('ma_val2','children'), [Input('ma-len2','value')])
def ma_val2(value):
    output['ma_val2'] = value
    return 

@app.callback(Output('amort_val','children'), [Input('amort-type','value')])
def amort_val(value):
    output['amort'] = value
    return 

@app.callback(Output('grow_val','children'), [Input('grow-assump','value')])
def grow_val(value):
    output['grow_assump'] = value
    return 

@app.callback(Output('grow_pct_val','children'), [Input('grow-pct','value')])
def grow_pct_val(value):
    output['grow_pct'] = value
    return 

@app.callback(Output('life_val','children'), [Input('life-loan','value')])
def life_val(value):
    output['life'] = value
    return 

@app.callback(Output('rs_val','children'), [Input('rs-len','value')])
def rs_val(value):
    output['rs'] = value
    return 

@app.callback(Output('rate_val','children'), [Input('rate','value')])
def rate_val(value):
    output['rate'] = value
    return

@app.callback(Output('win_val','children'), [Input('dt-win','value')])
def win_val(value):
    output['window'] = value
    return 

@app.callback(Output('nco_plot','figure'), [Input('submit-btn','n_clicks')])
def nco_plot(figure):
    port = output['portfolio']
    size = output['size']
    sdate = output['start']
    
    # Pull data from API
    data = cecl.pull_data(port, size, sdate)
    data_dict = cecl.model_dict(data)
    nco_plot = cecl.plotly_nco(data, data_dict)
    
    
    return nco_plot
##
##@app.callback(Output('loss_plot','figure'), [Input('submit-btn','n_clicks')])
##def loss_plot(figure):
##    port = output['portfolio']
##    size = output['size']
##    sdate = output['start']
##    var1 = output['v1']    
##    var2 = output['v2']   
##    cycle = output['cycle']
##    
##    # Pull data from API
##    data = cecl.pull_data(port, size, sdate)
##    
##    # Fit model
##    mod1 = cecl.mod_ols(var1,var2,data)
##    
##    # Prepare CCAR data
##    ccar_data = cecl.load_ccar_data(cycle,var1,var2,data,mod1)
##
##    loss_plot = cecl.plot_loss_pred(var1,var2,ccar_data)      
##
##    
##    plotly_loss = tls.mpl_to_plotly(loss_plot)
##    
##    return plotly_loss
#
@app.callback(Output('pf_plot','figure'), [Input('submit-btn','n_clicks')])
def pf_plot(figure):
    port = output['portfolio']
    size = output['size']
    sdate = output['start']
    var1 = output['v1']    
    var2 = output['v2']   
    cycle = output['cycle']
    mod_type1 = output['model1']
    mod_type2 = output['model2']
    ma_len1 = output['ma_val1']
    ma_len2 = output['ma_val2']
    choice = output['foresight']
    
    
    # Pull data from API
    data = cecl.pull_data(port, size, sdate)
    
    # Fit model
    mod1 = cecl.mod_ols(var1,var2,data)
    
    # Prepare CCAR data
    ccar_data = cecl.load_ccar_data(cycle,var1,var2,data,mod1)    

    # Prepare macro variable data
    mac_data = cecl.macro_data(var1,var2,data,ccar_data)
    mac_cast = cecl.macro_forecast(ccar_data,var1,var2)  
    
    # Macrovar forecast
    cast_var1 = cecl.gen_cast_var(var1,mod_type1,ma_len1,mac_data,mac_cast,data,ccar_data)
    cast_var2 = cecl.gen_cast_var(var2,mod_type2,ma_len2,mac_data,mac_cast,data,ccar_data)
  
    # NCO forecast
    nco_fct = cecl.nco_cast(mac_cast,var1,var2,mod1,choice,cast_var1,cast_var2)

    fig = cecl.plotly_pf(nco_fct,choice)
    
    return fig

@app.callback(Output('bal_plot','figure'), [Input('submit-btn','n_clicks')])
def bal_plot(figure):
    grow_assump = output['grow_assump']
    amort_type = output['amort']
    grow_pct = output['grow_pct']
    rate = '5'
    life = output['life']
    
    # Balance runoff
    bal = cecl.amort_bal(life, grow_assump, grow_pct, amort_type, rate)
    
    fig = cecl.plotly_bal(bal,life)
    
    return fig

@app.callback(Output('prov_plot','figure'), [Input('submit-btn','n_clicks')])
def prov_plot(figure):
    port = output['portfolio']
    size = output['size']
    sdate = output['start']
    var1 = output['v1']    
    var2 = output['v2']   
    cycle = output['cycle']
    mod_type1 = output['model1']
    mod_type2 = output['model2']
    ma_len1 = output['ma_val1']
    ma_len2 = output['ma_val2']
    choice = output['foresight']
    grow_assump = output['grow_assump']
    amort_type = output['amort']
    grow_pct = output['grow_pct']
    rate = '5'
    rs_len = output['rs']
    dt_len = output['window']
    scen = output['scenario']
    life = output['life']
    
    
    # Pull data from API
    data = cecl.pull_data(port, size, sdate)
    
    # Fit model
    mod1 = cecl.mod_ols(var1,var2,data)
    
    # Prepare CCAR data
    ccar_data = cecl.load_ccar_data(cycle,var1,var2,data,mod1)    

    # Prepare macro variable data
    mac_data = cecl.macro_data(var1,var2,data,ccar_data)
    mac_cast = cecl.macro_forecast(ccar_data,var1,var2)  
    
    # Macrovar forecast
    cast_var1 = cecl.gen_cast_var(var1,mod_type1,ma_len1,mac_data,mac_cast,data,ccar_data)
    cast_var2 = cecl.gen_cast_var(var2,mod_type2,ma_len2,mac_data,mac_cast,data,ccar_data)
  
    # NCO forecast
    nco_fct = cecl.nco_cast(mac_cast,var1,var2,mod1,choice,cast_var1,cast_var2)
    
    # Balance runoff
    bal = cecl.amort_bal(life, grow_assump, grow_pct, amort_type, rate)
    
    # Generate results
    result = cecl.cecl_calc(rs_len,choice,bal,dt_len,data,port,mod1,nco_fct,cycle)
    result = cecl.nco_calc(scen, cycle, mod1,result,bal)
    result = cecl.provision_calc(result)  

    prov_plot = cecl.provision_plotly(result)
    
#    plotly_prov = tls.mpl_to_plotly(prov_plot)
#    plotly_prov['layout'].update({
#            'plot_bgcolor':'rgba(0,0,0,0)',
#            'paper_bgcolor':'rgba(0,0,0,0)'
#            })
    
    return prov_plot



    

if __name__ == '__main__':
    app.run_server()
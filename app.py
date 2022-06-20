import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import numpy as np



dest="oscfiles/"
import pandas as pd
from glob import glob
files = sorted(glob(dest+'*.xlsx'))
import warnings
warnings.simplefilter("ignore")
Volt=[]
na=[]

ll=np.asarray(['Voltage','Current','Power'])
option = st.selectbox(
     'Which Oscillations plot you like?',
     ll)

#st.write('You selected:', option)

#st.write('you selected',np.argwhere(ll==option)[0])

for fil in files:
    Data=pd.read_excel(fil,engine='openpyxl')
    
    try:
        Volt.append(Data[Data.columns[int(np.argwhere(ll==option))+2]])
        na.append(Data.iloc[0][1])
    except:
        continue


import numpy as np
df2 = pd.DataFrame()
df2["time"] = pd.to_datetime(Data[Data.columns[0]])

#missing_range = [(400,430),(500,700),(1300,1420),(2100,2230)]
#for start,end in missing_range:
#    df2.iloc[start:end,1] = np.nan

import plotly.graph_objects as go
fig = go.Figure()
for r in range (1,len(Volt)):
    df2[str(na[r])] = Volt[r]/np.max(Volt[r])
    fig.add_trace(go.Scatter(x=df2["time"], y=df2[str(na[r])],
                      mode='lines',
                      name=str(na[r])))

    
    fig.update_layout(
    autosize=True,
    yaxis=dict(
        title_text="Amplitude (PU)",
        titlefont=dict(size=30),),
    #yaxis2=dict(title='Freq',overlaying='y',side='right',titlefont=dict(size=30),),
    xaxis=dict(
        title_text="Time",
        titlefont=dict(size=30),

    ),
  )
fig.update_yaxes(automargin=True)
st.header(option+" plot")
st.plotly_chart(fig)
#fig.show()
df2=df2.fillna(0)
df2.to_csv(dest+"data2.csv", index=False)
st.download_button(
     label="Download data as CSV",
     data=df2.to_csv().encode('utf-8'),
     file_name='data.csv',
     mime='text/csv',
 )

import matplotlib.pyplot as plt
import scipy.fft

fig = go.Figure()
ff=[]
AA=[]
for r in range (1,df2.shape[1]):
  f=[]
  A=[]
  s=[]
  for k in range (10*3+2):
    data1=df2.iloc[int(df2.shape[0]*0.1*0.3)*k:int(df2.shape[0]*0.1*0.3)*(k+1),r].values
    A_signal_fft = scipy.fft.fft(data1)
    A_signal_fft=np.sqrt(A_signal_fft.real**2+A_signal_fft.imag**2)
    frequencies = scipy.fft.fftfreq(int(df2.shape[0]*0.1*0.3), 1/25)
    df3 = pd.DataFrame()
    df3['freq']=np.abs(frequencies[1:])
    df3['amp']=np.abs(A_signal_fft)[1:]/int(df2.shape[0]*0.1*0.3)
    if(len(df3[df3['amp']==df3['amp'].max()]['freq'].values)>0 and df3[df3['amp']==df3['amp'].max()]['freq'].values[0] >0 and df3[df3['amp']==df3['amp'].max()]['freq'].values[0]<1):
      f.append(np.round(df3[df3['amp']==df3['amp'].max()]['freq'].values[0],2)) 
      A.append(df3[df3['amp']==df3['amp'].max()]['amp'].values[0])
      s.append(k*df3.shape[0]*0.04)
    else:
      f.append(0)
      A.append(0)
   f=np.asarray(f)
   A=np.asarray(A)
   ff.append(f)
   AA.append(A)
  fig.add_trace(go.Scatter(x=s, y=A,
                        mode='lines+text',yaxis='y1',
                        text=f,
                        name=str(na[r])))
  #fig.add_trace(go.Scatter(x=s, y=f,mode='markers',yaxis='y2',))
fig.update_layout(
    autosize=True,
    #width=1500,
    #height=800,
    yaxis=dict(
        title_text="Amplitude (PU)",
        titlefont=dict(size=30),),
    #yaxis2=dict(title='Freq',overlaying='y',side='right',titlefont=dict(size=30),),
    xaxis=dict(
        title_text="Time (Sec)",
        titlefont=dict(size=30),

    ),
  )
    

fig.update_yaxes(automargin=True)
#fig.show()
st.header("Frequency-Time")
st.plotly_chart(fig)

df4=pd.DataFrame()
ff=np.asarray(ff)
AA=np.asarray(AA)
fig = go.Figure()

for k in range (10):
  df4['freq']=ff[:,k]
  df4['amp']=AA[:,k]


  fig.add_trace(go.Scatter(x=df2.columns[2:], y=df4['amp'],
                          mode='lines+text',yaxis='y1',
                          text=ff[:,k],
                          name='sample' +str(k)))
    #fig.add_trace(go.Scatter(x=s, y=f,mode='markers',yaxis='y2',))
fig.update_layout(
    autosize=True,
    #width=1500,
    #height=800,
    yaxis=dict(
        title_text="Amplitude (PU)",
        titlefont=dict(size=30),),
    #yaxis2=dict(title='Freq',overlaying='y',side='right',titlefont=dict(size=30),),
    xaxis=dict(
        title_text="stations",
        titlefont=dict(size=30),

    ),
  )
    

fig.update_yaxes(automargin=True)
st.header("plant wise plot")
st.plotly_chart(fig)

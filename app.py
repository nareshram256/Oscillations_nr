import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import numpy as np

import os


dest="oscfiles/"
dest1="databse/"
import pandas as pd
from glob import glob
if st.button('clear the data'):
    files = sorted(glob(dest+'*.xlsx'))
    if len(files)>0:
        for fil in files:
            os.remove(fil)
import warnings
warnings.simplefilter("ignore")
Volt=[]
na=[]

spectras = st.file_uploader("upload file", type={"xlsx"},accept_multiple_files = True)
for spectra in spectras:
    if spectra is not None:
        with open(os.path.join(dest,str(spectra.name)),"wb") as f:
            f.write((spectra).getbuffer())
        with open(os.path.join(dest1,str(spectra.name)),"wb") as f:
            f.write((spectra).getbuffer())    
    else:
        st.write("Upload excel files")


        

#st.write('You selected:', option)

#st.write('you selected',np.argwhere(ll==option)[0])
files = sorted(glob(dest+'*.xlsx'))

Data=pd.read_excel(files[0],engine='openpyxl')
ll=np.asarray(Data.columns[0:])
option = st.selectbox(
'Which Oscillations plot you like?',
 ll)
if len(files)>0:
    for fil in files:
        Data=pd.read_excel(fil,engine='openpyxl')
        try:
            #st.write(Data[ll].values)
            #Volt.append(Data[Data.columns[int(np.argwhere(ll==option))]])
            Volt.append(Data.iloc[:,int(np.argwhere(ll==option))])
            na.append(Data.iloc[0][1])
        except:
            continue

    #st.write(Volt)
    import numpy as np
    df2 = pd.DataFrame()
    df2["time"] = pd.to_datetime(Data[Data.columns[0]])
    #missing_range = [(400,430),(500,700),(1300,1420),(2100,2230)]
    #for start,end in missing_range:
    #    df2.iloc[start:end,1] = np.nan
    #st.write(na)
    import plotly.graph_objects as go
    fig = go.Figure()
    for r in range (0,len(Volt)):
        df2[na[r]] = Volt[r]/np.max(Volt[r])
        #st.write(Volt[r],np.max(Volt[r]))
        fig.add_trace(go.Scatter(x=df2["time"], y=df2[na[r]],
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
    st.header(" Description of data ")
    st.write(df2.describe())
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
    sample= st.select_slider(
     'Select sample length',
     options=[100, 200, 400, 800, 1000, 1200, 1400])
    st.write('chosen sample length is', sample)
    #st.write("max frequency detected would be %f Hz"%(25/options))
    for r in range (1,df2.shape[1]):
      f=[]
      A=[]
      s=[]
      for k in range (int(df2.shape[0]/int(sample))-1):
        data1=df2.iloc[int(sample)*k:int(sample)*(k+1),r].values
        A_signal_fft = scipy.fft.fft(data1)
        #A_signal_fft=np.sqrt(A_signal_fft.real**2+A_signal_fft.imag**2)
        frequencies = scipy.fft.fftfreq(int(sample), 1/(25))
        df3 = pd.DataFrame()
        df3['freq']=np.abs(frequencies[1:])
        df3['amp']=np.abs(A_signal_fft)[1:]/int(sample)
        #st.write(df3[(df3['freq']>1/(int(sample)*0.04)) & (df3['freq']<1)]['amp'].max())
        dummy=df3[(df3['freq']>1/(int(sample)*0.04)) & (df3['freq']<1)]['amp'].max()
        #if(len(df3[df3['amp']==df3['amp'].max()]['freq'].values)>0 and df3[df3['amp']==df3['amp'].max()]['freq'].values[0] >1/(int(sample)*0.04) and df3[df3['amp']==df3['amp'].max()]['freq'].values[0]<1):
        if(len(df3[df3['amp']==dummy]['freq'].values)>0):
          f.append(np.round(df3[df3['amp']==dummy]['freq'].values[0],2)) 
          A.append(dummy)
          #st.write(int(sample)*(k+1),df2["time"])
          s.append(df2.iloc[int(sample)*(k+1),0])
        else:
          #st.write(df2.columns[r])
          #st.write(df3[df3['amp']==df3['amp'].max()]['freq'].values)
          f.append(0)
          A.append(0)
          s.append(df2.iloc[int(sample)*(k+1),0])
      f=np.asarray(f)
      A=np.asarray(A)
      ff.append(f)
      AA.append(A)
      fig.add_trace(go.Scatter(x=s, y=A,
                            mode='lines+text',yaxis='y1',
                            text=f,
                            name=str(df2.columns[r])))
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
            title_text="Time ",
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

    for k in range (int(0.5*df2.shape[0]/int(sample))):
      df4['freq']=ff[:,k*2]
      df4['amp']=AA[:,k*2]


      fig.add_trace(go.Scatter(x=df2.columns[1:], y=df4['amp'],
                              mode='lines+text',yaxis='y1',
                              text=ff[:,k*2],
                              name='sample' +str(k*2)))
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
   
    st.sidebar.latex(r'''
     Y(t)=\sum_{k=0}^{n-1} c^k e^{(a[k]*t+ib[k]*t)}
     ''')
    if st.button('custom data'):
        modes = st.sidebar.number_input('insert number for modes')
        c=[]
        a=[]
        b=[]
        for run in range (int(modes)):
            c.append(st.sidebar.number_input('insert number for c'))
        for run in range (int(modes)):
            a.append(st.sidebar.number_input('insert number for a'))
        for run in range (int(modes)):
            b.append(st.sidebar.number_input('insert number for b'))    
    else:
        modes=4
        a=[-0.01,0.001,0,-0.01]
        b=[0,0.3,0.4,0.2]
        c=[1,0.2,0.1,0.01]
    Y=[]
    for n in range (3000):
        A=0
        for i in range (modes):
            A+=c[i]*np.exp((a[i]*t+1j*b[i]*t))
    Y.append(A)
    df5=pd.DataFrame()
    df5['tme']=np.arange(0,3000,1)
    df5['amp']=np.asarray(Y).real
    df5=df5.fillna(0)
    st.sidebar.download_button(
         label="Download custom data as CSV",
         data=df5.to_csv().encode('utf-8'),
         file_name='data.csv',
         mime='text/csv',
     )
    st.sidebar.button("Developped by NR LDC")

"""
==================================================================================================
Author: Surisetti Naresh Ram, naresh.ram@posoco.in
Note:
    the data and the analysis to be used fairly.
==================================================================================================
"""


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
import os
st.title("Oscillations Frequency  Est.  Tool")
st.sidebar.latex(r'''
     Y(t)=\sum_{k=0}^{n-1} c^k e^{(a[k]*t+ib[k]*t)}
 ''')
if st.sidebar.number_input('insert 1 for custom'):
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
    modes=3
    a=[0,-0.009,-0.005]
    b=[0,0.6,0.9]
    c=[1,0.4,0.8]
#st.sidebar.write('a=',list(a))
#st.sidebar.write(list(b))
check=0
Y=[]
q=[]
datetime_object = datetime.now()
for t in range (3000):
    A=0
    for i in range (modes):
        A+=c[i]*np.exp((a[i]*t/25+1j*b[i]*2*3.14*t/25))
    Y.append(np.abs(A))
    time_change = timedelta(seconds=0.04)
    datetime_object=datetime_object+time_change
    q.append(datetime_object)
df5=pd.DataFrame()
df5['tme']=np.asarray(q)
df5['amp']=np.asarray(Y)
df5=df5.fillna(0)
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data
df_xlsx = to_excel(df5)
st.sidebar.download_button(
     label="Download default oscillations data as xlsx",
     data=df_xlsx,
     file_name='df_eqn.xlsx',
     mime='xlsx/csv',
 )



dest="oscfiles/"
dest1="databse/"
June="old/"

import pandas as pd
from glob import glob
if st.button('clear the data'):
    files = sorted(glob(dest+'*.xlsx'))
    if len(files)>0:
        for fil in files:
            os.remove(fil)
import warnings
warnings.simplefilter("ignore")


check=st.number_input('1: default/June data    \t    2: Custom Upload')
if(check==2):
    try:
        files = sorted(glob(dest+'*.xlsx'))
        if len(files)>0:
            for fil in files:
                os.remove(fil)
    except:
        pass
    spectras = st.file_uploader("upload file", type={"xlsx"},accept_multiple_files = True)
    for spectra in spectras:
        if spectra is not None:
            with open(os.path.join(dest,str(spectra.name)),"wb") as f:
                f.write((spectra).getbuffer())
            with open(os.path.join(dest1,str(spectra.name)),"wb") as f:
                f.write((spectra).getbuffer())    
        else:
            st.write("Upload excel files")

    try:
      files = sorted(glob(dest+'*.xlsx'))
    except:
      files = sorted(glob(dest+'*.csv'))
elif (check==1):   
    files = sorted(glob(June+'*.xlsx'))

    
#st.write('You selected:', option)

#st.write('you selected',np.argwhere(ll==option)[0])

Volt=[]
na=[]
try:
    if(len(files)>0):
        try:
          Data=pd.read_excel(files[0],engine='openpyxl')
        except:
          Data=pd.read_csv(files[0],engine='openpyxl')
        ll=np.asarray(Data.columns[1:])
        option = st.selectbox(
        'Which Oscillations plot you like?',
         ll)
        for fil in files:
            Data=pd.read_excel(fil,engine='openpyxl')
            try:
                Volt.append(Data.iloc[:,int(np.argwhere(ll==option))+1])
                na.append(Data.iloc[0][1])
            except:
                continue

        import numpy as np
        #st.write(len(Volt))
        df2 = pd.DataFrame()
        df2["time"] = pd.to_datetime(Data[Data.columns[0]])

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
        #df2.to_csv(dest+"data2.csv", index=False)
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
        L_A=[]
        L_f=[]
        L_ph=[]
        #st.write("max frequency detected would be %f Hz"%(25/options))
        sample= st.select_slider(
         'Select sample length',
          options=[100, 200, 400, 800, 1000, 1200, 1400, 1600,1800,2000,2200,2400,2600,2800])
        st.write('chosen sample length is', sample)
        
        for r in range (1,df2.shape[1]):
          f=[]
          A=[]
          s=[]
          #my_f=[]
          data2=df2.iloc[:,r].values
          A_signal_fft = scipy.fft.fft(data2)
          frequencies = scipy.fft.fftfreq(df2.shape[0], 1/(25))
          df3 = pd.DataFrame()
          #for u in range (int(data2.shape[0]/2)):
          my_f=np.arctan2(A_signal_fft.imag,A_signal_fft.real)*180/3.14
          #df3['freq']=np.abs(np.gradient(my_f)/(2*3.14)) 
          df3['freq']=np.abs(frequencies[0:int(df2.shape[0]/2)])
          df3['amp']=2*np.abs(A_signal_fft[0:int(df2.shape[0]/2)])/int(df2.shape[0])
          df3['phase']=my_f[0:int(df2.shape[0]/2)]
          st.write(df3.T)
          dummy1=df3[(df3['freq']>0) & (df3['freq']<2)]['amp']
          dummy2=df3[(df3['freq']>0) & (df3['freq']<2)]['freq']
          dummy3=df3[(df3['freq']>0) & (df3['freq']<2)]['phase']
          L_A.append(dummy1.values)
          L_f.append( df3['freq'].values)
          L_ph.append(dummy3.values)
          for k in range (int(df2.shape[0]/int(sample))-1):
            data1=df2.iloc[int(sample)*k:int(sample)*(k+1),r].values
            A_signal_fft = scipy.fft.fft(data1)
            #A_signal_fft=np.sqrt(A_signal_fft.real**2+A_signal_fft.imag**2)
            frequencies = scipy.fft.fftfreq(int(sample), 1/(25))
            df3 = pd.DataFrame()
            df3['freq']=np.abs(frequencies[:])
            df3['amp']=np.abs(A_signal_fft)[:]/int(sample)
            if(k==0):
               st.write(df2.columns[r]) 
               st.write(df3.T)
            #dummy=df3[(df3['freq']>1/(int(sample)*0.04)) & (df3['freq']<1)]['amp'].max()
            dummy=df3[(df3['freq']>0) & (df3['freq']<2)]['amp'].max()
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
          fig.add_trace(go.Scatter(x=f, y=A,
                                mode='lines+text',yaxis='y1',
                                text=f,
                                name=str(df2.columns[r])))
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
        #fig.update_xaxes(type="log")
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
                title_text="Amplitude (pu)",
                titlefont=dict(size=30),),
            #yaxis2=dict(title='Freq',overlaying='y',side='right',titlefont=dict(size=30),),
            xaxis=dict(
                title_text="stations",
                titlefont=dict(size=30),

            ),
          )


        fig.update_xaxes(automargin=True)
        #fig.update_yaxes(type="log")
        st.header("plant wise plot")
        st.plotly_chart(fig)
        
        
        #st.write(L_f.shape)
        fig = go.Figure()
        L_f=np.asarray(L_f)
        L_A=np.asarray(L_A)
        for r in range (0,len(df2.columns)-1):
            fig.add_trace(go.Scatter(x=L_f[r], y=L_A[r],
                                mode='lines',yaxis='y1',
                                name=str(df2.columns[r+1])))
        fig.update_layout(
            autosize=True,
            #width=1500,
            #height=800,
            yaxis=dict(
                title_text="Amplitude (pu)",
                titlefont=dict(size=30),),
            #yaxis2=dict(title='Freq',overlaying='y',side='right',titlefont=dict(size=30),),
            xaxis=dict(
                title_text="Freq ",
                titlefont=dict(size=30),

            ),
          )

        fig.update_yaxes(type="log")
        #fig.update_yaxes(automargin=True)
        st.header("Sepctral-Graph")
        st.plotly_chart(fig)  
        
        fig = go.Figure()
        L_ph=np.asarray(L_ph)
        for r in range (0,len(df2.columns)-1):
            fig.add_trace(go.Scatter(x=L_f[r], y=L_ph[r],
                                mode='lines',yaxis='y1',
                                name=str(df2.columns[r+1])))
        fig.update_layout(
            autosize=True,
            #width=1500,
            #height=800,
            yaxis=dict(
                title_text="phase (degree)",
                titlefont=dict(size=30),),
            #yaxis2=dict(title='Time',overlaying='y',side='right',titlefont=dict(size=30),),
            xaxis=dict(
                title_text="Time ",
                titlefont=dict(size=30),

            ),
          )

        #fig.update_yaxes(type="log")
        fig.update_yaxes(automargin=True)
        st.header("phase-Graph")
        st.plotly_chart(fig)  
        
except:
    pass

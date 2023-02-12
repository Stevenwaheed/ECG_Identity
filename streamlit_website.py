import streamlit as st
import joblib
import pandas as pd
import numpy as np
import heartpy
import neurokit2 as nk

model_path = ''


'''
    this function  takes four parameters point1: (X1, Y1)(an ECG peak), point2: (X2, Y2)(an ECG peak) 
    to get the distances between ecg peaks, and return the distance between these two peaks.
    {PR Distances, PQ Distances, QS Distances, ST Distances, RT Distances}
'''

def Distances(X1, Y1, X2, Y2):
    distances_results = []
    
    for x1, y1, x2, y2 in zip(X1, Y1, X2, Y2):
        result = np.math.sqrt((x2 - x1)**2 + (y2-y1)**2)
        distances_results.append(result)
        
    return distances_results



 
'''
    this function takes four parameters point1: (X1, Y1)(an ECG peak), point2: (X2, Y2)(an ECG peak) 
    to get the slope of the lines between ecg peaks, and return the slope between these two peaks.
    {PR Slope, PQ Slope, QS Slope, ST Slope, RT Slope}
'''

def Slope(X1, Y1, X2, Y2):
    slope_results = []
    
    for x1, y1, x2, y2 in zip(X1, Y1, X2, Y2):
        result = (y2 - y1) / (x2 - x1)
        slope_results.append(result)
        
    return slope_results




'''
    this function takes two ECG peaks and gets the amplitudes between ecg peaks from the dataframe that we've created.
    and return the total amplitude between these two waves.
    {PR Amplitude, PQ Amplitude, QS Amplitude, ST Amplitude
    , RT Amplitude, PS Amplitude, PT Amplitude, TQ Amplitude
    ,QR Amplitude, RS Amplitude}
'''

def Amplitudes(peak1, peak2):
    amplitudes = np.abs(peak1.values - peak2.values)
    return amplitudes

def intervals(Peaks1, Peaks2):
    
    res = np.abs(Peaks2 - Peaks1)
    return res



'''
    this function takes two ECG peaks with nulls and removes the nulls from the ECG wave,
    and return the correct two ECG peaks.
'''

def remove_nulls(column1, column2):
    
    if column1 == 'ECG_R_Peaks' and column2 != 'ECG_R_Peaks':
        
        # change the NULL values with true and false, where true means it's a null value.
        # to show only the indices of nulls.
        
        TF_selection = pd.DataFrame(np.array(features[column2])).notna().values.flatten()
        new_features_col2 = pd.DataFrame(np.array(features[column2])).dropna()
        new_features_col1 = pd.DataFrame(np.array(info[column1])[TF_selection])
    
    
    if column1 != 'ECG_R_Peaks' and column2 == 'ECG_R_Peaks':
        TF_selection = pd.DataFrame(np.array(features[column1])).notna().values.flatten()
        new_features_col1 = pd.DataFrame(np.array(features[column1])).dropna()
        new_features_col2 = pd.DataFrame(np.array(info[column2])[TF_selection])
    
    
    if column1 != 'ECG_R_Peaks' and column2 != 'ECG_R_Peaks':
        TF_selection = pd.DataFrame(np.array(features[column1])).notna().values.flatten()
        new_features_col1 = pd.DataFrame(np.array(features[column1])).dropna()
        new_features_col2 = pd.DataFrame(np.array(features[column2])[TF_selection])
        
        TF_selection = new_features_col2.notna().values.flatten()
        new_features_col2 = new_features_col2.dropna()
        new_features_col1 = pd.DataFrame(np.array(new_features_col1.values.flatten())[TF_selection])
        
    return new_features_col1, new_features_col2





'''
    this function takes two ECG peaks, and return the distances, slopes and amplitudes between peaks.
'''

def get_ECG_features(column1, column2):
    

    X1 = total_df[column1]
    Y1 = signals.iloc[total_df[column1], 1]

    X2 = features[column2]
    Y2 = signals.iloc[total_df[column2], 1]
        
    # Calculate Distances
    distances = Distances(X1, Y1, X2, Y2)
    
    # Calculate Slope
    slopes = Slope(X1, Y1, X2, Y2)

    return distances, slopes


# load the model
Extra_tree = joblib.load('Extra tree banha version2.h5')

file = st.file_uploader('Upload ECG', type=['csv'])

if st.button('Predict'):
    df = pd.read_csv(file)
                
    person_name = df.columns[1]
    sample_rate = int(df.iloc[7, 1].split('.')[0])
    df.drop(person_name, inplace=True, axis=1)
    
    df.drop(range(0, 10), inplace=True)
    df['signals'] = df['Name']
    df.drop('Name', inplace=True, axis=1)
    df['signals'] = df['signals'].astype('float')

    df.dropna(inplace=True)
    
    signals, info = nk.ecg_process(df['signals'], sampling_rate=sample_rate)
    signals, info = nk.ecg_process(signals['ECG_Clean'], sampling_rate=sample_rate)
    sigs, features = nk.ecg_delineate(signals['ECG_Clean'], sampling_rate=sample_rate)
    
    total_df = pd.DataFrame(columns=['ECG_R_Peaks'])
                
    total_df['ECG_R_Peaks'] = info['ECG_R_Peaks']
    total_df['ECG_P_Peaks'] = features['ECG_P_Peaks']
    total_df['ECG_Q_Peaks'] = features['ECG_Q_Peaks']
    total_df['ECG_S_Peaks'] = features['ECG_S_Peaks']
    total_df['ECG_T_Peaks'] = features['ECG_T_Peaks']

    total_df.dropna(inplace=True)
    
    # Features between PR
    PR_distances = get_ECG_features('ECG_R_Peaks', 'ECG_P_Peaks')[0]
    PR_slopes = get_ECG_features('ECG_R_Peaks', 'ECG_P_Peaks')[1]
    PR_amplitudes = Amplitudes(total_df['ECG_R_Peaks'], total_df['ECG_P_Peaks'])

    # Features between PQ
    PQ_distances = get_ECG_features('ECG_P_Peaks', 'ECG_Q_Peaks')[0]
    PQ_slopes = get_ECG_features('ECG_P_Peaks', 'ECG_Q_Peaks')[1]
    PQ_amplitudes = Amplitudes(total_df['ECG_P_Peaks'], total_df['ECG_Q_Peaks'])

    # Features between QS
    QS_distances = get_ECG_features('ECG_Q_Peaks', 'ECG_S_Peaks')[0]
    QS_slopes = get_ECG_features('ECG_Q_Peaks', 'ECG_S_Peaks')[1]
    QS_amplitudes = Amplitudes(total_df['ECG_Q_Peaks'], total_df['ECG_S_Peaks'])

    # Features between RT
    RT_distances = get_ECG_features('ECG_R_Peaks', 'ECG_T_Peaks')[0]
    RT_slopes = get_ECG_features('ECG_R_Peaks', 'ECG_T_Peaks')[1]
    RT_amplitudes = Amplitudes(total_df['ECG_R_Peaks'], total_df['ECG_T_Peaks'])

    # Features between ST
    ST_distances = get_ECG_features('ECG_S_Peaks', 'ECG_T_Peaks')[0]
    ST_slopes = get_ECG_features('ECG_S_Peaks', 'ECG_T_Peaks')[1]
    ST_amplitudes = Amplitudes(total_df['ECG_S_Peaks'], total_df['ECG_T_Peaks'])


    # Amplitudes
    PS_amplitudes=Amplitudes(total_df['ECG_S_Peaks'], total_df['ECG_P_Peaks'])
    PT_amplitudes=Amplitudes(total_df['ECG_T_Peaks'], total_df['ECG_P_Peaks'])
    TQ_amplitudes=Amplitudes(total_df['ECG_T_Peaks'], total_df['ECG_Q_Peaks'])
    RQ_amplitudes=Amplitudes(total_df['ECG_Q_Peaks'], total_df['ECG_R_Peaks'])
    RS_amplitudes=Amplitudes(total_df['ECG_R_Peaks'], total_df['ECG_S_Peaks'])

    IQR = intervals(total_df['ECG_Q_Peaks'], total_df['ECG_R_Peaks'])
    IRS = intervals(total_df['ECG_R_Peaks'], total_df['ECG_S_Peaks'])
    IPQ = intervals(total_df['ECG_P_Peaks'], total_df['ECG_Q_Peaks'])
    IQS = intervals(total_df['ECG_Q_Peaks'], total_df['ECG_S_Peaks'])
    IPS = intervals(total_df['ECG_P_Peaks'], total_df['ECG_S_Peaks'])
    IPR = intervals(total_df['ECG_P_Peaks'], total_df['ECG_R_Peaks'])
    IST = intervals(total_df['ECG_S_Peaks'], total_df['ECG_T_Peaks'])
    IQT = intervals(total_df['ECG_Q_Peaks'], total_df['ECG_T_Peaks'])
    IRT = intervals(total_df['ECG_R_Peaks'], total_df['ECG_T_Peaks'])
    IPT = intervals(total_df['ECG_P_Peaks'], total_df['ECG_T_Peaks'])



    Extracted_Features_DF = pd.DataFrame(columns=[
        'PR Distances', 'PR Slope', 'PR Amplitude',
        'PQ Distances', 'PQ Slope', 'PQ Amplitude',
        'QS Distances', 'QS Slope', 'QS Amplitude',
        'ST Distances', 'ST Slope', 'ST Amplitude',
        'RT Distances', 'RT Slope', 'RT Amplitude',

        'PS Amplitude', 'PT Amplitude', 'TQ Amplitude',
        'QR Amplitude', 'RS Amplitude'
    ])


    lengths = [len(PR_distances), len(PR_slopes), len(PR_amplitudes)
           , len(PQ_distances), len(PQ_slopes), len(PQ_amplitudes)
           , len(QS_distances), len(QS_slopes), len(QS_amplitudes)
           , len(ST_distances), len(ST_slopes), len(ST_amplitudes)
           , len(RT_distances), len(RT_slopes), len(RT_amplitudes)
           , len(PS_amplitudes), len(PT_amplitudes), len(TQ_amplitudes)
           , len(RQ_amplitudes), len(RS_amplitudes)
           
           , len(IQR), len(IRS), len(IPQ)
           , len(IQS), len(IPS), len(IPR)
           , len(IST), len(IQT), len(IRT)
           , len(IPT)
          ]

    minimum = min(lengths) - 1

    Extracted_Features_DF['PR Distances'] = PR_distances[:minimum]
    Extracted_Features_DF['PR Slope'] = PR_slopes[:minimum]
    Extracted_Features_DF['PR Amplitude'] = PR_amplitudes[:minimum]

    Extracted_Features_DF['PQ Distances'] = PQ_distances[:minimum]
    Extracted_Features_DF['PQ Slope'] = PQ_slopes[:minimum]
    Extracted_Features_DF['PQ Amplitude'] = PQ_amplitudes[:minimum]

    Extracted_Features_DF['QS Distances'] = QS_distances[:minimum]
    Extracted_Features_DF['QS Slope'] = QS_slopes[:minimum]
    Extracted_Features_DF['QS Amplitude'] = QS_amplitudes[:minimum]

    Extracted_Features_DF['ST Distances'] = ST_distances[:minimum]
    Extracted_Features_DF['ST Slope'] = ST_slopes[:minimum]
    Extracted_Features_DF['ST Amplitude'] = ST_amplitudes[:minimum]

    Extracted_Features_DF['RT Distances'] = RT_distances[:minimum]
    Extracted_Features_DF['RT Slope'] = RT_slopes[:minimum]
    Extracted_Features_DF['RT Amplitude'] = RT_amplitudes[:minimum]

    Extracted_Features_DF['PS Amplitude'] = PS_amplitudes[:minimum]
    Extracted_Features_DF['PT Amplitude'] = PT_amplitudes[:minimum]
    Extracted_Features_DF['TQ Amplitude'] = TQ_amplitudes[:minimum]
    Extracted_Features_DF['QR Amplitude'] = RQ_amplitudes[:minimum]
    Extracted_Features_DF['RS Amplitude'] = RS_amplitudes[:minimum]


    Extracted_Features_DF['QR Interval'] = IQR[:minimum]
    Extracted_Features_DF['RS Interval'] = IRS[:minimum]
    Extracted_Features_DF['PQ Interval'] = IPQ[:minimum]
    Extracted_Features_DF['QS Interval'] = IQS[:minimum]
    Extracted_Features_DF['PS Interval'] = IPS[:minimum]
    Extracted_Features_DF['PR Interval'] = IPR[:minimum]
    Extracted_Features_DF['ST Interval'] = IST[:minimum]
    Extracted_Features_DF['QT Interval'] = IQT[:minimum]
    Extracted_Features_DF['RT Interval'] = IRT[:minimum]
    Extracted_Features_DF['PT Interval'] = IPT[:minimum]

    Extracted_Features_DF.dropna(inplace=True)
    
    preds = Extra_tree.predict(Extracted_Features_DF)
    p = pd.DataFrame(preds)
    st.text('person: '+  str(p.value_counts().index[0][0]))

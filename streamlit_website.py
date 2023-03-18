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
    
    for x1, y1, x2, y2 in zip(X1.values.flatten(), Y1, X2.values.flatten(), Y2):
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
    
    for x1, y1, x2, y2 in zip(X1.values.flatten(), Y1, X2.values.flatten(), Y2):

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
    amplitudes = np.abs(peak1 - peak2)
    return amplitudes



def intervals(Peaks1, Peaks2):
    
    res = np.abs(Peaks2 - Peaks1)
    return res



'''
    this function takes two ECG peaks with nulls and removes the nulls from the ECG wave,
    and return the correct two ECG peaks.
'''

def remove_nulls(peaks, rpeaks):
    
    if len(peaks) > len(rpeaks):
        TF_selection = pd.DataFrame(peaks[:len(rpeaks)]).notna().values.flatten()
        new_peaks = pd.DataFrame(peaks[:len(rpeaks)]).dropna()
        new_rpeaks = pd.DataFrame(rpeaks[TF_selection])
        
        return new_peaks, new_rpeaks
    
    elif len(peaks) < len(rpeaks):
        TF_selection = pd.DataFrame(peaks).notna().values.flatten()
        new_peaks = pd.DataFrame(peaks).dropna()
        rpeaks = rpeaks[:len(TF_selection)]
        new_rpeaks = pd.DataFrame(rpeaks[TF_selection])
        
        return new_peaks, new_rpeaks
    
    else:
        total_df = pd.DataFrame(columns=['ECG_R_Peaks'])
        
        total_df['ECG_R_Peaks'] = rpeaks
        total_df['ECG_Peaks'] = peaks
        
        total_df.dropna(inplace=True)
        
        return total_df['ECG_Peaks'], total_df['ECG_R_Peaks']





'''
    this function takes two ECG peaks, and return the distances, slopes and amplitudes between peaks.
'''

def get_ECG_features(peaks1, peaks2):
    
    X1 = peaks2
    Y1 = signals.iloc[peaks2.values.flatten(), 1]

    X2 = peaks1
    Y2 = signals.iloc[peaks1.values.flatten(), 1]
    
    # Calculate Distances
    distances = Distances(X1, Y1, X2, Y2)
    
    # Calculate Slope
    slopes = Slope(X1, Y1, X2, Y2)

    return distances, slopes



# load the model
Extra_tree = joblib.load('Extra tree test 11 (97).h5')

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
    df['signals'].dropna(inplace=True)

    signals, info = nk.ecg_process(df['signals'], sampling_rate=sample_rate)

    signals, info = nk.ecg_process(signals.iloc[:, 1], sampling_rate=sample_rate)

    filtered_data = heartpy.filtering.filter_signal(signals.iloc[:, 1], filtertype='bandpass', cutoff=[2.5, 40], sample_rate=sample_rate, order=3)
    corrected_data = heartpy.hampel_correcter(filtered_data, sample_rate=sample_rate)
    final_signal = np.array(filtered_data)+np.array(corrected_data)

    filtered_data2 = heartpy.filtering.filter_signal(final_signal, filtertype='bandpass', cutoff=[3, 20], sample_rate=sample_rate, order=3)
    corrected_data2 = heartpy.filtering.hampel_correcter(filtered_data2, sample_rate=sample_rate)
    final_signal2 = np.array(filtered_data2) + np.array(corrected_data2)
    
    # rpeaks = nk.ecg_findpeaks(signals.iloc[:, 1], sampling_rate=sample_rate)

    sigs, features = nk.ecg_delineate(final_signal2, sampling_rate=sample_rate, method='peak')


    p_peaks, pr_peaks = remove_nulls(features['ECG_P_Peaks'], info['ECG_R_Peaks'])
    # print(len(p_peaks), len(pr_peaks))

    q_peaks, qr_peaks = remove_nulls(features['ECG_Q_Peaks'], info['ECG_R_Peaks'])
    # print(len(q_peaks), len(qr_peaks))

    s_peaks, sr_peaks = remove_nulls(features['ECG_S_Peaks'], info['ECG_R_Peaks'])
    # print(len(s_peaks), len(sr_peaks))

    t_peaks, tr_peaks = remove_nulls(features['ECG_T_Peaks'], info['ECG_R_Peaks'])
    # print(len(t_peaks), len(tr_peaks))


    PR_distances = get_ECG_features(pr_peaks, p_peaks)[0]
    PR_slopes = get_ECG_features(pr_peaks, p_peaks)[1]
    PR_amplitudes = Amplitudes(pr_peaks.values.flatten(), p_peaks.values.flatten())

    PQ_distances = get_ECG_features(p_peaks, q_peaks)[0]
    PQ_slopes = get_ECG_features(p_peaks, q_peaks)[1]
    PQ_amplitudes=Amplitudes(np.array(features['ECG_P_Peaks']), np.array(features['ECG_Q_Peaks']))

    QS_distances = get_ECG_features(q_peaks, s_peaks)[0]
    QS_slopes = get_ECG_features(q_peaks, s_peaks)[1]
    QS_amplitudes = Amplitudes(np.array(features['ECG_Q_Peaks']), np.array(features['ECG_S_Peaks']))

    RT_distances = get_ECG_features(tr_peaks, t_peaks)[0]
    RT_slopes = get_ECG_features(tr_peaks, t_peaks)[1]
    RT_amplitudes = Amplitudes(tr_peaks.values.flatten(), t_peaks.values.flatten())

    ST_distances=get_ECG_features(s_peaks, t_peaks)[0]
    ST_slopes=get_ECG_features(s_peaks, t_peaks)[1]
    ST_amplitudes=Amplitudes(np.array(features['ECG_S_Peaks']), np.array(features['ECG_T_Peaks']))

    PS_amplitudes = Amplitudes(np.array(features['ECG_P_Peaks']), np.array(features['ECG_S_Peaks']))
    PT_amplitudes = Amplitudes(np.array(features['ECG_T_Peaks']), np.array(features['ECG_P_Peaks']))
    TQ_amplitudes = Amplitudes(np.array(features['ECG_T_Peaks']), np.array(features['ECG_Q_Peaks']))
    RQ_amplitudes = Amplitudes(q_peaks.values.flatten(), qr_peaks.values.flatten())
    RS_amplitudes = Amplitudes(sr_peaks.values.flatten(), s_peaks.values.flatten())

    QR_interval = intervals(q_peaks.values.flatten(), qr_peaks.values.flatten())
    RS_interval = intervals(sr_peaks.values.flatten(), s_peaks.values.flatten())
    PQ_interval = intervals(np.array(features['ECG_P_Peaks']), np.array(features['ECG_Q_Peaks']))
    QS_interval = intervals(np.array(features['ECG_Q_Peaks']), np.array(features['ECG_S_Peaks']))
    PS_interval = intervals(np.array(features['ECG_P_Peaks']), np.array(features['ECG_S_Peaks']))
    PR_interval = intervals(p_peaks.values.flatten(), pr_peaks.values.flatten())
    ST_interval = intervals(np.array(features['ECG_S_Peaks']), np.array(features['ECG_T_Peaks']))
    QT_interval = intervals(np.array(features['ECG_Q_Peaks']), np.array(features['ECG_T_Peaks']))
    RT_interval = intervals(tr_peaks.values.flatten(), t_peaks.values.flatten())
    PT_interval = intervals(np.array(features['ECG_P_Peaks']), np.array(features['ECG_T_Peaks']))

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

               , len(QR_interval), len(RS_interval), len(PQ_interval)
               , len(QS_interval), len(PS_interval), len(PR_interval)
               , len(ST_interval), len(QT_interval), len(RT_interval)
               , len(PT_interval)
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


    Extracted_Features_DF['QR Interval'] = RS_interval[:minimum]
    Extracted_Features_DF['RS Interval'] = RS_interval[:minimum]
    Extracted_Features_DF['PQ Interval'] = PQ_interval[:minimum]
    Extracted_Features_DF['QS Interval'] = QS_interval[:minimum]
    Extracted_Features_DF['PS Interval'] = PS_interval[:minimum]
    Extracted_Features_DF['PR Interval'] = PR_interval[:minimum]
    Extracted_Features_DF['ST Interval'] = ST_interval[:minimum]
    Extracted_Features_DF['QT Interval'] = QT_interval[:minimum]
    Extracted_Features_DF['RT Interval'] = RT_interval[:minimum]
    Extracted_Features_DF['PT Interval'] = PT_interval[:minimum]

    Extracted_Features_DF.dropna(inplace=True)
    
    preds = Extra_tree.predict(Extracted_Features_DF)
    p = pd.DataFrame(preds)
    st.text('person: '+  str(p.value_counts().index[0][0]))

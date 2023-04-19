# DataFrame packages
import numpy as np
import pandas as pd

# visualization packages
import seaborn as sns
import matplotlib.pyplot as plt

# heart packages
import wfdb
import heartpy
import neurokit2 as nk

# ML packages
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score

from sklearn.model_selection import train_test_split

# others
import os
import joblib



def Distances(X1, Y1, X2, Y2):
    distances_results = []
    
    for x1, y1, x2, y2 in zip(X1, Y1, X2, Y2):
        result = np.math.sqrt((x2 - x1)**2 + (y2-y1)**2)
        distances_results.append(result)
        
    return distances_results


def Slope(X1, Y1, X2, Y2):
    slope_results = []
    
    for x1, y1, x2, y2 in zip(X1, Y1, X2, Y2):
        result = (y2 - y1) / (x2 - x1)
        slope_results.append(result)
        
    return slope_results



def Amplitudes(peak1, peak2):
    amplitudes = np.abs(peak1.values - peak2.values)
    return amplitudes

def intervals(Peaks1, Peaks2):
    
    res = np.abs(Peaks2 - Peaks1)
    return res


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


# persons_path = 'C:\\Users\\Steven20367691\\Desktop\\persons\\'
# person_files = os.listdir(persons_path)

PR_distances = []
PR_slopes = []
PR_amplitudes = []

PQ_distances = []
PQ_slopes = []
PQ_amplitudes = []

QS_distances = []
QS_slopes = []
QS_amplitudes = []

RT_distances = []
RT_slopes = []
RT_amplitudes = []

ST_distances = []
ST_slopes = []
ST_amplitudes = []

PS_amplitudes = []
PT_amplitudes = []
TQ_amplitudes = []
RQ_amplitudes = []
RS_amplitudes = []


QR_interval_list = []
RS_interval_list = []
PQ_interval_list = []
QS_interval_list = []
PS_interval_list = []
PR_interval_list = []
ST_interval_list = []
QT_interval_list = []
RT_interval_list = []
PT_interval_list = []

PR_Amp_list = []
PQ_Amp_list = []
RT_Amp_list = []
PS_Amp_list = []
PT_Amp_list = []
QS_Amp_list = []
TQ_Amp_list = []
TS_Amp_list = []
QR_Amp_list = []
RS_Amp_list = []


label = []


path = 'C:\\Users\\Steven20367691\\Desktop\\Team ECG'

team = os.listdir(path)
for member in team:
    files = os.listdir(path+'\\'+member)
    print('--->>>', member)
    for file in files:
        if file.split('.')[1] == 'csv':
            print(file)

            
            try:
                df = pd.read_csv(path+'\\'+member+'\\'+file)
                
                person_name = df.columns[1]
                sample_rate = int(df.iloc[7, 1].split('.')[0])
                df.drop(person_name, inplace=True, axis=1)
                
                df.drop(range(0, 10), inplace=True)
                df['signals'] = df['Name']
                df.drop('Name', inplace=True, axis=1)
                df['signals'] = df['signals'].astype('float')

                df.dropna(inplace=True)
                
                
                signals, info = nk.ecg_process(df['signals'], sampling_rate=sample_rate)

                signals, info = nk.ecg_process(signals.iloc[:, 1], sampling_rate=sample_rate)
                
                filtered_data = heartpy.filtering.filter_signal(signals.iloc[:, 1], filtertype='bandpass', cutoff=[2.5, 40], sample_rate=sample_rate, order=3)
                corrected_data = heartpy.hampel_correcter(filtered_data, sample_rate=sample_rate)
                final_signal = np.array(filtered_data)+np.array(corrected_data)

                filtered_data2 = heartpy.filtering.filter_signal(final_signal, filtertype='bandpass', cutoff=[3, 20], sample_rate=sample_rate, order=3)
                corrected_data2 = heartpy.filtering.hampel_correcter(filtered_data2, sample_rate=sample_rate)
                final_signal2 = np.array(filtered_data2) + np.array(corrected_data2)
                
                sigs, features = nk.ecg_delineate(final_signal2, sampling_rate=sample_rate, method='peak')
                
                total_df = pd.DataFrame(columns=['ECG_R_Peaks'])
                
                
                
                total_df['ECG_R_Peaks'] = info['ECG_R_Peaks']
                total_df['ECG_P_Peaks'] = features['ECG_P_Peaks']
                total_df['ECG_Q_Peaks'] = features['ECG_Q_Peaks']
                total_df['ECG_S_Peaks'] = features['ECG_S_Peaks']
                total_df['ECG_T_Peaks'] = features['ECG_T_Peaks']
                
                total_df.dropna(inplace=True)
                
                PR_distances.extend(get_ECG_features('ECG_R_Peaks', 'ECG_P_Peaks')[0])
                PR_slopes.extend(get_ECG_features('ECG_R_Peaks', 'ECG_P_Peaks')[1])
                PR_amplitudes.extend(Amplitudes(total_df['ECG_R_Peaks'], total_df['ECG_P_Peaks']))

                PQ_distances.extend(get_ECG_features('ECG_P_Peaks', 'ECG_Q_Peaks')[0])
                PQ_slopes.extend(get_ECG_features('ECG_P_Peaks', 'ECG_Q_Peaks')[1])
                PQ_amplitudes.extend(Amplitudes(total_df['ECG_P_Peaks'], total_df['ECG_Q_Peaks']))

                QS_distances.extend(get_ECG_features('ECG_Q_Peaks', 'ECG_S_Peaks')[0])
                QS_slopes.extend(get_ECG_features('ECG_Q_Peaks', 'ECG_S_Peaks')[1])
                QS_amplitudes.extend(Amplitudes(total_df['ECG_Q_Peaks'], total_df['ECG_S_Peaks']))

                RT_distances.extend(get_ECG_features('ECG_R_Peaks', 'ECG_T_Peaks')[0])
                RT_slopes.extend(get_ECG_features('ECG_R_Peaks', 'ECG_T_Peaks')[1])
                RT_amplitudes.extend(Amplitudes(total_df['ECG_R_Peaks'], total_df['ECG_T_Peaks']))

                ST_distances.extend(get_ECG_features('ECG_S_Peaks', 'ECG_T_Peaks')[0])
                ST_slopes.extend(get_ECG_features('ECG_S_Peaks', 'ECG_T_Peaks')[1])
                ST_amplitudes.extend(Amplitudes(total_df['ECG_S_Peaks'], total_df['ECG_T_Peaks']))

            
                PS_amplitudes.extend(Amplitudes(total_df['ECG_S_Peaks'], total_df['ECG_P_Peaks']))
                PT_amplitudes.extend(Amplitudes(total_df['ECG_T_Peaks'], total_df['ECG_P_Peaks']))
                TQ_amplitudes.extend(Amplitudes(total_df['ECG_T_Peaks'], total_df['ECG_Q_Peaks']))
                RQ_amplitudes.extend(Amplitudes(total_df['ECG_Q_Peaks'], total_df['ECG_R_Peaks']))
                RS_amplitudes.extend(Amplitudes(total_df['ECG_R_Peaks'], total_df['ECG_S_Peaks']))
            
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
                
                QR_interval_list.extend(IQR)
                RS_interval_list.extend(IRS)
                PQ_interval_list.extend(IPQ)
                QS_interval_list.extend(IQS)
                PS_interval_list.extend(IPS)
                PR_interval_list.extend(IPR)
                ST_interval_list.extend(IST)
                QT_interval_list.extend(IQT)
                RT_interval_list.extend(IRT)
                PT_interval_list.extend(IPT)

                label.extend([member]*len(get_ECG_features('ECG_S_Peaks', 'ECG_T_Peaks')[0]))
                    
                    
            except:
                print('error..')
                pass


Extracted_Features_DF = pd.DataFrame(columns=[
    'PR Distances', 'PR Slope', 'PR Amplitude',
    'PQ Distances', 'PQ Slope', 'PQ Amplitude',
    'QS Distances', 'QS Slope', 'QS Amplitude',
    'ST Distances', 'ST Slope', 'ST Amplitude',
    'RT Distances', 'RT Slope', 'RT Amplitude',

    'PS Amplitude', 'PT Amplitude', 'TQ Amplitude',
    'QR Amplitude', 'RS Amplitude'
])


Extracted_Features_DF['PR Distances'] = PR_distances
Extracted_Features_DF['PR Slope'] = PR_slopes
Extracted_Features_DF['PR Amplitude'] = PR_amplitudes

Extracted_Features_DF['PQ Distances'] = PQ_distances
Extracted_Features_DF['PQ Slope'] = PQ_slopes
Extracted_Features_DF['PQ Amplitude'] = PQ_amplitudes

Extracted_Features_DF['QS Distances'] = QS_distances
Extracted_Features_DF['QS Slope'] = QS_slopes
Extracted_Features_DF['QS Amplitude'] = QS_amplitudes

Extracted_Features_DF['ST Distances'] = ST_distances
Extracted_Features_DF['ST Slope'] = ST_slopes
Extracted_Features_DF['ST Amplitude'] = ST_amplitudes

Extracted_Features_DF['RT Distances'] = RT_distances
Extracted_Features_DF['RT Slope'] = RT_slopes
Extracted_Features_DF['RT Amplitude'] = RT_amplitudes

Extracted_Features_DF['PS Amplitude'] = PS_amplitudes
Extracted_Features_DF['PT Amplitude'] = PT_amplitudes
Extracted_Features_DF['TQ Amplitude'] = TQ_amplitudes
Extracted_Features_DF['QR Amplitude'] = RQ_amplitudes
Extracted_Features_DF['RS Amplitude'] = RS_amplitudes


Extracted_Features_DF['QR Interval'] = RS_interval_list
Extracted_Features_DF['RS Interval'] = RS_interval_list
Extracted_Features_DF['PQ Interval'] = PQ_interval_list
Extracted_Features_DF['QS Interval'] = QS_interval_list
Extracted_Features_DF['PS Interval'] = PS_interval_list
Extracted_Features_DF['PR Interval'] = PR_interval_list
Extracted_Features_DF['ST Interval'] = ST_interval_list
Extracted_Features_DF['QT Interval'] = QT_interval_list
Extracted_Features_DF['RT Interval'] = RT_interval_list
Extracted_Features_DF['PT Interval'] = PT_interval_list

Extracted_Features_DF['Person'] = label


Extracted_Features_DF.to_csv('C:\\Users\\Steven20367691\\Desktop\\ecg.csv')
print(Extracted_Features_DF)
# split the data
df = Extracted_Features_DF.dropna()

X = df.iloc[:, :-1]
y = df['Person']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


ExtraTree = ExtraTreesClassifier(n_estimators=200, criterion='entropy', verbose=2)


ExtraTree.fit(X_train, y_train)

preds = ExtraTree.predict(X_test)

print(preds)

# ExtraTree model
ExtraTree_preds = ExtraTree.predict(X_test)
print('accuracy_score:', accuracy_score(ExtraTree_preds, y_test.values))
print('f1_score:', f1_score(y_test.values, ExtraTree_preds, average='weighted'))
print('recall_score:', recall_score(ExtraTree_preds, y_test.values, average='weighted'))
print('precision_score:', precision_score(ExtraTree_preds, y_test.values, average='weighted'))



# model_path = 'C:\\Users\\Steven20367691\\Desktop\\'

# Save the model
joblib.dump(ExtraTree, 'Extra tree.h5')

print(y.value_counts())

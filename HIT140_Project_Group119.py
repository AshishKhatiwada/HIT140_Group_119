import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st


# step-1 Define column names based on the provided description
column_names = [
    "SubjectID", "Jitter%", "Jitter_Abs", "Jitter_RAP", "Jitter_PPQ5",
    "Jitter_DDP", "Shimmer%", "Shimmer_Abs", "Shimmer_APQ3", "Shimmer_APQ5",
    "Shimmer_APQ11", "Shimmer_DDA", "Harmonicity_AutoCorr", "Harmonicity_NHR",
    "Harmonicity_HNR", "Pitch_Median", "Pitch_Mean", "Pitch_StdDev",
    "Pitch_Min", "Pitch_Max", "Pulse_Num", "Pulse_Periods", "Pulse_MeanPeriod",
    "Pulse_StdDevPeriod", "Voice_FractionUnvoiced", "Voice_NumVoiceBreaks",
    "Voice_DegreeVoiceBreaks", "UPDRS", "PD_Indicator"
]
# step-2 Data Loading and Inspection
raw_data = pd.read_csv('po1_data.txt', sep=',', header=None, names=column_names)
# print(data.head())

# step-3 Data Preprocessing: Handle missing values 
data = raw_data.dropna()


# step-4 Descriptive Analysis: Display summary statistics for each feature
desAnalysisPD = data.groupby('PD_Indicator').describe()
print(desAnalysisPD)

# step-5 Inferential Statistical Analysis

# Split data into PD and non-PD groups
pd_data = data[data['PD_Indicator'] == 1] 
non_pd_data = data[data['PD_Indicator'] == 0]
# print people with pd
print(pd_data)
# print people with non-pd
print(non_pd_data)


# step-6 Visualization:Create box plots for selected features
significant_features = []
for col in column_names[1:-2]:  # Exclude Subject ID and PD Indicator
    t_stat, p_value = st.ttest_ind(pd_data[col], non_pd_data[col])
    if p_value < 0.05:
        significant_features.append(col)
        print(f"Feature '{col}': p-value = {p_value:.4f} (salient)")
    else:
        print(f"Feature '{col}': p-value = {p_value:.4f}")

    
    
# Visualize significant features
plt.figure(figsize=(10, 6))
pd_data[significant_features].boxplot()
plt.title("Boxplot of Significant Features for PD Group")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


    


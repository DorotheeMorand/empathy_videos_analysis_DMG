

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import spearmanr

# data from participant dyads (video creation study)

# to do: avoid doing everything 2x (both participants) .. how?

    # https://matplotlib.org/stable/plot_types/index.html
    # https://numpy.org/doc/stable/reference/generated/numpy.interp.html
    # https://docs.scipy.org/doc/scipy/reference/interpolate.html 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

########## open data, identify peaks, basic graph ##################################################################################

# open data
with open(r"P22_shame_P22_peaks.json", 'r') as data_p1:
    ecg_data_p1 = json.load(data_p1)  # data participant 1
with open(r"P22_shame_P29_peaks.json", 'r') as data_p2:
    ecg_data_p2 = json.load(data_p2)  # data participant 2

# does_overlap - from Floris biotop
def does_overlap(intv1, intv2):
    # Return whether two intervals overlap
    # The intervals are given as (start, end) tuples.
    (a1, b1) = intv1
    if a1 > b1: (a1, b1) = (b1, a1)  # if coded inversely, flip
    (a2, b2) = intv2
    if a2 > b2: (a2, b2) = (b2, a2)
    # So now a1 <= b1 and a2 <= b2
    overlapmin = max([a1, a2])
    overlapmax = min([b1, b2])
    return overlapmin <= overlapmax

# exclude intervals overlapping with invalid areas
def exclude_invalid(peak_times, invalid_areas):
    valid_intervals = []
    for i in range(len(peak_times) - 1): # -1 so it does not go out of range (for intervals)
        start_time = peak_times[i] # peak 1
        end_time = peak_times[i + 1] # peak 2
        interval = (start_time, end_time)
        if not any(does_overlap(interval, invalid_area) for invalid_area in invalid_areas):
            valid_intervals.append(interval)
    return valid_intervals

# identify peak_times
peak_times_p1 = [data['t'] for data in ecg_data_p1['peaks']['ECG, X, RSPEC-R']] # modify here if needed
peak_times_p1.sort()  
peak_times_p2 = [data['t'] for data in ecg_data_p2['peaks']['ECG, X, RSPEC-R.1']] # modify here if needed
peak_times_p2.sort()  

# identify invalid_areas
invalid_areas_p1 = [(data[1], data[2]) for data in ecg_data_p1['invalid']['ECG, X, RSPEC-R']] # modify here if needed
invalid_areas_p2 = [(data[1], data[2]) for data in ecg_data_p2['invalid']['ECG, X, RSPEC-R.1']] # modify here if needed

# exclude invalid intervals
valid_intervals_p1 = exclude_invalid(peak_times_p1, invalid_areas_p1)
valid_intervals_p2 = exclude_invalid(peak_times_p2, invalid_areas_p2)

# identify R-R intervals 
time_diffs_p1 = [end - start if ecg_data_p1['peaks']['ECG, X, RSPEC-R'][i]['valid'] and
                  ecg_data_p1['peaks']['ECG, X, RSPEC-R'][i-1]['valid'] else None
                  for i, (start, end) in enumerate(valid_intervals_p1)]
time_diffs_p2 = [end - start if ecg_data_p2['peaks']['ECG, X, RSPEC-R.1'][i]['valid'] and
                  ecg_data_p2['peaks']['ECG, X, RSPEC-R.1'][i-1]['valid'] else None
                  for i, (start, end) in enumerate(valid_intervals_p2)]

# adjust length to have as much peaks as intervals (so peaks - 1)
peak_times_p1 = peak_times_p1[:min(len(peak_times_p1[:-1]), len(time_diffs_p1))]
peak_times_p2 = peak_times_p2[:min(len(peak_times_p2[:-1]), len(time_diffs_p2))]





def calculate_valid_RRs(ecg_data,label):
    """
    Identifies valid RR intervals in the data (ignoring those that are 
    partially in time intervals marked as artefactual).
    """
    peak_times = [data['t'] for data in ecg_data['peaks'][label]]
    something
    something
    something
    return peak_times,time_diffs



## Idea: zip together
t_and_diff = list(zip(peak_times_p1,time_diffs_p1))
# for unpacking: list(zip(*t_and_diff))


peak_times_p1, time_diffs_p1 = calculate_valid_RRs(ecg_data_p1,'ECG, X, RSPEC-R')
peak_times_p2, time_diffs_p2 = calculate_valid_RRs(ecg_data_p2,'ECG, X, RSPEC-R.1')


# List comprehensions :
#    [ (t,d) for (t,d) in t_and_diff if t<30 ]



# Graph showing R-R intervals through time for both participants
plt.plot(peak_times_p1, time_diffs_p1, label='participant 1', color = 'blue')
plt.plot(peak_times_p2, time_diffs_p2, label='participant 2', color = 'red')

plt.xlabel('Time (s)')
plt.ylabel('R-R intervals')
plt.legend()
plt.show()




#
# [ (p1_RR,p2_RR) ]
# [ (.55,.73), (.56,.72), (.57,NAN), ... ]
# [ (p1r,p2r) for (p1r,p2r) in intervals if ~np.isnan(p1r) and ~np.isnan(p2r) ]



_ = interpolate_RRs(peak_times_p1,time_diffs_p1)



########## initial correlations #########################################################################################################

# convert 'none' to 'nan' (to omit them for correlations)
time_diffs_p1 = np.array([np.nan if x is None else x for x in time_diffs_p1])  ## can be done in calculate_valid_RRs
time_diffs_p2 = np.array([np.nan if x is None else x for x in time_diffs_p2])

# interpolate participant 2 data to match participant 1 time points
    # 'what would data from Particiant B be if the timepoints were the same as participant 1'
time_diffs_p2_interpolated = interp1d(peak_times_p2, time_diffs_p2, fill_value='extrapolate')(peak_times_p1) # interpolation based on p1 data points

# filter out NaN values
time_diffs_p1_filter = time_diffs_p1[~np.isnan(time_diffs_p1) & ~np.isnan(time_diffs_p2_interpolated)]
time_diffs_p2_filter = time_diffs_p2_interpolated[~np.isnan(time_diffs_p1) & ~np.isnan(time_diffs_p2_interpolated)]

# spearman correlation
correlation, p_value = spearmanr(time_diffs_p1_filter, time_diffs_p2_filter)
print(f"initial r = {correlation:.4f}, initial p = {p_value:.4f}")

# scatter plot with R-R intervals for Participant 1 and 2
plt.scatter(time_diffs_p1_filter, time_diffs_p2_filter, color = 'black')
plt.xlabel('R-R intervals of Participant A')
plt.ylabel('R-R intervals of Participant B')
plt.title('Correlation R-R Intervals')
plt.show()

########## correlations with time shifts ##################################################################################################

time_shifts = np.arange(-10, 10, 0.1)  # 10 sec shift (-5sec to +5 sec, increments of 0.1sec)

# list to store the correlations for each shift
shifted_corr_value = []
shifted_p_values = []

# time shifts and correlations
for shift in time_shifts:
    shifted_times_p2 = np.array(peak_times_p2) + shift  # shift data from participant 2 
    
    # interpolate participant 2 data with each shift to match data points of participant 1
    time_diffs_p2_interpolated = interp1d(shifted_times_p2, time_diffs_p2, fill_value='extrapolate')(peak_times_p1)  

    # filter out NaN values for the current shift
    time_diffs_p1_filter = time_diffs_p1[~np.isnan(time_diffs_p1) & ~np.isnan(time_diffs_p2_interpolated)]
    time_diffs_p2_filter = time_diffs_p2_interpolated[~np.isnan(time_diffs_p1) & ~np.isnan(time_diffs_p2_interpolated)]

    # calculate correlation for the current shift
    corr, p_value = spearmanr(time_diffs_p1_filter, time_diffs_p2_filter)  
    shifted_corr_value.append(corr)  # store the correlations here
    shifted_p_values.append(p_value)   # store the p values here
    
# find the time shift with the highest correlation
max_corr_time_shift = time_shifts[np.argmax(shifted_corr_value)]
max_corr_value = shifted_corr_value[np.argmax(shifted_corr_value)]
max_corr_p_value = shifted_p_values[np.argmax(shifted_corr_value)]
print(f"highest r = {max_corr_value:.4f}, highest p = {max_corr_p_value:.4f}, time shift = {max_corr_time_shift:.4f}s")

# y = spearman r ; x = time shift
plt.plot(time_shifts, shifted_corr_value)
plt.axvline(x=max_corr_time_shift, color='red', label=f'Max r at {max_corr_time_shift:.4f}s')
plt.xlabel('Time shift (s)')
plt.ylabel('Spearman correlation (r)')
plt.axhline(y=0,color='gray',dashes=[1,4])
plt.legend()
plt.show()

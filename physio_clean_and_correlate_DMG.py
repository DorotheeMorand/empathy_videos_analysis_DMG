import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
from scipy.interpolate import interp1d
from scipy.stats import spearmanr
from matplotlib.backends.backend_pdf import PdfPages  

# data from participant dyads 
# video creation study: targets talk about emotional life events, observers listen. Then, they exchange roles.
# goal: clean ECG data/remove invalid areas; RR correlations between 2 participants (w 10 sec window)

    # https://matplotlib.org/stable/plot_types/index.html
    # https://numpy.org/doc/stable/reference/generated/numpy.interp.html
    # https://docs.scipy.org/doc/scipy/reference/interpolate.html 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    # https://matplotlib.org/stable/api/backend_pdf_api.html

########## functions to identify peaks and remove invalid areas ############################################################

# does_overlap - from Floris biotop
def does_overlap(intv1, intv2):
    (a1, b1) = intv1
    if a1 > b1: (a1, b1) = (b1, a1)  # if coded inversely, flip
    (a2, b2) = intv2
    if a2 > b2: (a2, b2) = (b2, a2)
    overlapmin = max([a1, a2])
    overlapmax = min([b1, b2])
    return overlapmin <= overlapmax

# exclude intervals overlapping with invalid areas
def exclude_invalid(peak_times, invalid_areas):
    valid_intervals = []
    for i in range(len(peak_times) - 1):  # -1 so it does not go out of range (for intervals)
        start_time = peak_times[i]  # peak 1
        end_time = peak_times[i + 1]  # peak 2
        interval = (start_time, end_time)
        if not any(does_overlap(interval, invalid_area) for invalid_area in invalid_areas):
            valid_intervals.append(interval)
    return valid_intervals

# define peak times and calculate valid RR intervals
def calculate_valid_RRs(ecg_data, label):
    
    # calculate peak times
    peak_times = [data['t'] for data in ecg_data['peaks'][label]]
    peak_times.sort()
    # identify invalid areas
    invalid_areas = [(data[1], data[2]) for data in ecg_data['invalid'][label]] # data[1] = start time; data[2] = end time of invalid area
    # exclude invalid intervals
    valid_intervals = exclude_invalid(peak_times, invalid_areas)
    # identify R-R intervals
    time_diffs = [end - start if ecg_data['peaks'][label][i]['valid'] and
                  ecg_data['peaks'][label][i-1]['valid'] else None
                  for i, (start, end) in enumerate(valid_intervals)]
    # adjust length to have as many peaks as intervals (so peaks - 1) otherwise does not work
    peak_times = peak_times[:min(len(peak_times[:-1]), len(time_diffs))]
    # convert 'none' to 'nan' (to omit them for correlations)
    time_diffs = np.array([np.nan if x is None else x for x in time_diffs])
    # return a list of peak times and RR intervals
    return peak_times, time_diffs

# interpolate RR intervals based on common time points of both participants (later specified at each sec)
def interpolate_RR(peak_times, time_diffs, common_time_points):
    return interp1d(peak_times, time_diffs, fill_value='extrapolate')(common_time_points)

########## function to clean data and calculate correlation #################################################################

def clean_and_correlate(target, observer, emotions):
    
    # directory of the current script file
    script_directory = os.path.dirname(os.path.abspath(__file__))  # This line should be at the top

    # remove any known missing files MODIFY HERE IF NEEDED
    missing_files = [
        "P03_neutral_P03_peaks",
        "P03_neutral_P12_peaks"]

    # dataframe to store correlation results
    correlation_results = pd.DataFrame(columns=['target', 
                                                'observer', 
                                                'emotion', 
                                                'initial_r', 
                                                'max_r', 
                                                'initial_p', 
                                                'max_p', 
                                                'time_shift'])

    # new folder 'RR_correlation_graphs' if it doesn't exist
    output_folder = os.path.join(script_directory, 'RR_correlation_graphs')
    os.makedirs(output_folder, exist_ok=True)

    # Save graphs (pdf) in the new folder
    pdf_filename = os.path.join(output_folder, f'graphs_{target}_{observer}.pdf')
    pdf_pages = PdfPages(pdf_filename) 
     
    # figure showing 1) RR intervals through time and 2) shifted correlation -> for all emotions
    figure, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 15))  # 6 rows for 6 emotions
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # adjust spacing between plots

    add_legend = True


     # loop through each emotion to clean data, perform correlations, plot results
    for emotion_graph, emotion in enumerate(emotions):

        # check if files are missing
        for file in [f"{target}_{emotion}_{target}_peaks", f"{target}_{emotion}_{observer}_peaks"]:
            if file in missing_files:
                print(f"Missing file: {file}")
        
        # if files are missing, skip to next emotion
        if f"{target}_{emotion}_{target}_peaks" in missing_files or f"{target}_{emotion}_{observer}_peaks" in missing_files:
            continue
        
        # open data
        with open(os.path.join(script_directory, 'participants_physio_data', f"{f"{target}_{emotion}_{target}_peaks"}.json")) as data_target:
                ecg_data_target = json.load(data_target)
        with open(os.path.join(script_directory, 'participants_physio_data', f"{f"{target}_{emotion}_{observer}_peaks"}.json")) as data_observer:
                ecg_data_observer = json.load(data_observer)
        
        # calculate valid peaks and RR intervals (with previous function)
        peak_times_target, time_diffs_target = calculate_valid_RRs(ecg_data_target, f'ECG {target}')
        peak_times_observer, time_diffs_observer = calculate_valid_RRs(ecg_data_observer, f'ECG {observer}')
        
        # check common time points (to interpolate) -> should be the same but we never know
        common_time_points = np.arange(max(min(peak_times_target), min(peak_times_observer)), 
                                       min(max(peak_times_target), max(peak_times_observer)), 
                                       1)  # 1 = at each second
        
        # interpolate RR intervals
        time_diffs_target_interpol = interpolate_RR(peak_times_target, time_diffs_target, common_time_points)
        time_diffs_observer_interpol = interpolate_RR(peak_times_observer, time_diffs_observer, common_time_points)
        
        # filter NaN values
        time_diffs_target_filter = time_diffs_target_interpol[~np.isnan(time_diffs_target_interpol) & ~np.isnan(time_diffs_observer_interpol)]
        time_diffs_observer_filter = time_diffs_observer_interpol[~np.isnan(time_diffs_target_interpol) & ~np.isnan(time_diffs_observer_interpol)]
        
        # spearman correlation
        correlation, p_value = spearmanr(time_diffs_target_filter, time_diffs_observer_filter)
        
        # time shift analysis
        time_shifts = np.arange(-5, 5, 0.1)  # from -5 to +5 sec in 0.1 second increments
        shifted_corr_value = []  # store r values for each time shift
        shifted_p_values = []    # store p values for each time shift
        
        # loop through all time shifts and find the highest correlation
        for shift in time_shifts:
            
            shifted_times_observer = np.array(peak_times_observer) + shift  # shift the observer's peak times
            
            # interpolate RR intervals of observer at each second
            new_common_time_points = np.arange(
                max(min(shifted_times_observer), min(peak_times_target)), 
                min(max(shifted_times_observer), max(peak_times_target)),  
                1)  # 1 = at each second
            time_diffs_observer_interpolated = interpolate_RR(shifted_times_observer, time_diffs_observer, new_common_time_points)
            time_diffs_target_interpolated = interpolate_RR(peak_times_target, time_diffs_target, new_common_time_points)
            
            # filter NaN values, ensure both arrays have the same length
            time_diffs_target_filter = time_diffs_target_interpolated[~np.isnan(time_diffs_target_interpolated) & ~np.isnan(time_diffs_observer_interpolated)]
            time_diffs_observer_filter = time_diffs_observer_interpolated[~np.isnan(time_diffs_target_interpolated) & ~np.isnan(time_diffs_observer_interpolated)]
            
            # store correlation values
            corr, p_value = spearmanr(time_diffs_target_filter, time_diffs_observer_filter)
            shifted_corr_value.append(corr)
            shifted_p_values.append(p_value)
            
        # find the max correlation and values (r and p and shift time)
        max_corr_time_shift = time_shifts[np.argmax(shifted_corr_value)]
        max_corr_value = shifted_corr_value[np.argmax(shifted_corr_value)]
        max_corr_p_value = shifted_p_values[np.argmax(shifted_corr_value)]
        
        # results to dataframe
        correlation_results = pd.concat([correlation_results,
                                         pd.DataFrame.from_dict({'target': [target],
                                                                 'observer': [observer],
                                                                 'emotion': [emotion],
                                                                 'initial_r': [correlation],
                                                                 'max_r': [max_corr_value],
                                                                 'initial_p': [p_value],
                                                                 'max_p': [max_corr_p_value],
                                                                 'time_shift': [max_corr_time_shift]})], 
                                         axis=0)
        
        # plot results
        plot_RR_intervals = axes[emotion_graph, 0]  # 0 = 1st column -> RR intervals over time
        plot_RR_intervals.plot(peak_times_target, time_diffs_target, label='Target', color='blue')  # target in blue
        plot_RR_intervals.plot(peak_times_observer, time_diffs_observer, label='Observer', color='red')  # observer in red
        plot_RR_intervals.text(0.5, 1.05, f'{emotion}', ha='center', va='center', transform=plot_RR_intervals.transAxes, fontsize=10)
        if add_legend:  # legend only for the first plot
            plot_RR_intervals.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
            add_legend = False  # false after adding the legend once

        plot_correlations = axes[emotion_graph, 1]  # 1 = 2nd column -> correlation values over time shifts
        plot_correlations.plot(time_shifts, shifted_corr_value)
        plot_correlations.axvline(x=max_corr_time_shift, color='red', label=f'Max r at {max_corr_time_shift:.4f}s')
        plot_correlations.axvline(x=0, color='lightgray', dashes=[1, 4])
        plot_correlations.axhline(y=0, color='gray', dashes=[1, 4]) 
        
        # 3rd column = correlation info
        stats_text = f"initial r: {correlation:.4f}\nmax r: {max_corr_value:.4f}\ninitial p: {p_value:.4f}\nmax p: {max_corr_p_value:.4f}\ntime shift: {max_corr_time_shift:.2f} s"
        plot_correlations.text(1.05, 0.5, stats_text, transform=plot_correlations.transAxes, fontsize=8, verticalalignment='center')

    # labels and title
    figure.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=10)
    figure.text(0.04, 0.5, 'R-R intervals', va='center', rotation='vertical', fontsize=10)
    figure.text(0.96, 0.5, 'Spearman correlation (r)', va='center', rotation='vertical', fontsize=10)
    figure.suptitle(f'RR through time and correlation results for {target} (target) - {observer} (observer)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0.04, 0.04, 0.96, 0.96])  # spacing

    pdf_pages.savefig(figure)  # figure to the pdf
    pdf_pages.close()  # close pdf after saving

    return correlation_results

# define emotions and participant pairs
emotions = ['joy', 'sadness', 'anger', 'pride', 'shame', 'neutral']
participant_pairs = [
    ("P01", "P13"),
    ("P03", "P12"), 
    ("P15", "P17"),
    ("P22", "P29"),
    ("P27", "P34"),
    ("P32", "P38"),
    ("P37", "P44"),
    ("P42", "P45")
    ]

# loop through each pair for clean_and_correlate
results_list = []
for target, observer in participant_pairs:
    # analyze both roles (because participants exchanged roles)
    results_target_observer = clean_and_correlate(target=target, observer=observer, emotions=emotions)
    results_observer_target = clean_and_correlate(target=observer, observer=target, emotions=emotions)
    
    # store results
    results_list.append(results_target_observer)
    results_list.append(results_observer_target)

# all results in one dataframe
combined_results = pd.concat(results_list, ignore_index=True)

# results in an excel file
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'correlation_results.xlsx')
combined_results.to_excel(output_path, index=False, engine='xlsxwriter')
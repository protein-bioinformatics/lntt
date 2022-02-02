#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MIT License

Copyright (c) 2021 by the authors listed in the LICENSE file

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from sklearn import metrics
from statsmodels.stats import multitest
from statistics import median, mean
import matplotlib.pyplot as plt
from math import log, exp, log10, isnan, sqrt
from random import random
import os
import sys
import json
import urllib.parse
import urllib.request
from random import randint
import multiprocessing
import threading
import traceback


normalized_suffix = ", normalized"
pval_adjust_methods = {"bonferroni": "Bonferoni, one-step correction",
                       "sidak": "Sidak, one-step correction",
                       "holm-sidak": "Holm-Sidak, step down method using Sidak adjustments",
                       "holm": "Holm, step-down method using Bonferroni adjustments",
                       "simes-hochberg": "Simes-Hochberg, step-up method (independent)",
                       "hommel": "Hommel closed method based on Simes tests (non-negative)",
                       "fdr_bh": "Benjamini/Hochberg (non-negative)",
                       "fdr_by": "Benjamini/Yekutieli (negative)",
                       "fdr_tsbh": "two stage Benjamini/Hochberg fdr correction (non-negative)",
                       "fdr_tsbky": "two stage fdr correction (non-negative)"
                      }
ttest_types = {"two-sided": "two-sided",
               "less": "one-sided, less",
               "greater": "one-sided, greater"}
                    


class LNTT(multiprocessing.Process):
    
    COLOR_REGULATION_TREATED = "#998ec3"
    COLOR_REGULATION_TREATED_LABEL = "#6a6388"
    COLOR_REGULATION_WT = "#f1a340"
    COLOR_REGULATION_WT_LABEL = "#b67b30"
    COLOR_UNREGULATED = "grey"
    COLOR_FC_BOUNDARIES = "darkred"
    COLOR_DISTRIBUTION_BAR = "#67a9cf"
    
    def __init__(self, lntt_queue, external_queue):
        super(LNTT, self).__init__()
        self.logging = external_queue
        self.lntt_queue = lntt_queue
        self.internal_queue = multiprocessing.Queue()
        self.interrupt = False

        
    def run(self):
        while True:
            task = self.lntt_queue.get()
            if type(task) == dict:
                self.interrupt = False
                multiprocessing.Process(target = self.process, args=(task, self.internal_queue)).start()
            elif type(task) == int:
                self.internal_queue.put(task)
            else:
                self.internal_queue.put(1)
                break
    
    
    
    def data_frame_imputation(self, data_frame, column_names, parameters, internal_queue, report):
        data_frame_imputation = parameters["data_imputation_value"]
        
        self.logging.put("Imputing the data containing at most %i missing values per entitiy" % data_frame_imputation)
        num_pre = len(data_frame)
        if data_frame_imputation > 0:
            if report != None: report.append("Values were imputed when an entity contained up to %i missing values." % data_frame_imputation)
            rows_to_drop = list(data_frame[column_names].isnull().sum(axis = 1) >= data_frame_imputation)
            rows_to_drop = np.array([idx for idx, v in enumerate(rows_to_drop) if v])
            data_frame.drop(rows_to_drop, inplace = True)
            data_frame.reset_index(inplace = True, drop = True)
            
            set_col_names = set(column_names)
            for ci, col in enumerate(data_frame):
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                if col not in set_col_names: continue
                l = sorted(list(v for v in data_frame[col] if not isnan(v)))
                max_val = l[int(len(l) * 0.05)]
                
                for i, v in enumerate(data_frame[col].isnull()):
                    if v:
                        data_frame.iloc[i, ci] = max_val * random()
        
        else:
            # delete all rows that are not fully filled
            if report != None: report.append("All entities at least one with missing value were discarded.")
            rows_to_drop = list(data_frame[column_names].isnull().any(axis = 1))
            rows_to_drop = np.array([idx for idx, v in enumerate(rows_to_drop) if v])
            data_frame.drop(rows_to_drop, inplace = True)
            data_frame.reset_index(inplace = True, drop = True)

        num_post = len(data_frame)
        self.logging.put("After data imputation, %s of %s entities remain" % (num_post, num_pre))
        if report != None: report.append("After this filtering step, %i entities remained." % len(data_frame))





    def cv_filtering(self, data_frame, column_names, parameters, internal_queue, report):
        cv_threshold = parameters["cv_threshold"]
        
        num_pre = len(data_frame)
        self.logging.put("Filtering out entities having a CV <= %0.3f" % cv_threshold)
        if report != None: report.append("The entities were filtered by the covariance of variation (CV): when the entities had a CV < %0.1f %% over all measurement (regardless of their conditions), they were discarded." % (cv_threshold * 100))
        cv = data_frame[column_names].std(axis = 1, skipna = True) / data_frame[column_names].mean(axis = 1, skipna = True)
        try:
            result = internal_queue.get_nowait()
            if result != None:
                self.interrupt = True
                return
        except:
            pass
        rows_to_drop = np.array([i for i, v in enumerate(list(cv < cv_threshold)) if v])
        data_frame.drop(rows_to_drop, inplace = True)
        data_frame.reset_index(inplace = True, drop = True)
        num_post = len(data_frame)
        
        self.logging.put("After CV filtering, %s of %s entities remain" % (num_post, num_pre))
        if report != None: report.append("After CV filtering, %i entities remained." % len(data_frame))
        
        
        
        

    def normalization(self, original_data_frame, output_folder, column_names, conditions, parameters, internal_queue, report):
        global normalized_suffix
        reference_titles = conditions[parameters["norm_ref_condition"]]
        with_plotting = parameters["with_normalization_plotting"]

        self.logging.put("Normalizing columns applying rank invariant set normalization")
        if report != None: report.append("Normalization was performed by applying 'global rank-invariant set normalization' (DOI: 10.1186/1471-2105-9-520).")
        col_cnt = len([c for c in original_data_frame])
        data_frame = pd.DataFrame(original_data_frame[column_names])
        
        n_rows, n_cols = len(data_frame), len(data_frame.keys())
        
        # save data_frame before normalization
        violin_data_frame, violin_data_frame_2, violin_col_names = [], [], []
        violin_distribution, violin_distribution_2, vd_col_names = [], [], []

        reference, max_val = 0, 0
        for col in reference_titles:
            l = np.sum(~np.isnan(data_frame[col]))
            if max_val < l:
                reference = col
                max_val = l
            
        try:
            result = internal_queue.get_nowait()
            if result != None:
                self.interrupt = True
                return
        except:
            pass
                

        # computing standard deviation of reference columns for median window estimation
        if len(reference_titles) > 1:
            std = data_frame[reference_titles].std(axis = 1, skipna = True)
            
        
        # sorting all columns according to reference
        masked_ref = np.ma.masked_where(np.isnan(data_frame[reference]), data_frame[reference])
        sorting = np.argsort(masked_ref)
        unsorting = np.zeros(len(sorting), dtype = "int64")
        for i, v in enumerate(sorting): unsorting[v] = i
        
        # sort all columns according to the reference column
        for t in data_frame:
            data_frame[t] = np.array([data_frame[t][j] for j in sorting], dtype=np.float64)
            
        
        # set the reference columns and its masked version
        RefX = np.array(data_frame[reference])
        masked_X = np.ma.masked_where(np.isnan(RefX), RefX)
        try:
            result = internal_queue.get_nowait()
            if result != None:
                self.interrupt = True
                return
        except:
            pass


        # linear regression for estimating appropriate boundaries for
        # consecutive moving average window, in equation: y = m * x + n
        if len(reference_titles) > 1:
            std = np.array([std[j] for j in sorting], dtype=np.float64)
            std_not_nan = ~np.isnan(std)
            ref_not_nan = ~np.isnan(RefX)
            m, n = np.polyfit(masked_X[ref_not_nan & std_not_nan], std[ref_not_nan & std_not_nan], 1)
        else:
            m, n = 1. / 10., 0

        # compute boundaries of the median window for all values
        L = np.searchsorted(RefX, masked_X - 3 * m * masked_X)
        R = np.searchsorted(masked_X, masked_X + 3 * m * masked_X)
        reference_col_num = 0
        min_points_in_window = 10
        num_nan = max(np.sum(np.isnan(RefX)), 1)
        try:
            result = internal_queue.get_nowait()
            if result != None:
                self.interrupt = True
                return
        except:
            pass
        
        for cc, col_name in enumerate(data_frame):
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            self.logging.put("  - Normalizing column '%s'" % col_name)
            Y = data_frame[col_name]
            masked_Y = np.ma.masked_where(np.isnan(Y), Y)
            violin_data = np.log10(Y)
            violin_distribution.append(np.array(violin_data[~np.isnan(violin_data)]).tolist())
            
            if col_name == reference:
                reference_col_num = cc
                
                violin_data = np.log10(Y)
                violin_distribution_2.append(np.array(violin_data[~np.isnan(violin_data)]).tolist())
                Y_no_nans = ~np.isnan(Y)
                vd_col_names.append("%s (%i)" % (col_name, np.sum(Y_no_nans)))
                original_data_frame.insert(col_cnt, "%s%s" % (col_name, normalized_suffix),  np.array([Y[j] for j in unsorting]), True)
                col_cnt += 1
                continue
            violin_data_col = np.log10(data_frame[col_name] / RefX)
            violin_data_frame.append(np.array(violin_data_col[~np.isnan(violin_data_col)]).tolist())

            Y_l = np.array(Y / RefX)
            
            prevL, last = np.zeros(len(Y), dtype = "int64"), 0
            Y_no_nans = ~np.isnan(Y_l)
            for i in range(1, len(Y)):
                prevL[i] = last
                if Y_no_nans[i]: last = i
                
            nextR, last = np.zeros(len(Y), dtype = "int64"), len(Y) - 1
            for i in range(len(Y) - 2, -1, -1):
                nextR[i] = last
                if Y_no_nans[i]: last = i
            
            
            min_points = min(min_points_in_window, log(np.sum(Y_no_nans)) + 1)
            
            YY = np.array([np.nan] * n_rows, dtype = "float64")
            nan_start = -1
            for z, (l, r, y) in enumerate(zip(L, R, Y)):
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                if np.isnan(RefX[z]):
                    nan_start = z
                    break
                if np.ma.is_masked(y): continue
                Y_sub = Y_l[l : r + 1]
                n = list(Y_sub[~np.isnan(Y_sub)])
                
                while len(n) < min_points:
                    if l > 0:
                        l = prevL[l]
                        if ~np.isnan(Y_l[l]): n.append(Y_l[l])
                    
                    if r < n_rows - num_nan:
                        r = nextR[r]
                        if ~np.isnan(Y_l[r]): n.append(Y_l[r])
                    
                    if l == 0 and r >= n_rows - num_nan: break
                    
                YY[z] = np.median(n)
                
                
            normalized_column_data = Y / YY
            violin_data_col = np.log10(normalized_column_data / RefX)
            violin_data_frame_2.append(np.array(violin_data_col[~np.isnan(violin_data_col)]).tolist())
            Y_no_nans = ~np.isnan(normalized_column_data / RefX)
            violin_col_names.append("%s (%i)" % (col_name, np.sum(Y_no_nans)))
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            
            
            # infer normalization factor for intensities without corresponding reference value
            # by considering normalized neighbors
            if np.isnan(RefX[-1]):
                Y_sorted_index = np.argsort(np.argsort(masked_Y))
                for i in range(nan_start, n_rows):
                    try:
                        result = internal_queue.get_nowait()
                        if result != None:
                            self.interrupt = True
                            return
                    except:
                        pass
                    if np.isnan(Y[i]): continue
                    n, l, r = [], i, i
                    if not np.isnan(YY[Y_sorted_index[i]]): n.append(YY[Y_sorted_index[i]])
                    while len(n) < 5:
                        if l > 0:
                            l -= 1
                            if not np.isnan(YY[Y_sorted_index[l]]): n.append(YY[Y_sorted_index[l]])
                        if r < n_rows - 1:
                            r += 1
                            if not np.isnan(YY[Y_sorted_index[r]]): n.append(YY[Y_sorted_index[r]])

                    YY[i] = np.median(n)
                
            normalized_column_data = Y / YY
            violin_data = np.log10(normalized_column_data)
            violin_distribution_2.append(np.array(violin_data[~np.isnan(violin_data)]).tolist())
            Y_no_nans = ~np.isnan(normalized_column_data)
            vd_col_names.append("%s (%i)" % (col_name, np.sum(Y_no_nans)))
            
            
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
        
            # plot a normalization figure to check the normalization quality
            if with_plotting:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), sharey = True) #, tight_layout=True)
                
                plot_X = RefX[~np.isnan(RefX) & ~np.isnan(Y)]
                plot_Y = Y[~np.isnan(RefX) & ~np.isnan(Y)]
                plot_Y_corr = normalized_column_data[~np.isnan(RefX) & ~np.isnan(Y)]
                plot_YY = YY[~np.isnan(RefX) & ~np.isnan(Y)]
                
                
                ax2.axhline(linewidth=2, color='r')
                ax1.scatter(np.log10(plot_X), np.log2(plot_Y / plot_X))
                ax1.plot(np.log10(plot_X), np.log2(plot_YY), '-', color = "r")
                ax2.scatter(np.log10(plot_X), np.log2(plot_Y_corr / plot_X))
                
                
                ax1.set_ylabel("log${}_2$ (protein abundance ratio)")
                
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.xlabel("log${}_{10}$ protein abundances of '%s'" % reference)
                
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                clean_col_name = col_name.replace("/", "").replace("\\", "")
                clean_col_name = clean_col_name.replace(":", "").replace("*", "")
                clean_col_name = clean_col_name.replace("?", "").replace("!", "")
                clean_col_name = clean_col_name.replace("<", "").replace(">", "")
                clean_col_name = clean_col_name.replace("\"", "").replace("'", "").replace("|", "")
                
                fig.suptitle("Condition '%s' against reference\n'%s' during normalization" % (col_name, reference))
                #plt.tight_layout()
                fig_filename = os.path.join(output_folder, "%s-normalization.pdf" % clean_col_name)
                plt.savefig(fig_filename, dpi = 600)
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                fig_filename = os.path.join(output_folder, "%s-normalization.png" % clean_col_name)
                plt.savefig(fig_filename, dpi = 600)
                plt.cla()   # Clear axis
                plt.clf()   # Clear figure
                plt.close() # Close a figure window

            original_data_frame.insert(col_cnt, "%s%s" % (col_name, normalized_suffix),  np.array([normalized_column_data[j] for j in unsorting]), True)
            col_cnt += 1
            
        
        
        
        
        if with_plotting:
            violin_file = []
            violin_file.append("import matplotlib.pyplot as plt")
            violin_file.append("COLOR_REGULATION_TREATED = '%s'" % LNTT.COLOR_REGULATION_TREATED)
            violin_file.append("COLOR_REGULATION_TREATED_LABEL = '%s'" % LNTT.COLOR_REGULATION_TREATED_LABEL)
            violin_file.append("violin_data_frame = %s" % str(violin_data_frame))
            violin_file.append("violin_data_frame_2 = %s" % str(violin_data_frame_2))
            violin_file.append("violin_col_names = %s" % str(violin_col_names))
            violin_file.append("reference = '%s'" % reference)
            
            
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex = True, sharey = True)
            violin_parts = ax1.violinplot(violin_data_frame, range(len(violin_data_frame)), widths=0.8, showmeans=True, showextrema=True)
            for i, pc in enumerate(violin_parts['bodies']):
                pc.set_facecolor(LNTT.COLOR_REGULATION_TREATED)
                pc.set_edgecolor(LNTT.COLOR_REGULATION_TREATED_LABEL)
                pc.set_alpha(1)
            ax1.set_ylabel('log${}_{10}$ protein abundance\nratios to reference')
            ax1.grid()
            
            violin_file.append("fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex = True, sharey = True)")
            violin_file.append("violin_parts = ax1.violinplot(violin_data_frame, range(len(violin_data_frame)), widths=0.8, showmeans=True, showextrema=True)")
            violin_file.append("for i, pc in enumerate(violin_parts['bodies']):")
            violin_file.append("    pc.set_facecolor(COLOR_REGULATION_TREATED)")
            violin_file.append("    pc.set_edgecolor(COLOR_REGULATION_TREATED_LABEL)")
            violin_file.append("    pc.set_alpha(1)")
            violin_file.append("ax1.set_ylabel('log${}_{10}$ protein abundance\\nratios to reference')")
            violin_file.append("ax1.grid()")
            
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            
            violin_parts = ax2.violinplot(violin_data_frame_2, range(len(violin_data_frame_2)), widths=0.8, showmeans=True, showextrema=True)
            for i, pc in enumerate(violin_parts['bodies']):
                pc.set_facecolor(LNTT.COLOR_REGULATION_TREATED)
                pc.set_edgecolor(LNTT.COLOR_REGULATION_TREATED_LABEL)
                pc.set_alpha(1)
            ax2.set_ylabel('log${}_{10}$ protein abundance\nratios to reference')
            ax2.grid()
            plt.xticks(range(len(violin_col_names)), violin_col_names, rotation = 45, horizontalalignment='right')
            
            violin_file.append("violin_parts = ax2.violinplot(violin_data_frame_2, range(len(violin_data_frame_2)), widths=0.8, showmeans=True, showextrema=True)")
            violin_file.append("for i, pc in enumerate(violin_parts['bodies']):")
            violin_file.append("    pc.set_facecolor(COLOR_REGULATION_TREATED)")
            violin_file.append("    pc.set_edgecolor(COLOR_REGULATION_TREATED_LABEL)")
            violin_file.append("    pc.set_alpha(1)")
            violin_file.append("ax2.set_ylabel('log${}_{10}$ protein abundance\\nratios to reference')")
            violin_file.append("ax2.grid()")
            violin_file.append("plt.xticks(range(len(violin_col_names)), violin_col_names, rotation = 45, horizontalalignment='right')")
            
            
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            
            fig.suptitle('Overview of protein rations during normalization\\nagainst reference column \'%s\'' % reference)
            fig_filename = os.path.join(output_folder, 'normalization-overview.pdf')
            plt.savefig(fig_filename, dpi = 600)
            
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            
            fig_filename = os.path.join(output_folder, 'normalization-overview.png')
            plt.savefig(fig_filename, dpi = 600)
            plt.cla()   # Clear axis
            plt.clf()   # Clear figure
            plt.close() # Close a figure window
            
            
            violin_file.append("fig.suptitle('Overview of protein rations during normalization\\nagainst reference column \\'%s\\'' % reference)")
            violin_file.append("plt.savefig('normalization-overview.pdf', dpi = 600)")
            violin_file.append("plt.savefig('normalization-overview.png', dpi = 600)")
            violin_file.append("plt.cla()   # Clear axis")
            violin_file.append("plt.clf()   # Clear figure")
            violin_file.append("plt.close() # Close a figure window")
            
            
            violin_file_filename = os.path.join(output_folder, 'normalization-overview.py')
            with open(violin_file_filename, "wt") as out:
                out.write("%s\n" % "\n".join(violin_file))
            
            
            
            
            
            
            
            
            distribution_file = []
            distribution_file.append("import matplotlib.pyplot as plt")
            distribution_file.append("COLOR_REGULATION_WT = '%s'" % LNTT.COLOR_REGULATION_WT)
            distribution_file.append("COLOR_REGULATION_TREATED_LABEL = '%s'" % LNTT.COLOR_REGULATION_TREATED_LABEL)
            distribution_file.append("COLOR_REGULATION_TREATED = '%s'" % LNTT.COLOR_REGULATION_TREATED)
            distribution_file.append("violin_distribution = %s" % str(violin_distribution))
            distribution_file.append("violin_distribution_2 = %s" % str(violin_distribution_2))
            distribution_file.append("reference_col_num = %s" % str(reference_col_num))
            distribution_file.append("vd_col_names = %s" % str(vd_col_names))
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex = True, sharey = True)
            fig.suptitle("Overview of protein abundance distributions during normalization")
            violin_parts = ax1.violinplot(violin_distribution, range(len(violin_distribution)), widths=0.8, showmeans=True, showextrema=True)
            for i, pc in enumerate(violin_parts['bodies']):
                pc.set_facecolor(LNTT.COLOR_REGULATION_WT if i == reference_col_num else LNTT.COLOR_REGULATION_TREATED)
                pc.set_edgecolor(LNTT.COLOR_REGULATION_TREATED_LABEL)
                pc.set_alpha(1)
            ax1.set_ylabel("log${}_{10}$ protein abundances")
            ax1.grid()
            
            distribution_file.append("fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex = True, sharey = True)")
            distribution_file.append("fig.suptitle('Overview of protein abundance distributions during normalization')")
            distribution_file.append("violin_parts = ax1.violinplot(violin_distribution, range(len(violin_distribution)), widths=0.8, showmeans=True, showextrema=True)")
            distribution_file.append("for i, pc in enumerate(violin_parts['bodies']):")
            distribution_file.append("    pc.set_facecolor(COLOR_REGULATION_WT if i == reference_col_num else COLOR_REGULATION_TREATED)")
            distribution_file.append("    pc.set_edgecolor(COLOR_REGULATION_TREATED_LABEL)")
            distribution_file.append("    pc.set_alpha(1)")
            distribution_file.append("ax1.set_ylabel('log${}_{10}$ protein abundances')")
            distribution_file.append("ax1.grid()")
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            
            violin_parts = ax2.violinplot(violin_distribution_2, range(len(violin_distribution_2)), widths=0.8, showmeans=True, showextrema=True)
            for i, pc in enumerate(violin_parts['bodies']):
                pc.set_facecolor(LNTT.COLOR_REGULATION_WT if i == reference_col_num else LNTT.COLOR_REGULATION_TREATED)
                pc.set_edgecolor(LNTT.COLOR_REGULATION_TREATED_LABEL)
                pc.set_alpha(1)
            ax2.set_ylabel('log${}_{10}$ protein abundances')
            ax2.grid()
            plt.xticks(range(len(vd_col_names)), vd_col_names, rotation = 45, horizontalalignment='right')
            
            distribution_file.append("violin_parts = ax2.violinplot(violin_distribution_2, range(len(violin_distribution_2)), widths=0.8, showmeans=True, showextrema=True)")
            distribution_file.append("for i, pc in enumerate(violin_parts['bodies']):")
            distribution_file.append("    pc.set_facecolor(COLOR_REGULATION_WT if i == reference_col_num else COLOR_REGULATION_TREATED)")
            distribution_file.append("    pc.set_edgecolor(COLOR_REGULATION_TREATED_LABEL)")
            distribution_file.append("    pc.set_alpha(1)")
            distribution_file.append("ax2.set_ylabel('log${}_{10}$ protein abundances')")
            distribution_file.append("ax2.grid()")
            distribution_file.append("plt.xticks(range(len(vd_col_names)), vd_col_names, rotation = 45, horizontalalignment='right')")
            
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass

            #plt.tight_layout()
            fig_filename = os.path.join(output_folder, 'distribution-overview.pdf')
            plt.savefig(fig_filename, dpi = 600)
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            fig_filename = os.path.join(output_folder, 'distribution-overview.png')
            plt.savefig(fig_filename, dpi = 600)
            plt.cla()   # Clear axis
            plt.clf()   # Clear figure
            plt.close() # Close a figure window

        
            distribution_file.append("plt.savefig('distribution-overview.pdf', dpi = 600)")
            distribution_file.append("plt.savefig('distribution-overview.png', dpi = 600)")
            distribution_file.append("plt.cla()   # Clear axis")
            distribution_file.append("plt.clf()   # Clear figure")
            distribution_file.append("plt.close() # Close a figure window")
            
            distribution_file_filename = os.path.join(output_folder, 'distribution-overview.py')
            with open(distribution_file_filename, "wt") as out:
                out.write("%s\n" % "\n".join(distribution_file))
            
        



    def statistical_tesing(self, test_num, data_frame, stat_parameters, conditions, output_folder, parameters, internal_queue, report):
        global normalized_suffix, pval_adjust_methods, ttest_types
        
        test_method = stat_parameters["test"] if "test" in stat_parameters else "ttest"
        
        

        if test_method == "ttest": # T-Test
            if "pval_adjust" in parameters and parameters["pval_adjust"] not in pval_adjust_methods:
                self.logging.put("Error: p-value correction method '%s' unknown. Supported methods:\n - %s" % (parameters["pval_adjust"], "\n - ".join(m for m in pval_adjust_methods)))
                self.interrupt = True
                return

            # getting all necessary parameters for T-Test
            volcano_boundaries = stat_parameters["log_fc_thresholds"] if "log_fc_thresholds" in stat_parameters else [-1, 1]
            alpha = stat_parameters["significance_level"] if "significance_level" in stat_parameters else 0.05
            with_plotting = stat_parameters["with_stat_test_plotting"] if "with_stat_test_plotting" in stat_parameters else False
            with_normalization = parameters["with_normalization"]
            pval_adjust = stat_parameters["pval_adjust"] if "pval_adjust" in stat_parameters else "fdr_bh"
            equal_var = stat_parameters["ttest_equal_variance"] if "ttest_equal_variance" in stat_parameters else False
            alternative = stat_parameters["ttest_type"] if "ttest_type" in stat_parameters else "two-sided"
            ttest_paired = stat_parameters["ttest_paired"] if "ttest_paired" in stat_parameters else False
            filter_out = float(stat_parameters["min_values_per_condition"]) if "min_values_per_condition" in stat_parameters else 0
            use_log_values = stat_parameters["log_values"] if "log_values" in stat_parameters else False
            
            scipy_version = [int(m) for m in scipy.__version__.split(".")]
            with_ttest_type = (scipy_version[0] == 1 and scipy_version[1] >= 6) or scipy_version[0] > 1
            
            # setting default parameters when not provided by user and warn
            test_name = "Test%i-%s-%s" % (test_num + 1, stat_parameters["condition_treated"], stat_parameters["condition_wt"])
            self.logging.put("Performing statistical testing on '%s'" % test_name)
            if "pval_adjust" not in stat_parameters: self.logging.put("Warning: no p-value correction was defined, 'fdr_bh' is selected by default.")
            if "ttest_equal_variance" not in stat_parameters: self.logging.put("Warning: no equal variance on T-test was specified, unequal variance (Welch T-test) is selected by default.")
            if "ttest_type" not in stat_parameters: self.logging.put("Warning: no T-test type was specified, 'two-sided' T-test is selected by default.")
            if "ttest_paired" not in stat_parameters: self.logging.put("Warning: no specification if T-test is paired or independent, 'independent' is selected by default.")
            if "significance_level" not in stat_parameters: self.logging.put("Warning: no significance level is defined, 0.05 is selected by default.")
            if "log_fc_thresholds" not in stat_parameters and with_plotting: self.logging.put("Warning: no log2 fold change thresholds are defined, [-1, 1] is selected by default.")
            if not with_ttest_type: self.logging.put("Warning: scipy version is too old for considering a ttest type specification, 'two-sided' T-test is performed by default.")
            if "log_values" not in stat_parameters: self.logging.put("Warning, no information about the usage of log values provided, using regular values for test on default.")
            
        
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            if with_normalization:
                condition_treated_columns = ["%s%s" % (c, normalized_suffix) for c in conditions[stat_parameters["condition_treated"]]]
                condition_wt_columns = ["%s%s" % (c, normalized_suffix) for c in conditions[stat_parameters["condition_wt"]]]
            else:
                condition_treated_columns = ["%s" % c for c in conditions[stat_parameters["condition_treated"]]]
                condition_wt_columns = ["%s" % c for c in conditions[stat_parameters["condition_wt"]]]
            
            
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            if report != None: report.append("%s '%s'%s T-test was performed%s on the conditions '%s' and '%s' with '%s' correction." % ("Paired" if ttest_paired else "Independent", ttest_types[alternative], " Welch" if equal_var else "", "using log-transformed data" if use_log_values else "", stat_parameters["condition_treated"], stat_parameters["condition_wt"], pval_adjust_methods[pval_adjust]))
            
            col_cnt = len([c for c in data_frame])
            column_names = condition_treated_columns + condition_wt_columns
            
            
            # T test
            l = data_frame.shape[0]
            p_values = np.zeros(l, dtype = "float64")
            t_statistic = np.zeros(l, dtype = "float64")
            ttest_func = stats.ttest_rel if ttest_paired else stats.ttest_ind
            p_val_parameters = {"alternative": alternative, "equal_var": equal_var} if with_ttest_type else {"equal_var": equal_var}
                
            for index, row in data_frame[column_names].iterrows():
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                c1 = np.log(row[condition_treated_columns]) if use_log_values else row[condition_treated_columns]
                c2 = np.log(row[condition_wt_columns]) if use_log_values else row[condition_wt_columns]
                c1_no_nan = c1[~np.isnan(c1)]
                c2_no_nan = c2[~np.isnan(c2)]
                
                if len(c1_no_nan) == 0 and len(c2_no_nan) == 0: p_values[index] = -3
                elif len(c1_no_nan) == 0: p_values[index] = -2
                elif len(c2_no_nan) == 0: p_values[index] = -1
                elif len(c1_no_nan) == 1 or len(c2_no_nan) == 1 or len(c1_no_nan) / len(c1) < filter_out or len(c2_no_nan) / len(c2) < filter_out: p_values[index] = -4
                else:
                    values = ttest_func(c1_no_nan, c2_no_nan, **p_val_parameters)
                    t_statistic[index] = values[0]
                    p_values[index] = values[1]
            
            
            good_p = p_values >= 0
            valid_p_values = np.array(p_values[good_p])
            t_statistic[p_values < 0] = np.nan
            
            # p value correction
            adjusted_p_values_total = multitest.multipletests(valid_p_values, alpha = alpha, method = pval_adjust, is_sorted = False)
            reject_tmp, adjusted_p_values_tmp = adjusted_p_values_total[:2]
            
            
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            
            # retrieving original vector length after appyling correction function
            reject, pos = np.empty(l, dtype = np.bool_), 0
            adjusted_p_values = np.empty(l, dtype = np.float64)
            for i, gp in enumerate(good_p):
                reject[i] = False
                adjusted_p_values[i] = p_values[i]
                if gp:
                    reject[i] = reject_tmp[pos]
                    adjusted_p_values[i] = adjusted_p_values_tmp[pos]
                    pos += 1
            
            
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            
            # compute fold changes and plot volcano plot with regular p-values
            means_condition_treated = data_frame[condition_treated_columns].mean(axis = 1, skipna = True)
            means_condition_wt = data_frame[condition_wt_columns].mean(axis = 1, skipna = True)
            fold_change = means_condition_treated / means_condition_wt
            log2_fold_change = np.log2(fold_change)
            

            condition_up = (log2_fold_change >= volcano_boundaries[1]) & (reject)
            condition_down = (log2_fold_change <= volcano_boundaries[0]) & (reject)
            condition_unreg = ((volcano_boundaries[0] < log2_fold_change) & (log2_fold_change < volcano_boundaries[1])) | (~reject)
            

            
            test_result = np.array([""] * l, dtype = "object")
            test_result[condition_up] = "sig. up regulated in %s" % stat_parameters["condition_treated"]
            test_result[condition_down] = "sig. up regulated in %s" % stat_parameters["condition_wt"]
            test_result[condition_unreg] = "not regulated"
            test_result[(p_values == -3) | (p_values == -4)] = "not tested"
            test_result[p_values == -1] = "exclusively in %s" % stat_parameters["condition_treated"]
            test_result[p_values == -2] = "exclusively in %s" % stat_parameters["condition_wt"]
            
            
            
            condition_up_unadj = (log2_fold_change >= volcano_boundaries[1]) & ((0 <= p_values) & (p_values <= alpha))
            condition_down_unadj = (log2_fold_change <= volcano_boundaries[0]) & ((0 <= p_values) & (p_values <= alpha))
            condition_unreg_unadj = ((volcano_boundaries[0] < log2_fold_change) & (log2_fold_change < volcano_boundaries[1])) | (p_values > alpha)
            
            test_result_unadj = np.array([""] * l, dtype = "object")
            test_result_unadj[condition_up_unadj] = "sig. up regulated in %s" % stat_parameters["condition_treated"]
            test_result_unadj[condition_down_unadj] = "sig. up regulated in %s" % stat_parameters["condition_wt"]
            test_result_unadj[condition_unreg_unadj] = "not regulated"
            test_result_unadj[(p_values == -3) | (p_values == -4)] = "not tested"
            test_result_unadj[p_values == -1] = "exclusively in %s" % stat_parameters["condition_treated"]
            test_result_unadj[p_values == -2] = "exclusively in %s" % stat_parameters["condition_wt"]
            
            
            
            data_frame.insert(col_cnt, "%s log2 FC" % test_name,  log2_fold_change, True)
            col_cnt += 1
            data_frame.insert(col_cnt, "%s tStatistic" % test_name,  t_statistic, True)
            col_cnt += 1
            data_frame.insert(col_cnt, "%s pValue" % test_name,  p_values, True)
            col_cnt += 1
            data_frame.insert(col_cnt, "%s test result" % test_name,  test_result_unadj, True)
            col_cnt += 1
            data_frame.insert(col_cnt, "%s adj pValue" % test_name,  adjusted_p_values, True)
            col_cnt += 1
            data_frame.insert(col_cnt, "%s test result, adjusted" % test_name,  test_result, True)
            col_cnt += 1
            
            
            
            
            
            if with_plotting:
                with_labeling = stat_parameters["labeling_column"] != "" and len(stat_parameters["labeling_column"]) > 0 and stat_parameters["labeling_column"] != "no labeling"
                with_custom = "custom_labeling" in stat_parameters and with_labeling and stat_parameters["custom_labeling"] in data_frame
                
                
                # create main volcano plot
                plt_log2_fold_change = np.array(log2_fold_change[good_p])
                plt_p_values = np.array(p_values[good_p])
                plt_adj_p_values = np.array(adjusted_p_values[good_p])
                plt_reject = reject[good_p]
                m_log_p_values = - np.log10(plt_p_values)
                m_log_adj_p_values = - np.log10(plt_adj_p_values)
                
                condition_up = (plt_log2_fold_change >= volcano_boundaries[1]) & (plt_p_values <= alpha)
                condition_down = (plt_log2_fold_change <= volcano_boundaries[0]) & (plt_p_values <= alpha)
                condition_unreg = ((volcano_boundaries[0] < plt_log2_fold_change) & (plt_log2_fold_change < volcano_boundaries[1])) | (plt_p_values > alpha)
                
                
                custom_labels = np.array([False] * l)
                if with_labeling:
                    if with_custom:
                        valid_labels = set(stat_parameters["custom_labels"])
                        col = stat_parameters["custom_labeling"]
                        for i, row in data_frame.iterrows():
                            custom_labels[i] = (str(row[col]) in valid_labels)
                        custom_labels_plot = custom_labels[good_p]
                            
                    else:
                        custom_labels_plot = custom_labels[good_p]
                        custom_labels_plot[condition_up | condition_down] = True
                            
                
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                exclusively_treated = p_values == -1
                exclusively_wt = p_values == -2
                
                num_subplots = 1 + exclusively_wt.any() + exclusively_treated.any()
                plot_width = 6 + 3 * exclusively_wt.any() + 3 * exclusively_treated.any()
                plot_height = 6
                plot_ratios = [1, 2] if exclusively_wt.any() else [2]
                if exclusively_treated.any(): plot_ratios.append(1)
                
                fig, ax = plt.subplots(1, num_subplots, figsize=(plot_width, plot_height), gridspec_kw={'width_ratios': plot_ratios})#, tight_layout=True)
                if num_subplots == 1: ax_main = ax
                else: ax_main = ax[1] if exclusively_wt.any() else ax[0]
                
                
                # force to have same y scheme
                exclusively_y_min, exclusively_y_max = 1e9, -1
                if exclusively_wt.any():
                    exclusively_y_min = min(means_condition_wt[exclusively_wt])
                    exclusively_y_max = max(means_condition_wt[exclusively_wt])
                if exclusively_treated.any():
                    exclusively_y_min = min(exclusively_y_min, min(means_condition_treated[exclusively_treated]))
                    exclusively_y_max = max(exclusively_y_max, max(means_condition_treated[exclusively_treated]))
                
                
                # create exclusively wild type plot, if necessary
                if exclusively_wt.any():
                    self.plot_exclusive_plot(ax[0],
                                        means_condition_wt[exclusively_wt],
                                        list(data_frame[stat_parameters["labeling_column"]][exclusively_wt]) if with_labeling else None,
                                        stat_parameters["condition_wt"], 
                                        LNTT.COLOR_REGULATION_WT,
                                        LNTT.COLOR_REGULATION_WT_LABEL,
                                        exclusively_y_min,
                                        exclusively_y_max, 
                                        internal_queue,
                                        set(data_frame[stat_parameters["labeling_column"]][custom_labels]) if with_custom else None)
                    
                
                xrange_min, xrange_max, yrange_min, yrange_max = 0, 0, 0, 0
                regulated_exist = np.sum(condition_unreg) < len(plt_p_values)
                
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                if with_labeling and regulated_exist:
                    xx = np.array(plt_log2_fold_change[custom_labels_plot], dtype = "float64")
                    yy = np.array(m_log_p_values[custom_labels_plot], dtype = "float64")
                    labels = list(str(s) for s in data_frame[stat_parameters["labeling_column"]][good_p][custom_labels_plot])
                    label_xx, label_yy = self.automated_annotation(xx, yy, internal_queue)
                    up_down = np.zeros(l, dtype = np.int64)[good_p]
                    up_down[condition_down] = 1
                    up_down[condition_up] = 2
                    regulated = list(up_down[custom_labels_plot])
                    xrange_min, xrange_max = min(xrange_min, np.min(label_xx)), max(xrange_max, np.max(label_xx))
                    yrange_min, yrange_max = min(yrange_min, np.min(label_yy)), max(yrange_max, np.max(label_yy))
                        
                # determine the boundaries of the plot
                xrange_min = min(xrange_min, np.min(plt_log2_fold_change))
                xrange_max = max(xrange_max, np.max(plt_log2_fold_change))
                yrange_min = min(yrange_min, np.min(m_log_p_values))
                yrange_max = max(yrange_max, np.max(m_log_p_values))
                if abs(xrange_max) > abs(xrange_min): xrange_min = -xrange_max
                else: xrange_max = abs(xrange_min)
                

                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                # set plot boundaries
                ax_main.set_xlim(xrange_min - 0.1, xrange_max + 0.1) 
                ax_main.set_ylim(yrange_min - 0.3, yrange_max / 0.8)
                ax_main.grid()

                # plot the dots
                ax_main.scatter(plt_log2_fold_change[condition_up], m_log_p_values[condition_up], color = LNTT.COLOR_REGULATION_TREATED, alpha = 0.5, label = "sig. up regulated in condition '%s' (%i)" % (stat_parameters["condition_treated"], np.sum(condition_up)))
                ax_main.scatter(plt_log2_fold_change[condition_down], m_log_p_values[condition_down], color = LNTT.COLOR_REGULATION_WT, alpha = 0.5, label = "sig. up regulated in condition '%s' (%i)" % (stat_parameters["condition_wt"], np.sum(condition_down)))
                ax_main.scatter(plt_log2_fold_change[condition_unreg], m_log_p_values[condition_unreg], color = LNTT.COLOR_UNREGULATED, alpha = 0.15, label = "not regulated (%i)" % np.sum(condition_unreg))
                
                # plot boundaries
                ax_main.plot([volcano_boundaries[0], volcano_boundaries[0]], [-10e10, 10e10], color = LNTT.COLOR_FC_BOUNDARIES, linewidth = 1, linestyle = "--")
                ax_main.plot([volcano_boundaries[1], volcano_boundaries[1]], [-10e10, 10e10], color = LNTT.COLOR_FC_BOUNDARIES, linewidth = 1, linestyle = "--")
                ax_main.plot([-10e10, 10e10], [-log10(alpha), -log10(alpha)], color = LNTT.COLOR_FC_BOUNDARIES, linewidth = 1, linestyle = "--")
                
                color_code = ['#000000', LNTT.COLOR_REGULATION_WT_LABEL, LNTT.COLOR_REGULATION_TREATED_LABEL]
                
                # plot the labels if desired
                if with_labeling and regulated_exist:
                    for xp, yp, xl, yl, label, reg in zip(xx, yy, label_xx, label_yy, labels, regulated):
                        if len(label) > 10: label = "%s..." % label[:7]
                        ax_main.annotate(label, (xp, yp), xytext=(xl, yl), color = color_code[reg], fontsize = 7, weight = "bold", verticalalignment='center', horizontalalignment='center', arrowprops = dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-90", linewidth=0.5, alpha = 0.5), bbox=dict(pad = -1, facecolor="none", edgecolor="none"))
                    
                
                ax_main.set_xlabel("log${}_2$ fold change")
                ax_main.set_ylabel("-log${}_{10}$ p-value")
                ax_main.set_title("Volcano plot without multiple testing")
                ax_main.legend(loc = "upper right", fontsize = 7)
                
                clean_test_name = test_name.replace("/", "").replace("\\", "")
                clean_test_name = clean_test_name.replace(":", "").replace("*", "")
                clean_test_name = clean_test_name.replace("?", "").replace("!", "")
                clean_test_name = clean_test_name.replace("<", "").replace(">", "")
                clean_test_name = clean_test_name.replace("\"", "").replace("'", "").replace("|", "")
                
                # create exclusively treated plot, if necessary
                if exclusively_treated.any():
                    self.plot_exclusive_plot(ax[1 + exclusively_wt.any()],
                                        means_condition_treated[exclusively_treated],
                                        list(data_frame[stat_parameters["labeling_column"]][exclusively_treated]) if with_labeling else None,
                                        stat_parameters["condition_treated"], 
                                        LNTT.COLOR_REGULATION_TREATED,
                                        LNTT.COLOR_REGULATION_TREATED_LABEL,
                                        exclusively_y_min,
                                        exclusively_y_max,
                                        internal_queue,
                                        set(data_frame[stat_parameters["labeling_column"]][custom_labels]) if with_custom else None)
                
                
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                fig_filename = os.path.join(output_folder, "%s_volcano.pdf" % clean_test_name)
                plt.savefig(fig_filename, dpi = 600)
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                fig_filename = os.path.join(output_folder, "%s_volcano.png" % clean_test_name)
                plt.savefig(fig_filename, dpi = 600)
                plt.cla()   # Clear axis
                plt.clf()   # Clear figure
                plt.close() # Close a figure window






                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass





                # plot p-value distribution
                n, bins, patches = plt.hist(plt_p_values, 50, facecolor = LNTT.COLOR_DISTRIBUTION_BAR, alpha=0.75)
                plt.xlabel("p-value")
                plt.ylabel("occurrence")
                plt.title("The p-value distribution")
                #plt.tight_layout()
                fig_filename = os.path.join(output_folder, "%s_histogram.pdf" % clean_test_name)
                plt.savefig(fig_filename, dpi = 600)
                fig_filename = os.path.join(output_folder, "%s_histogram.png" % clean_test_name)
                plt.savefig(fig_filename, dpi = 600)
                plt.cla()   # Clear axis
                plt.clf()   # Clear figure
                plt.close() # Close a figure window


                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                
                
                




                fig, ax = plt.subplots(1, num_subplots, figsize=(plot_width, plot_height), gridspec_kw={'width_ratios': plot_ratios}) #, tight_layout=True)
                if num_subplots == 1: ax_main = ax
                else: ax_main = ax[1] if exclusively_wt.any() else ax[0]
                

                
                # compute volcano plot with adjusted p-values
                condition_up = (plt_log2_fold_change >= volcano_boundaries[1]) & (plt_reject)
                condition_down = (plt_log2_fold_change <= volcano_boundaries[0]) & (plt_reject)
                condition_unreg = ((volcano_boundaries[0] < plt_log2_fold_change) & (plt_log2_fold_change < volcano_boundaries[1])) | (~plt_reject)
                regulated_exist = np.sum(condition_unreg) < len(plt_p_values)
                
                custom_labels = np.array([False] * l)
                if with_labeling:
                    if with_custom:
                        valid_labels = set(stat_parameters["custom_labels"])
                        col = stat_parameters["custom_labeling"]
                        for i, row in data_frame.iterrows():
                            custom_labels[i] = (str(row[col]) in valid_labels)
                        custom_labels_plot = custom_labels[good_p]
                        
                    else:                        
                        custom_labels_plot = custom_labels[good_p]
                        custom_labels_plot[condition_up | condition_down] = True
                
                
                # force to have same y scheme
                exclusively_y_min, exclusively_y_max = 1e9, -1
                if exclusively_wt.any():
                    exclusively_y_min = min(means_condition_wt[exclusively_wt])
                    exclusively_y_max = max(means_condition_wt[exclusively_wt])
                if exclusively_treated.any():
                    exclusively_y_min = min(exclusively_y_min, min(means_condition_treated[exclusively_treated]))
                    exclusively_y_max = max(exclusively_y_max, max(means_condition_treated[exclusively_treated]))
                
                
                # create exclusively wild type plot, if necessary
                if exclusively_wt.any():
                    self.plot_exclusive_plot(ax[0],
                                        means_condition_wt[exclusively_wt],
                                        list(data_frame[stat_parameters["labeling_column"]][exclusively_wt]) if with_labeling else None,
                                        stat_parameters["condition_wt"], 
                                        LNTT.COLOR_REGULATION_WT,
                                        LNTT.COLOR_REGULATION_WT_LABEL,
                                        exclusively_y_min,
                                        exclusively_y_max,
                                        internal_queue,
                                        set(data_frame[stat_parameters["labeling_column"]][custom_labels]) if with_custom else None)
                   
                
                xrange_min, xrange_max, yrange_min, yrange_max = 0, 0, 0, 0
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                
                if with_labeling and regulated_exist:
                    xx = np.array(plt_log2_fold_change[custom_labels_plot], dtype = "float64")
                    yy = np.array(m_log_adj_p_values[custom_labels_plot], dtype = "float64")
                    labels = list(str(s) for s in data_frame[stat_parameters["labeling_column"]][good_p][custom_labels_plot])
                    label_xx, label_yy = self.automated_annotation(xx, yy, internal_queue)
                    up_down = np.zeros(l, dtype = np.int64)[good_p]
                    up_down[condition_down] = 1
                    up_down[condition_up] = 2
                    regulated = list(up_down[custom_labels_plot])
                    xrange_min, xrange_max = min(xrange_min, np.min(label_xx)), max(xrange_max, np.max(label_xx))
                    yrange_min, yrange_max = min(yrange_min, np.min(label_yy)), max(yrange_max, np.max(label_yy))
                        
                # determine the boundaries of the plot
                xrange_min = min(xrange_min, np.min(plt_log2_fold_change))
                xrange_max = max(xrange_max, np.max(plt_log2_fold_change))
                yrange_min = min(yrange_min, np.min(m_log_adj_p_values))
                yrange_max = max(yrange_max, np.max(m_log_adj_p_values))
                if abs(xrange_max) > abs(xrange_min): xrange_min = -xrange_max
                else: xrange_max = abs(xrange_min)
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                        
                        
                # set plot boundaries
                ax_main.set_xlim(xrange_min - 0.1, xrange_max + 0.1) 
                ax_main.set_ylim(yrange_min - 0.3, yrange_max / 0.8)
                ax_main.grid()
                
                
                ax_main.scatter(plt_log2_fold_change[condition_up], m_log_adj_p_values[condition_up], color = LNTT.COLOR_REGULATION_TREATED, alpha = 0.5, label = "sig. up regulated in condition '%s' (%i)" % (stat_parameters["condition_treated"], np.sum(condition_up)))
                ax_main.scatter(plt_log2_fold_change[condition_down], m_log_adj_p_values[condition_down], color = LNTT.COLOR_REGULATION_WT, alpha = 0.5, label = "sig. up regulated in condition '%s' (%i)" % (stat_parameters["condition_wt"], np.sum(condition_down)))
                ax_main.scatter(plt_log2_fold_change[condition_unreg], m_log_adj_p_values[condition_unreg], color = LNTT.COLOR_UNREGULATED, alpha = 0.15, label = "not regulated (%i)" % np.sum(condition_unreg))
                
                
                ax_main.plot([volcano_boundaries[0], volcano_boundaries[0]], [-10e10, 10e10], color = LNTT.COLOR_FC_BOUNDARIES, linewidth = 1, linestyle = "--")
                ax_main.plot([volcano_boundaries[1], volcano_boundaries[1]], [-10e10, 10e10], color = LNTT.COLOR_FC_BOUNDARIES, linewidth = 1, linestyle = "--")
                ax_main.plot([-10e10, 10e10], [-log10(alpha), -log10(alpha)], color = LNTT.COLOR_FC_BOUNDARIES, linewidth = 1, linestyle = "--")
                
                
                color_code = ['#000000', LNTT.COLOR_REGULATION_WT_LABEL, LNTT.COLOR_REGULATION_TREATED_LABEL]
                if with_labeling and regulated_exist:
                    for xp, yp, xl, yl, label, reg in zip(xx, yy, label_xx, label_yy, labels, regulated):
                        try:
                            result = internal_queue.get_nowait()
                            if result != None:
                                self.interrupt = True
                                return
                        except:
                            pass
                        if len(label) > 10: label = "%s..." % label[:7]
                        ax_main.annotate(label, (xp, yp), xytext=(xl, yl), color = color_code[reg], fontsize = 7, weight = "bold", verticalalignment='center', horizontalalignment='center', arrowprops = dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-90", linewidth=0.5, alpha = 0.5), bbox=dict(pad = -1, facecolor="none", edgecolor="none"))
                
                ax_main.set_xlim(xrange_min - 0.1, xrange_max + 0.1) 
                ax_main.set_ylim(yrange_min - 0.1, yrange_max / 0.8)
                ax_main.set_xlabel("log${}_2$ fold change")
                ax_main.set_ylabel("-log${}_{10}$ p-value")
                ax_main.set_title("Volcano plot with multiple testing correction")
                ax_main.legend(loc = "upper right", fontsize = 7)
                exclusively_treated_ax = None
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                
                # create exclusively treated plot, if necessary
                if exclusively_treated.any():
                    self.plot_exclusive_plot(ax[1 + exclusively_wt.any()],
                                        means_condition_treated[exclusively_treated],
                                        list(data_frame[stat_parameters["labeling_column"]][exclusively_treated]) if with_labeling else None,
                                        stat_parameters["condition_treated"], 
                                        LNTT.COLOR_REGULATION_TREATED,
                                        LNTT.COLOR_REGULATION_TREATED_LABEL,
                                        exclusively_y_min,
                                        exclusively_y_max,
                                        internal_queue,
                                        set(data_frame[stat_parameters["labeling_column"]][custom_labels]) if with_custom else None)
                
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                fig_filename = os.path.join(output_folder, "%s_volcano_adjusted.pdf" % clean_test_name)
                plt.savefig(fig_filename, dpi = 600)
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                fig_filename = os.path.join(output_folder, "%s_volcano_adjusted.png" % clean_test_name)
                plt.savefig(fig_filename, dpi = 600)
                plt.cla()   # Clear axis
                plt.clf()   # Clear figure
                plt.close() # Close a figure window
                
    
    

        elif test_method == "anova": # T-Test
            if "pval_adjust" in parameters and parameters["pval_adjust"] not in pval_adjust_methods:
                print("Error: p-value correction method '%s' unknown. Supported methods:\n - %s" % (parameters["pval_adjust"], "\n - ".join(m for m in pval_adjust_methods)))
                exit()

            # getting all necessary parameters for T-Test
            alpha = stat_parameters["significance_level"] if "significance_level" in stat_parameters else 0.05
            with_normalization = parameters["with_normalization"]
            pval_adjust = stat_parameters["pval_adjust"] if "pval_adjust" in stat_parameters else "fdr_bh"
            filter_out = float(stat_parameters["min_values_per_condition"]) if "min_values_per_condition" in stat_parameters else 0
            use_log_values = stat_parameters["log_values"] if "log_values" in stat_parameters else False
            with_plotting = stat_parameters["with_stat_test_plotting"] if "with_stat_test_plotting" in stat_parameters else False
            
            
            
            # setting default parameters when not provided by user and warn
            test_name = "ANOVA-%s" % "-".join(stat_parameters["anova_conditions"])
            self.logging.put("Performing statistical testing on '%s'" % test_name)
            if "pval_adjust" not in stat_parameters: self.logging.put("Warning: no p-value correction was defined, 'fdr_bh' is selected by default.")
            if "significance_level" not in stat_parameters: self.logging.put("Warning: no significance level is defined, 0.05 is selected by default.")
            if "log_fc_thresholds" not in stat_parameters and with_plotting: self.logging.put("Warning: no log2 fold change thresholds are defined, [-1, 1] is selected by default.")
            if "log_values" not in stat_parameters: self.logging.put("Warning, no information about the usage of log values provided, using regular values for test on default.")
            
        
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            
            column_names = set()
            condition_columns = {}
            
            for condition in stat_parameters["anova_conditions"]:
                condition_columns[condition] = []
                for c in conditions[condition]:
                    col_name = "%s%s" % (c, normalized_suffix if with_normalization else "")
                    column_names.add(col_name)
                    condition_columns[condition].append(col_name)
                
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            if report != None: report.append("ANOVA test was performed%s on the conditions '%s'." % ("using log-transformed data" if use_log_values else "", ", ".join(stat_parameters["anova_conditions"])))
            
            col_cnt = len([c for c in data_frame])
            
            # T test
            l = data_frame.shape[0]
            p_values = np.zeros(l, dtype = "float64")
            f_statistic = np.zeros(l, dtype = "float64")
                
            
            for index, row in data_frame[column_names].iterrows():
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                
                anova_conditions = []
                fine_for_test = True
                for condition in stat_parameters["anova_conditions"]:
                    values = np.zeros(len(condition_columns[condition]), dtype = np.float64)
                    
                    
                    for i, val in enumerate(row[condition_columns[condition]]):
                        values[i] = np.log(val) if use_log_values else val
                        
                    values_no_nan = values[~np.isnan(values)]
                    
                    if len(values_no_nan) == 0:
                        p_values[index] = -1
                        fine_for_test = False
                        break
                    elif len(values_no_nan) == 1 or len(values_no_nan) / len(values) < filter_out:
                        p_values[index] = -2
                        fine_for_test = False
                        break
                    
                    anova_conditions.append(values_no_nan)
                    
                if fine_for_test:
                    values = stats.f_oneway(*anova_conditions)
                    f_statistic[index] = values[0]
                    p_values[index] = values[1]
            
            
            good_p = p_values >= 0
            valid_p_values = np.array(p_values[good_p])
            f_statistic[p_values < 0] = np.nan
            
            
            if len(valid_p_values) > 0:
                # p value correction
                adjusted_p_values_total = multitest.multipletests(valid_p_values, alpha = alpha, method = pval_adjust, is_sorted = False)
                reject_tmp, adjusted_p_values_tmp = adjusted_p_values_total[:2]
                
            else:
                reject_tmp, adjusted_p_values_tmp = [], []
            
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            
            # retrieving original vector length after appyling correction function
            reject, pos = np.empty(l, dtype = np.bool_), 0
            adjusted_p_values = np.empty(l, dtype = np.float64)
            for i, gp in enumerate(good_p):
                reject[i] = False
                adjusted_p_values[i] = p_values[i]
                if gp:
                    reject[i] = reject_tmp[pos]
                    adjusted_p_values[i] = adjusted_p_values_tmp[pos]
                    pos += 1
            
            
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            
            # compute fold changes and plot volcano plot with regular p-values
            
            test_result = np.array([""] * l, dtype = "object")
            test_result[reject] = "sig. regulated"
            test_result[reject == 0] = "not regulated"
            test_result[adjusted_p_values < 0] = "not tested, insufficient set(s)"
            
            
            
            data_frame.insert(col_cnt, "%s fStatistic" % test_name,  f_statistic, True)
            col_cnt += 1
            data_frame.insert(col_cnt, "%s p­Value" % test_name,  p_values, True)
            col_cnt += 1
            data_frame.insert(col_cnt, "%s adj p­Value" % test_name,  adjusted_p_values, True)
            col_cnt += 1
            data_frame.insert(col_cnt, "%s test result" % test_name,  test_result, True)
            col_cnt += 1
            
            
    
            
            if with_plotting:
                
                clean_test_name = test_name.replace("/", "").replace("\\", "")
                clean_test_name = clean_test_name.replace(":", "").replace("*", "")
                clean_test_name = clean_test_name.replace("?", "").replace("!", "")
                clean_test_name = clean_test_name.replace("<", "").replace(">", "")
                clean_test_name = clean_test_name.replace("\"", "").replace("'", "").replace("|", "")
                
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                
                
                # plot p-value distribution
                n, bins, patches = plt.hist(p_values[p_values >= 0], 50, facecolor = LNTT.COLOR_DISTRIBUTION_BAR, alpha = 0.75)
                plt.xlabel("p-value")
                plt.ylabel("occurrence")
                plt.title("The p-value distribution")
                #plt.tight_layout()
                fig_filename = os.path.join(output_folder, "%s_histogram.pdf" % clean_test_name)
                plt.savefig(fig_filename, dpi = 600)
                fig_filename = os.path.join(output_folder, "%s_histogram.png" % clean_test_name)
                plt.savefig(fig_filename, dpi = 600)
                plt.cla()   # Clear axis
                plt.clf()   # Clear figure
                plt.close() # Close a figure window               
                
                
                
                
                
    
    
        
        
    def plot_exclusive_plot(self, ax, values, labels, condition, point_color, label_color, yrange_min, yrange_max, internal_queue, custom_labels = None):
        yrange_min = log10(yrange_min) - 0.2
        yrange_max = log10(yrange_max) + 0.2
        x_values = np.zeros(len(values))
        y_values = np.array(np.log10(values))
        y_values.sort()
        ax.scatter(x_values, y_values, color = point_color, alpha = 0.5)
        ax.set_ylabel("log${}_{10}$ abundance")
        ax.set_xticks([])
        ax.grid()
        
        if labels != None:
            step = (yrange_max - yrange_min) / (len(y_values) - 1) if len(y_values) > 1 else 0
            for i, (yp, label) in enumerate(zip(y_values, labels)):
                if custom_labels != None and label not in custom_labels: continue
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                xl = -1 + 2 * (i & 1)
                yl = yrange_min + i * step if step > 0 else y_values[0]
                ax.annotate(label, (0, yp), xytext=(xl, yl), color = label_color, fontsize = 7, weight = "bold", verticalalignment='center', horizontalalignment='center', arrowprops = dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-90", linewidth=0.5, alpha = 0.5), bbox=dict(pad = -1, facecolor="none", edgecolor="none"))

        # set plot boundaries
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(yrange_min - 0.1, yrange_max + 0.1)
        ax.set_title("Exclusively in %s (%i)" % (condition, len(values)))
        
        
        
        
        
    def automated_annotation(self, xx, yy, internal_queue):
        label_xx, label_yy, l = np.array(xx, dtype = "float64"), np.array(yy, dtype = "float64"), len(xx)

        for rep in range(9):
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            all_xx = np.concatenate([xx, label_xx])
            all_yy = np.concatenate([yy, label_yy])
            
            a = np.concatenate([all_xx, all_yy])
            a = a.reshape([2 * l, 2])
            decay_factor = 0.5 / (np.sum(metrics.pairwise_distances(a, a)) / (2 * l)**2)
            
            for i in range(l):
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                l_xx, l_yy = label_xx[i], label_yy[i]
                d = np.sqrt((l_xx - all_xx)**2 + (l_yy - all_yy)**2)
                new_d = (decay_factor / (d[d != 0] + 1)**2) / d[d != 0]
                
                label_xx[i] = xx[i] + np.sum((l_xx - all_xx)[d != 0] * new_d)
                label_yy[i] = yy[i] + np.sum((l_yy - all_yy)[d != 0] * new_d * 1.3)
                
        return label_xx, label_yy
        
        
        
        
        
            
        
        
    def gene_name_request(self, data_frame, parameters, internal_queue, report):
        accession_column = parameters["accession_column"]
        
        date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        
        if report != None: report.append("Gene names are retrieved by accession numbers using Uniprot API (time stamp: %s)." % date_time)
        self.logging.put("Gene names are retrieved")
        max_accession_per_request = 500
        
        uniprot_url = 'https://www.uniprot.org/uploadlists/'
        
        accessions = list(data_frame[accession_column])
        num_accessions = len(accessions)
        accession_pos = {accession.split(";")[0]: i for i, accession in enumerate(accessions)}
        gene_names = list(accessions)
        
        requests = 0
        while requests < num_accessions:
            try:
                result = internal_queue.get_nowait()
                if result != None:
                    self.interrupt = True
                    return
            except:
                pass
            max_val = min(requests + max_accession_per_request, num_accessions)
            params = {
            'from': 'ACC+ID',
            'to': 'GENE+NAME',
            'format': 'tab',
            'query': " ".join(accessions[requests : max_val])
            }
            requests = max_val

            data = urllib.parse.urlencode(params)
            data = data.encode('utf-8')
            req = urllib.request.Request(uniprot_url, data)
            try:
                with urllib.request.urlopen(req, timeout=10) as f:
                    response = f.read()
                    for i, line in enumerate(response.decode('utf-8').split("\n")):
                        if self.interrupt: return
                        if i == 0 or len(line) == 0: continue
                        tokens = line.split("\t")
                        if tokens[0] in accession_pos:
                            gene_names[accession_pos[tokens[0]]] = tokens[1]
                self.logging.put("Already %i of %i accessions requested" % (requests, num_accessions))
                
            except:
                pass
                
        data_frame["Genes"] = gene_names




    def pairwise_log_fc(self, data_frame, output_folder_normalization, column_names, parameters, internal_queue, report):
        global normalized_suffix
        self.logging.put("Computing pairwise log fold changes over all columns")
        columns = column_names
        if "with_normalization" in parameters and parameters["with_normalization"]:
            columns = ["%s%s" % (c, normalized_suffix) for c in column_names]
            
        col_cnt = data_frame.shape[1]
        
        for i in range(len(columns) - 1):
            for j in range(i + 1, len(columns)):
                try:
                    result = internal_queue.get_nowait()
                    if result != None:
                        self.interrupt = True
                        return
                except:
                    pass
                title = "%s/%s log2 FC" % (columns[i], columns[j])
                log_fc = np.log(data_frame[columns[i]] / data_frame[columns[j]])
                data_frame.insert(col_cnt, title,  log_fc, True)
                col_cnt += 1





    def process(self, parameters, internal_queue):
        
        try:
            global pval_adjust_methods, ttest_types
            
            report = ["Data processing was performed as follows:"]
                
            # load necessary parameters
            output_folder = parameters["output_folder"]
            data_file = parameters["data_file"]
            data_sheet = parameters["data_sheet"]
            
            
            # load column names from parameter file
            conditions = parameters["conditions"]
            column_names = []
            for c, col_names in conditions.items(): column_names += col_names
            
            
            if "with_normalization" in parameters and parameters["with_normalization"] and parameters["norm_ref_condition"] not in conditions:
                print("Error: normalization reference condition '%s' does not appear in conditions list" % parameters["norm_ref_condition"])
                self.logging.put(None)
                return
            
                    
            self.logging.put("Loading data sheet")
            
                
            # load the table file
            skip_rows = parameters["skip_rows"] if "skip_rows" in parameters else []
            na_values = parameters["na_values"] if "na_values" in parameters else []
            delimiter = parameters["delimiter"] if "delimiter" in parameters else None
            na_values.append(0)
            na_values.append("0")
            try:
                data_frame = pd.read_excel(data_file, data_sheet, skiprows = skip_rows, na_values = na_values)
            except:
                try:
                    data_frame = pd.read_csv(data_file, delimiter = delimiter, skiprows = skip_rows, na_values = na_values)
                except:
                    print("Error: file '%s' could not be open." % data_file)
                    self.logging.put(None)
                    return
                
            try:
                if self.interrupt or internal_queue.get_nowait() != None:
                    self.logging.put("Analysis interrupted")
                    self.logging.put(None)
                    return
            except:
                pass
                
            # check if all provided column names are in table
            for col_name in column_names:
                if col_name not in data_frame:
                    self.logging.put("Error: table does not contain column name '%s'" % col_name)
                    self.logging.put(None)
                    return
                    
                    
                    
            
            # check for negative values
            for c in column_names:
                try:
                    if np.sum(data_frame[c] < 1e-15) > 1:
                        self.logging("Error: values encounted which are less or equal to zero in column '%s'. Please use only positive non-ratio numbers or declare zero values as not available 'na_values' in the parameter file." % c)
                        self.logging.put(None)
                        return
                except:
                    self.logging.put("An error was encounted, non numerical values were found in column '%s'." % c)
                    self.logging.put(None)
                    return
            
                
            try:
                if self.interrupt or internal_queue.get_nowait() != None:
                    self.logging.put("Analysis interrupted")
                    self.logging.put(None)
                    return
            except:
                pass
                    
            
            # creating all output folders, if necessary
            output_folder_normalization = os.path.join(output_folder, "Normalization-Figures")
            output_folder_stat_test = os.path.join(output_folder, "Stat-Test-Figures")
            
            if parameters["with_normalization"]: 
                try:
                    os.makedirs(output_folder_normalization)
                except:
                    pass
            
            try:
                if self.interrupt or internal_queue.get_nowait() != None:
                    self.logging.put("Analysis interrupted")
                    self.logging.put(None)
                    return
            except:
                pass
            
            if "statistical_testing" in parameters and len(parameters["statistical_testing"]) > 0:
                try:
                    os.makedirs(output_folder_stat_test)
                except:
                    pass
                
            if parameters["with_gene_name_request"] and ("accession_column" not in parameters or parameters["accession_column"] not in data_frame):
                print("Error: can not request gene names, accession column unknown.")
                self.logging.put(None)
                return
            
            
            if "with_data_imputation" in parameters and parameters["with_data_imputation"] and "data_imputation_value" not in parameters:
                print("Error: data imputation enabled, but no 'data_imputation_value' parameter provided in parameter file.")
                self.logging.put(None)
                return


            try:
                if self.interrupt or internal_queue.get_nowait() != None:
                    self.logging.put("Analysis interrupted")
                    self.logging.put(None)
                    return
            except:
                pass
            #############################################################################
            # doing the magic
            #############################################################################

            report.append("The initial list contained %i entries." % len(data_frame))
            self.logging.put("Start processing with %i entries." % len(data_frame))

            # normalization
            if "with_normalization" in parameters and parameters["with_normalization"]: self.normalization(data_frame, output_folder_normalization, column_names, conditions, parameters, internal_queue, report)

            try:
                if self.interrupt or internal_queue.get_nowait() != None:
                    self.logging.put("Analysis interrupted")
                    self.logging.put(None)
                    return
            except:
                pass


            # data imputation
            if "with_data_imputation" in parameters and parameters["with_data_imputation"]:
                self.data_frame_imputation(data_frame, column_names, parameters, internal_queue, report)

            try:
                if self.interrupt or internal_queue.get_nowait() != None:
                    self.logging.put("Analysis interrupted")
                    self.logging.put(None)
                    return
            except:
                pass



            # pairwise log_fc 
            if "pairwise_log_fc" in parameters and parameters["pairwise_log_fc"]: self.pairwise_log_fc(data_frame, output_folder_normalization, column_names, parameters, internal_queue, report)

            try:
                if self.interrupt or internal_queue.get_nowait() != None:
                    self.logging.put("Analysis interrupted")
                    self.logging.put(None)
                    return
            except:
                pass
            
            
            

            # cv filtering
            if "with_cv_threshold" in parameters and parameters["with_cv_threshold"]: self.cv_filtering(data_frame, column_names, parameters, internal_queue, report)
            
            try:
                if self.interrupt or internal_queue.get_nowait() != None:
                    self.logging.put("Analysis interrupted")
                    self.logging.put(None)
                    return
            except:
                pass
            
            
            # requesting gene names
            if "with_gene_name_request" in parameters and parameters["with_gene_name_request"]: self.gene_name_request(data_frame, parameters, internal_queue, report)

            try:
                if self.interrupt or internal_queue.get_nowait() != None:
                    self.logging.put("Analysis interrupted")
                    self.logging.put(None)
                    return
            except:
                pass
            

            # statistical tesing
            if "statistical_testing" in parameters and len(parameters["statistical_testing"]) > 0:
                if report != None: report.append("Statistical testing was done on the following conditions: '%s'." % "', '".join(c for c in conditions))
                for i, stat_parameters in enumerate(parameters["statistical_testing"]):
                    self.statistical_tesing(i, data_frame, stat_parameters, conditions, output_folder_stat_test, parameters, internal_queue, report)


            try:
                if self.interrupt or internal_queue.get_nowait() != None:
                    self.logging.put("Analysis interrupted")
                    self.logging.put(None)
                    return
            except:
                pass



            # storing the data table with all additional columns
            try:
                head, filename = os.path.split(data_file)
                base, file_extension = os.path.splitext(data_file)
                output_data_file = os.path.join(output_folder, filename.replace(file_extension, "_with_analysis_data.xlsx"))
                self.logging.put("Storing the data table with all additional columns in '%s'" % output_data_file)
                data_frame.to_excel(output_data_file, index = False)
            except:
                self.logging.put("An error occurred while storing the output table.")
                self.logging.put(None)
            
            
            try:
                report_filename = os.path.join(output_folder, "Report.txt")
                with open(report_filename, "wt") as report_file:
                    report_file.write(("%s\n") % " ".join(report))
                self.logging.put("Report document was stored in '%s'" % output_folder)
                self.logging.put("Process has sucessfully finished")
            except:
                self.logging.put("An error occurred while storing the output table.")

            self.logging.put(None)
            
        except Exception:
            self.logging.put("An unexpected error occurred: please contact the tool developer providing the following error message:")
            self.logging.put(str(traceback.format_exc()))
            self.logging.put(None)
    
    
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("usage: python3 %s parameter_file" % sys.argv[0])
        exit()
        
    try:
        parameter_file = sys.argv[1]
        parameters = json.loads(open(parameter_file, "rt").read())
        
    except:
        print("Your parameter file '%s' is not readable or seems to be corrupted." % parameter_file)
        exit()
        
    lntt_queue = multiprocessing.Queue()
    main_queue = multiprocessing.Queue()
    lntt = LNTT(lntt_queue, main_queue)
    lntt.start()
    lntt_queue.put(parameters)
    while True:
        result = main_queue.get()
        if type(result) == str:
            date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            print("%s: %s" % (date_time, result))
        else:
            break
    lntt_queue.put(None)
    

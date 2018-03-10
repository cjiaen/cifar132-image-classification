# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:42:17 2017

@author: cjiaen
"""

import os
import sys
import pandas as pd
import numpy as np

reference = pd.read_csv(r"C:\Users\cjiaen\Documents\Sem1\CS5242_NNDL\Kaggle\train.csv")

pred_incep = pd.read_csv(r"C:\Users\cjiaen\Documents\Sem1\CS5242_NNDL\Kaggle\validation\predictions_incep_40_valid.csv")
pred_incep = pd.read_csv(r"C:\Users\cjiaen\Documents\Sem1\CS5242_NNDL\Kaggle\validation\predictions_irv2_40_valid.csv")

pred_incep = pd.merge(left=pred_incep, right=reference, how='inner', on='image_name', suffixes=('_pred','_label'))

error_analysis = pd.DataFrame(index = range(132),
                              columns = range(132),
                              data = 0)

for idx in range(len(pred_incep)):
    error_analysis.loc[pred_incep.iloc[idx,1], pred_incep.iloc[idx,2]] += 1


analysis = pd.DataFrame(index=range(132), columns = ['sensitivity', 'precision'])
for cat in range(132):
    analysis.iloc[cat,0] = error_analysis.iloc[cat,cat]/np.sum(error_analysis.iloc[:,cat])
    analysis.iloc[cat,1] = error_analysis.iloc[cat,cat]/np.sum(error_analysis.iloc[cat,:])

analysis.to_csv("error_analysis_incep.csv")

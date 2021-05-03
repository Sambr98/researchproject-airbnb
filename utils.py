import numpy as np
import pandas as pd
import scipy as sp

def get_corr_pval(df, xs_labels, ys_labels):
    xs = []
    for lbl in xs_labels:
        new = np.array(df[lbl])
        new = new.reshape(new.shape[0],1)
        xs.append(new)
    ys = []
    for lbl in ys_labels:
        new = np.array(df[lbl])
        ys.append(new)
    
    df_corr = pd.DataFrame(columns=ys_labels)
    df_pval = pd.DataFrame(columns=ys_labels)
        
    for i in range(len(xs)):
        new_row_corr = {}
        new_row_pval = {}
        for j in range(len(ys)):
            new_row_corr[ys_labels[j]] = sp.stats.spearmanr(xs[i], ys[j]).correlation
            new_row_pval[ys_labels[j]] = sp.stats.spearmanr(xs[i], ys[j]).pvalue
        df_corr = df_corr.append(new_row_corr, ignore_index=True)
        df_pval = df_pval.append(new_row_pval, ignore_index=True)
    for idx in range(df_corr.shape[0]):
        df_corr = df_corr.rename({df_corr.index[idx]: xs_labels[idx]})
        df_pval = df_pval.rename({df_pval.index[idx]: xs_labels[idx]})
    
    return (df_corr, df_pval)

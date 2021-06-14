import pandas as pd
import matplotlib.pyplot as plt

def groupPloting(df,groups,length) :
    list_group_by=df[['Sale']+groups].groupby(groups).mean().values
    list_group_by = list(map(lambda x : x[0],list_group_by))[:length]
    index_group_by = list(map(lambda x : '('+str(x[0])[:2]+str(x[1])[:2]+')',df[['Sale']+groups].groupby(groups).mean().index))[:length]
    plt.bar(index_group_by,list_group_by)
    return plt
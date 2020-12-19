import pandas as pd

df1 = pd.read_csv('D:\Documents\GitHub\TUHH_ISM_tumor_detection\csv_features\groundtruth_train.csv',usecols=[0])
df2 = pd.read_csv('D:\Documents\GitHub\TUHH_ISM_tumor_detection\csv_features\\temp.csv',usecols=[0])


# for row in range(5):
#     print(df1.values[row][0] == df2.values[row][0])


for row in range(len(df1)):
    some_no = df1.values[row][0]
    if df2[df2['imageID']==some_no].index.values != 0 :
        pass
    else:
        print(some_no)
    
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

'''
@2016.11.01 by wttttt
@kaggel housePrices
'''
train_df = pd.read_csv('train.csv')
# convert the nominal attr to numbers
train_df['MSZoning1']=train_df['MSZoning'].map({'A':0,'C (all)':7,'FV':1,'I':2,'RH':6,'RL':3,'RP':5,'RM':4})
'''
#save file
train_df['MSZoning'].hist()
out_png ='test.png'
plt.savefig(out_png,dpi=150)
'''
# fill the missing value with mean()
train_df['LotFrontage1']=train_df['LotFrontage'].fillna(value=train_df['LotFrontage'].mean())  
train_df['Street1'] = train_df['Street'].map(lambda x: 1 if x=='Pave' else 0)

for s in ['LotShape','LandContour','Utilities','LotConfig','LandSlope']:
    h_dummies = pd.get_dummies(train_df[s],prefix=s)
    train_df.drop([s],axis=1,inplace=True)
    train_df=train_df.join(h_dummies)

'''
train_df['Alley1']=train_df['Alley'].map({'Pave':1,'Grvl':0.5})
train_df['Alley1']=train_df['Alley1'].fillna(0)
train_df['LotShape1'] = train_df['LotShape'].map({'Reg':0,'IR1':1,'IR2':2,'IR3':3})
train_df['LandContour1'] = train_df['LandContour'].map({'Lvl':0,'Bnk':1,'HLS':2,'Low':3})
train_df['Utilities1'] = train_df['Utilities'].map({'AllPub':0,'NoSewr':1,'NoSeWa':2,'ELO':3})
train_df['LotConfig1'] = train_df['LotConfig'].map({'Inside':0,'Corner':1,'CulDSac':2,'FR2':3,'FR3':4})
train_df['LandSlope1'] = train_df['LandSlope'].map({'Gtl':0,'Mod':1,'Sev':2})
'''

import warnings
import numpy as np
import pandas as pd
import gc
import pickle

# import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
from sklearn.ensemble import RandomForestClassifier

data_path = '/'
train_path = 'tcdata/hy_round1_train_20200102/'
# test_path = 'hy_round1_testA_20200102/'
# test_path2 = 'hy_round1_testB_20200221/'
test_path1 = 'tcdata/hy_round2_train_20200225/'
test_path2 = 'tcdata/hy_round2_testB_20200312/'

train_files = os.listdir(data_path+train_path)

test_files1 = os.listdir(data_path+test_path1)
test_files2 = os.listdir(data_path+test_path2)

ret = []
for file in tqdm(train_files):
    df_train = pd.read_csv(data_path+train_path+file)
    ret.append(df_train)
df_train = pd.concat(ret)
df_train.columns = ['ship','x','y','v','d','time','type']
# gc.collect()
ret = []
for file in tqdm(test_files1):
    df_test1 = pd.read_csv(data_path+test_path1+file)
    ret.append(df_test1)
df_test1 = pd.concat(ret)
df_test1.columns = ['ship','x','y','v','d','time','type']
# gc.collect()

ret = []
for file in tqdm(test_files2):
    df_test2 = pd.read_csv(data_path+test_path2+file)
    ret.append(df_test2)
df_test2 = pd.concat(ret)
df_test2.columns = ['ship','x','y','v','d','time']

gc.collect()

df_train_x = df_train['x'].copy()
df_train_y = df_train['y'].copy()
# df_train['x'] = df_train_y.apply(lambda x:round(x*(.053/5605)-24.797,3))
# df_train['y'] = df_train_x.apply(lambda x:round(x*(.085/8544)+56.618,3))

df_train['x'] = df_train_y.apply(lambda x:round(x/100000-28,3))
df_train['y'] = df_train_x.apply(lambda x:round(x/100000+56,3))


df_train = df_train.reset_index(drop = True)
df_train = df_train[df_train.v<15].reset_index(drop = True)
df_test1 = df_test1.reset_index(drop = True)
df_test1 = df_test1[df_test1.v<15].reset_index(drop = True)
df_test2 = df_test2.reset_index(drop = True)
df_test2 = df_test2[df_test2.v<15].reset_index(drop = True)
df_test2['type'] = '未知'

df_all = pd.concat([df_train,df_test1,df_test2],axis = 0)

badlat_ship = df_all[df_all.x<10].ship.unique()
for id_ in badlat_ship:
    df_all.loc[(df_all.ship ==id_)&(df_all.x<10),['x','y']] = np.nan
    df_all.loc[df_all.ship ==id_,['x','y']]=df_all[df_all.ship==id_][['x','y']].interpolate().fillna(method='bfill').values
    
df_all['d_sin'] = df_all['d'].apply(lambda x:np.sin(x/180*np.pi))
# np.sin(90/180*np.pi)
df_all['d_cos'] = df_all['d'].apply(lambda x:np.cos(x/180*np.pi))

df_all['x'] = df_all['x'].apply(lambda x:round(x,3))
df_all['y'] = df_all['y'].apply(lambda x:round(x,3))

def extract_dt(df):
    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.weekday
    return df

all_data = extract_dt(df_all)

all_data['x_diff'] = np.abs(all_data.x.diff(-1))
all_data.loc[all_data.drop_duplicates('ship',keep='last').index,'x_diff'] = np.nan
badlatid = all_data[(all_data.x_diff>1.5)].ship.unique()

all_data = all_data[~((all_data.ship==20055)&(all_data.x>20))].reset_index(drop=True) ##整片处理
all_data = all_data[~((all_data.ship==20777)&(all_data.x<20))].reset_index(drop=True) ##整片处理
all_data = all_data[~((all_data.ship==21882)&(all_data.x<20))].reset_index(drop=True) ##整片处理
all_data = all_data[~((all_data.ship==22185)&(all_data.x<28))].reset_index(drop=True) ##整片处理
all_data = all_data[~((all_data.ship==23517)&(all_data.x<28))].reset_index(drop=True) ##整片处理
all_data = all_data[~((all_data.ship==26185)&(all_data.x<25))].reset_index(drop=True) ##整片处理

all_data = all_data[~((all_data.ship==24022)&(all_data.x>24))].reset_index(drop=True) ##整片处理 待定


all_data = all_data[~((all_data.ship==20155)&(all_data.x>30))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==20348)&(all_data.x<20))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==20919)&(all_data.x>25))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==21181)&(all_data.x>30))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==21427)&(all_data.x<20))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==21521)&(all_data.x<24))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==21565)&(all_data.x<22))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==21823)&(all_data.x<20))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==21859)&((all_data.x<20)|(all_data.x>32)))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==21984)&(all_data.x>38))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==21990)&(all_data.x<25))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==22400)&((all_data.x<20)|(all_data.x>30)))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==22712)&(all_data.x<20))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==23226)&(all_data.x>20))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==23795)&(all_data.x<20))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==23872)&(all_data.x<20))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==24832)&(all_data.x<20))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==25536)&(all_data.x>28))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==25661)&(all_data.x>28))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==26158)&(all_data.x<20))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==26649)&(all_data.x<20))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==27609)&(all_data.x<20))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==27733)&(all_data.x<24))].reset_index(drop=True) ##单个处理

all_data = all_data[~((all_data.ship==20241)&(all_data.x<22))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==20354)&(all_data.x>24))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==20366)&(all_data.x>22))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==22637)&(all_data.x<25))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==23499)&(all_data.x<28))].reset_index(drop=True) ##单个处理
all_data = all_data[~((all_data.ship==26199)&(all_data.x<31))].reset_index(drop=True) ##单个处理

all_data = all_data[~((all_data.ship==20521)&((all_data.x>30)|(all_data.x<20)))].reset_index(drop=True) ##单个处理
del all_data['x_diff']

all_data['x_diff'] = all_data.x.diff(-1)
all_data.loc[all_data.drop_duplicates('ship',keep='last').index,'x_diff'] = np.nan
all_data['y_diff'] = all_data.y.diff(-1)
all_data.loc[all_data.drop_duplicates('ship',keep='last').index,'y_diff'] = np.nan
all_data['v_diff'] = all_data.v.diff(-1)
all_data.loc[all_data.drop_duplicates('ship',keep='last').index,'v_diff'] = np.nan
all_data['d_diff'] = all_data.d.diff(-1)
all_data.loc[all_data.drop_duplicates('ship',keep='last').index,'d_diff'] = np.nan


all_data['time_diff'] = all_data.time.diff(-1).dt.total_seconds()
all_data.loc[all_data.drop_duplicates('ship',keep='last').index,'time_diff'] = np.nan

all_data['x_t'] = all_data['x_diff']/all_data['time_diff']
all_data['y_t'] = all_data['y_diff']/all_data['time_diff']
all_data['v_t'] = all_data['v_diff']/all_data['time_diff']
all_data['d_t'] = all_data['d_diff']/all_data['time_diff']

all_data['v_d'] = all_data['v'].astype('str')+'_'+all_data['d'].astype('str')

all_data['v_d_change_abs'] = all_data['v'].astype('str')+'_'+(abs(all_data['d_diff']//10)).astype('str')#)+str(x['d_diff']),axis =1)

# def w2v_vector(train,w2c_col):
#     train[w2c_col] = train[w2c_col].astype("str")
#     sentences = []
#     print("----建立训练预料----"+w2c_col)
#     for i in tqdm(np.unique(train.ship)):
#         df = train[train.ship == i]
#         sentences.append(list(df[w2c_col]))
#     with open(data_path+w2c_col+"trans2.txt", "wb") as fp:   #Pickling
#         pickle.dump(sentences, fp)
    
# #     return sentences
# # train = all_data.loc[all_data.type!='未知',:].copy()
# train = all_data.copy()
# w2v_vector(train,'x')
# w2v_vector(train,'y')
# w2v_vector(train,'v')
# w2v_vector(train,'d')
# w2v_vector(train,'v_d')

# from gensim.models import Word2Vec

# def w2v_feature(train,w2c_col,vec_len=100,win_len=5,mode="mean"):
#     model_path = data_path+"w2c_{}_{}_{}trans2.model".format(w2c_col,vec_len,win_len)

#     if os.path.isfile(model_path):
#         model = Word2Vec.load(model_path)
#     else:
#         with open(data_path+w2c_col+"trans2.txt", "rb") as fp:   # Unpickling
#             sentences = pickle.load(fp)
#         print("----建立w2v模型----")

#         model = Word2Vec(sentences,size=vec_len,window=win_len,workers = 1,seed=1)
#     #     now = datetime.now().strftime('%H_%M_%S')
#         model.save(model_path)        
    
#     res = []
#     ship_name = []
#     print("----输出特征----")
#     for name in tqdm(np.unique(train.ship)):
#         df = train[train.ship == name]
#         vec_sum=[]
#         for i in list(df[w2c_col]):
#             try:
#                 vec_sum.append(model.wv[str(i)])
#             except:
#                 pass
#         if len(vec_sum)==0:
#             vec_sum=np.zeros((1,vec_len))
#         else:
#             vec_sum =np.array(vec_sum)        
# #         vec_sum =np.array(vec_sum)
#         if mode=="mean":
#             res2= np.mean(vec_sum,axis=0)
#         elif mode =="std":
#             res2= np.std(vec_sum,axis=0)
#         elif mode =="min":
#             res2= np.min(vec_sum,axis=0)
#         elif mode =="max":
#             res2= np.max(vec_sum,axis=0)    
#         ship_name.append(name)
#         res.append(res2.tolist())
#     res = np.array(res)
        
#     col  =["w2c_"+w2c_col +"_"+mode+"_"+str(i) for i in range(vec_len)]
        
# #     print(col)    
# #     col  =["w2c_"+w2c_col +"_"+str(i) for i in range(vec_len)]
#     w2c = pd.DataFrame(columns=col)
#     w2c["ship"] = ship_name

#     w2c[col]  =res

#     w2c.to_csv(data_path+"w2c_{}_{}_{}_{}trans2.csv".format(w2c_col,vec_len,win_len,mode),index=None)

from gensim.models import Word2Vec

def w2v_feature(train,w2c_col,vec_len=100,win_len=5,mode="mean"):
    model_path = data_path+"w2c_{}_{}_{}trans2.model".format(w2c_col,vec_len,win_len)
    train_ = train.copy()
    train_[w2c_col] = train_[w2c_col].astype('str')
#     if os.path.isfile(model_path):
#         model = Word2Vec.load(model_path)
#     else:
#         with open(data_path+w2c_col+"trans2.txt", "rb") as fp:   # Unpickling
#             sentences = pickle.load(fp)
#         print("----建立w2v模型----")

#         model = Word2Vec(sentences,size=vec_len,window=win_len,workers = 1,seed=1)
#     #     now = datetime.now().strftime('%H_%M_%S')
#         model.save(model_path)

#     with open(data_path+w2c_col+"trans2.txt", "rb") as fp:   # Unpickling
    sentences = train_.groupby("ship")[w2c_col].apply(lambda x: x.tolist()).tolist()
    print("----建立w2v模型----")

    model = Word2Vec(sentences,size=vec_len,window=win_len,workers = 1,seed=1,sg=1)
#     now = datetime.now().strftime('%H_%M_%S')
    model.save(model_path)  
    
    res = []
    ship_name = []
    print("----输出特征----")
    for name in tqdm(np.unique(train_.ship)):
        df = train_[train_.ship == name]
        vec_sum=[]
        for i in list(df[w2c_col]):
            try:
                vec_sum.append(model.wv[str(i)])
            except:
                pass
        if len(vec_sum)==0:
            vec_sum=np.zeros((1,vec_len))
        else:
            vec_sum =np.array(vec_sum)        
#         vec_sum =np.array(vec_sum)
        if mode=="mean":
            res2= np.mean(vec_sum,axis=0)
        elif mode =="std":
            res2= np.std(vec_sum,axis=0)
        elif mode =="min":
            res2= np.min(vec_sum,axis=0)
        elif mode =="max":
            res2= np.max(vec_sum,axis=0)    
        ship_name.append(name)
        res.append(res2.tolist())
    res = np.array(res)
        
    col  =["w2c_"+w2c_col +"_"+mode+"_"+str(i) for i in range(vec_len)]
        
#     print(col)    
#     col  =["w2c_"+w2c_col +"_"+str(i) for i in range(vec_len)]
    w2c = pd.DataFrame(columns=col)
    w2c["ship"] = ship_name

    w2c[col]  =res

    w2c.to_csv(data_path+"w2c_{}_{}_{}_{}trans2.csv".format(w2c_col,vec_len,win_len,mode),index=None)
    
w2v_feature(all_data,'x',vec_len=64,win_len=8,mode="mean")
w2v_feature(all_data,'y',vec_len=64,win_len=8,mode="mean")

w2v_feature(all_data,'d',vec_len=32,win_len=8,mode="mean")
# w2v_feature(all_data,'xy_10',vec_len=32,win_len=5,mode="mean")
w2v_feature(all_data,'v_d',vec_len=32,win_len=8,mode="mean")
# w2v_feature(all_data,'d_diff',vec_len=32,win_len=5,mode="std")
# w2v_feature(all_data,'d_diff',vec_len=32,win_len=5,mode="mean")
# w2v_feature(all_data,'v_d_change_abs',vec_len=32,win_len=5,mode="mean")
# w2v_feature(all_data,'x_y_v',vec_len=100,win_len=5,mode="mean")
w2v_feature(all_data,'v',vec_len=16,win_len=8,mode="std")

w2v_feature(all_data,'v',vec_len=16,win_len=8,mode="mean")


def group_feature(df, key, target, aggs):   
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t
def extract_feature(df, train):
#     t = group_feature(df, 'ship','d_diff',['max','mean','std','skew'])
#     train = pd.merge(train, t, on='ship', how='left')
#     t = group_feature(df, 'ship','d_diff_150',['sum'])
#     train = pd.merge(train, t, on='ship', how='left')
    
#     t = group_feature(df, 'ship','pca_xy',['max','min','mean','std','skew'])#,'sum'
#     train = pd.merge(train, t, on='ship', how='left')
#     t = group_feature(df, 'ship','pca_xy',['count'])
    t = group_feature(df, 'ship','x',['max','min','mean','std','skew'])#,'sum'  ,'mean','std','skew'
    train = pd.merge(train, t, on='ship', how='left')
    
    t = group_feature(df, 'ship','x',['count'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','y',['max','min','mean','std','skew'])#,'sum' ,
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','v',['max','min','mean','std','skew'])#,'sum'
    train = pd.merge(train, t, on='ship', how='left')
    
    
    t = group_feature(df, 'ship','x_t',['max','min','mean','std','skew'])#,'sum'  ,'mean','std','skew'
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','y_t',['max','min','mean','std','skew'])#,'sum'  ,'mean','std','skew'
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','v_t',['max','min','mean','std','skew'])#,'sum'  ,'mean','std','skew'
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','d_t',['max','min','mean','std','skew'])#,'sum'  ,'mean','std','skew'
    train = pd.merge(train, t, on='ship', how='left')
    
    t = group_feature(df, 'ship','x_diff',['max','min','mean','std','skew'])#,'sum'  ,'mean','std','skew'
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','y_diff',['max','min','mean','std','skew'])#,'sum'  ,'mean','std','skew'
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','v_diff',['max','min','mean','std','skew'])#,'sum'  ,'mean','std','skew'
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','d_diff',['max','min','mean','std','skew'])#,'sum'  ,'mean','std','skew'
    train = pd.merge(train, t, on='ship', how='left')    
    t = group_feature(df, 'ship','time_diff',['max','min','mean','std','skew'])#,'sum'  ,'mean','std','skew'
    train = pd.merge(train, t, on='ship', how='left')       
    
    
#     t = group_feature(df, 'ship','d',['max','min','mean','std','skew'])#,'sum
#     train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','d_sin',['max','min','mean','std','skew'])#,'sum  'max','min','mean',
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','d_cos',['max','min','mean','std','skew'])#,'sum  'max','min','mean',,'skew'
    train = pd.merge(train, t, on='ship', how='left')
    
    t = group_feature(df, 'ship','d',['mean'])#,'sum  'max','min','mean',,'skew'
    train = pd.merge(train, t, on='ship', how='left')

    train['x_max_x_min'] = train['x_max'] - train['x_min']
    train['y_max_y_min'] = train['y_max'] - train['y_min']
    train['y_max_x_min'] = train['y_max'] - train['x_min']
    train['x_max_y_min'] = train['x_max'] - train['y_min']
    train['x_min_y_min'] = train['x_min'] - train['y_min']
    train['x_max_y_max'] = train['x_max'] - train['y_max']

#     train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min']==0, 0.001, train['x_max_x_min'])
#     train['area'] = train['x_max_x_min'] * train['y_max_y_min']
    
    mode_hour = df.groupby('ship')['hour'].agg(lambda x:x.value_counts().index[0]).to_dict()
    train['mode_hour'] = train['ship'].map(mode_hour)
    
    t = group_feature(df, 'ship','hour',['max','min'])
    train = pd.merge(train, t, on='ship', how='left')
    
    hour_nunique = df.groupby('ship')['hour'].nunique().to_dict()
    date_nunique = df.groupby('ship')['date'].nunique().to_dict()
    train['hour_nunique'] = train['ship'].map(hour_nunique)
    train['date_nunique'] = train['ship'].map(date_nunique)

    t = df.groupby('ship')['time'].agg({'diff_time':lambda x:np.max(x)-np.min(x)}).reset_index()
    t['diff_day'] = t['diff_time'].dt.days
    t['diff_second'] = t['diff_time'].dt.seconds
    train = pd.merge(train, t, on='ship', how='left')
    return train

all_label = all_data.drop_duplicates('ship')
all_label = all_label.merge(all_data.drop_duplicates('ship',keep='last')[['ship','x','y']],on='ship',how='left')


type_map = {'拖网':0,'围网':1,'刺网':2,'未知':3}
type_map_rev = {0: '拖网', 1: '围网', 2: '刺网'}
# type_map_rev = {v:k for k,v in type_map.items()}
all_label['type'] = all_label['type'].map(type_map)

all_label = extract_feature(all_data, all_label)

temp = all_data.groupby('ship')['x'].quantile(0.25).reset_index().rename(columns={'x':'x_quant0.25'})
all_label = pd.merge(all_label, temp, on='ship', how='left')
temp = all_data.groupby('ship')['x'].quantile(0.75).reset_index().rename(columns={'x':'x_quant0.75'})
all_label = pd.merge(all_label, temp, on='ship', how='left')
temp = all_data.groupby('ship')['y'].quantile(0.25).reset_index().rename(columns={'y':'y_quant0.25'})
all_label = pd.merge(all_label, temp, on='ship', how='left')
temp = all_data.groupby('ship')['y'].quantile(0.75).reset_index().rename(columns={'y':'y_quant0.75'})
all_label = pd.merge(all_label, temp, on='ship', how='left')
temp = all_data.groupby('ship')['v'].quantile(0.25).reset_index().rename(columns={'v':'v_quant0.25'})
all_label = pd.merge(all_label, temp, on='ship', how='left')
temp = all_data.groupby('ship')['v'].quantile(0.75).reset_index().rename(columns={'v':'v_quant0.75'})
all_label = pd.merge(all_label, temp, on='ship', how='left')


w2c_v = pd.read_csv(data_path+'w2c_v_16_8_meantrans2.csv')#train_speaical_v_w2c_v_32_5_all
print(w2c_v.head(3))
add_fea1 = [x for x in w2c_v.columns if x not in ['ship']]
all_label = all_label.merge(w2c_v,on = ['ship'],how = 'left')

w2c_d = pd.read_csv(data_path+'w2c_d_32_8_meantrans2.csv')#train_
add_fea4 = [x for x in w2c_d.columns if x not in ['ship']]
all_label = all_label.merge(w2c_d,on = ['ship'],how = 'left')

# w2c_trans = pd.read_csv(data_path+'trans_feature1.csv')#train_
# add_featrans1 = [x for x in w2c_trans.columns if x not in ['ship']]
# all_label = all_label.merge(w2c_trans,on = ['ship'],how = 'left')

# w2c_trans = pd.read_csv(data_path+'trans_feature2.csv')#train_
# add_featrans2 = [x for x in w2c_trans.columns if x not in ['ship']]
# all_label = all_label.merge(w2c_trans,on = ['ship'],how = 'left')

w2c_vd = pd.read_csv(data_path+'w2c_v_d_32_8_meantrans2.csv')#train_

add_fea5 = [x for x in w2c_vd.columns if x not in ['ship']]
all_label = all_label.merge(w2c_vd,on = ['ship'],how = 'left')

w2c_x = pd.read_csv(data_path+'w2c_x_64_8_meantrans2.csv')#train_
add_fea7= [x for x in w2c_x.columns if x not in ['ship']]
all_label = all_label.merge(w2c_x,on = ['ship'],how = 'left')

w2c_y = pd.read_csv(data_path+'w2c_y_64_8_meantrans2.csv')#train_
add_fea8 = [x for x in w2c_y.columns if x not in ['ship']]
all_label = all_label.merge(w2c_y,on = ['ship'],how = 'left')

w2c_v_std = pd.read_csv(data_path+'w2c_v_16_8_stdtrans2.csv')#train_speaical_v_w2c_v_32_5_all
add_fea_std = [x for x in w2c_v_std.columns if x not in ['ship']]
all_label = all_label.merge(w2c_v_std,on = ['ship'],how = 'left')

param = {'num_leaves': 32,#20
#          'min_data_in_leaf': 10, 
         'objective':'multiclass',
         'max_depth': -1,
         'learning_rate': 0.1,#0.1,#
#          "min_child_samples": 15,
         "boosting": "gbdt",
#          "feature_fraction": 0.8,#0.9
#          "bagging_freq": 1,
#          "bagging_fraction": 0.8,#0.9
#          "bagging_seed": 1,
         "num_class": 3,
         "lambda_l1": 0.1,
         "metric":"None",
         "first_metric_only":True
         }

xv_cov = []
a_xy = []
fea_loop = pd.DataFrame(columns = ['ship','xy_cov','xy_a'])
fea_loop['ship'] = all_label.ship.values
for i in all_label.ship.values:
    x_ = df_all[df_all.ship==i].copy()
    xv_cov.append(x_['x'].cov(x_['y']))
#     fea_loop.loc[i,'xy_cov'] = x_['x'].cov(x_['y'])
    t_diff=x_['time'].diff().iloc[1:].dt.total_seconds()
    x_diff=x_['x'].diff().iloc[1:].abs()
    y_diff=x_['y'].diff().iloc[1:].abs()
    x_a_mean=(x_diff/t_diff).mean()
    y_a_mean=(y_diff/t_diff).mean()
#     fea_loop.loc[i,'xy_a'] = np.sqrt(x_a_mean**2+y_a_mean**2)
    a_xy.append(np.sqrt(x_a_mean**2+y_a_mean**2))
del x_
# all_label = all_label.merge(fea_loop,on = ['ship'],how = 'left')
fea_loop['xy_cov'] = xv_cov
fea_loop['xy_a'] = a_xy
all_label = all_label.merge(fea_loop,on = ['ship'],how = 'left')


# 1405_932
feature_name_ = ['x_x', 'y_x', 'weekday', 'x_diff', 'y_diff', 'v_diff', 'd_diff', 'time_diff', 'y_t', 'v_t',\
 'd_t', 'x_max', 'x_min', 'x_mean', 'x_std', 'x_skew', 'x_count', 'y_max', 'y_min', 'y_mean',\
 'y_std', 'y_skew', 'v_max', 'v_min', 'v_std', 'v_skew', 'x_t_max', 'x_t_min', 'x_t_mean',\
 'x_t_std', 'x_t_skew', 'y_t_max', 'y_t_min', 'y_t_mean', 'y_t_std', 'y_t_skew', 'v_t_max',\
 'v_t_min', 'v_t_mean', 'v_t_std', 'v_t_skew', 'd_t_max', 'd_t_min', 'd_t_mean', 'd_t_std',\
 'd_t_skew', 'x_diff_max', 'x_diff_min', 'x_diff_mean', 'x_diff_std', 'x_diff_skew',\
 'y_diff_max', 'y_diff_min', 'y_diff_mean', 'y_diff_std', 'y_diff_skew', 'v_diff_max',\
 'v_diff_mean', 'v_diff_std', 'd_diff_max', 'd_diff_min', 'd_diff_mean', 'd_diff_std',\
 'd_diff_skew', 'time_diff_max', 'time_diff_min', 'time_diff_mean',\
 'time_diff_std', 'time_diff_skew', 'd_sin_max', 'd_sin_min', 'd_sin_mean',\
 'd_sin_std', 'd_sin_skew', 'd_cos_max', 'd_cos_min', 'd_cos_mean', 'd_cos_std', 'd_cos_skew',\
 'd_mean', 'x_max_x_min', 'y_max_y_min', 'y_max_x_min', 'x_max_y_min', 'x_min_y_min',\
 'x_max_y_max', 'mode_hour', 'hour_max', 'hour_min', 'hour_nunique', 'date_nunique','xy_cov', 'xy_a',\
 'diff_day', 'diff_second', 'x_quant0.25', 'x_quant0.75', 'y_quant0.25', 'y_quant0.75', 'v_quant0.25', 'v_quant0.75']+add_fea1+add_fea4+add_fea5+add_fea7+add_fea8+add_fea_std

# feature_name_ = add_featrans+['x_mean', 'x_min','x_std', 'x_skew', 'x_count', 'y_max', 'y_min', 'y_mean', 'y_std', 'y_skew', 'v_mean', 'v_std', 'v_skew', 'd_sin_std', 'd_cos_std', 'x_max_x_min', 'y_max_y_min', 'y_max_x_min', 'x_max_y_min', 'x_min_y_min', 'x_max_y_max', 'mode_hour', 'date_nunique', 'diff_day', 'x_quant0.25', 'x_quant0.75', 'y_quant0.25', 'y_quant0.75', 'v_quant0.25', 'v_quant0.75',\
#                  'xy_cov', 'xy_a', 'd_mean','weekday', 'x_y', 'y_y', 'x_max']+add_fea1+add_fea4+add_fea5+add_fea7+add_fea8#+add_fea_x_y_v+add_fea_v_d_change_abs_mean+add_fea_d_diff_std
# feature_name_ = ['v_mean', 'v_std', 'v_skew', 'd_sin_std', 'd_cos_std', 'mode_hour', 'diff_day', 'v_quant0.25', 'v_quant0.75',\
#                  'd_mean','weekday', ]+add_fea1+add_fea4
#, 'xy_cov', 'xy_a'
X_train1 = all_label[all_label.type!=3][feature_name_].values[7000:15166]
X_train2 = all_label[all_label.type!=3][feature_name_].values[:7000]

y_train1 = all_label[all_label.type!=3]['type'].values[7000:15166]
y_train2 = all_label[all_label.type!=3]['type'].values[:7000]

X_test = all_label[all_label.type==3][feature_name_].values


from sklearn import metrics
fold_num = 5

from sklearn.preprocessing import label_binarize
def Macro_f1(preds,train_data):
    labels = label_binarize(train_data.get_label(),np.arange(3))
    preds = label_binarize(np.argmax(preds.reshape(3,-1),axis = 0),np.arange(3))  
#     label_binarize(np.argmax(preds.reshape(4,-1),axis = 0),np.arange(4))
#     np.argmax(preds.reshape(4,-1),axis = 0)
    macro_f1 = metrics.f1_score(labels,preds,average='macro')
    return 'macro_f1',macro_f1,True

folds = StratifiedKFold(n_splits=fold_num,random_state=1024,shuffle=True)
oof_lgb = np.zeros((len(X_train1),3))
predictions_lgb = np.zeros((len(X_test),3))
predictions_lgb2 = np.zeros((len(X_test),fold_num))
# predictions_lgb = np.zeros((len(df_test),4))
lgb_importance = []



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train1, y_train1)):

    print("fold n°{}".format(fold_+1))
    print(X_train1[trn_idx].shape,type(X_train1))
    x_tr = np.vstack((X_train1[trn_idx],X_train2))
    y_tr = np.vstack((y_train1[trn_idx].reshape(-1,1),y_train2.reshape(-1,1))).ravel()
    x_va = X_train1[val_idx]
    y_va = y_train1[val_idx]   

    trn_data = lgb.Dataset(x_tr, y_tr)

    val_data = lgb.Dataset(x_va, y_va)

    clf = lgb.train(param, trn_data, 2000, verbose_eval=200
    #                 ,feature_name = feature_name_
#                         ,fobj=focal_loss, feval=eval_error
                        ,early_stopping_rounds = 200
                        ,valid_sets = val_data
                        ,feval = Macro_f1
#                     ,callbacks = [lgb.early_stopping(100,True)]

                   )# early_stopping_rounds = 100
    # oof_lgb[val_idx,:] = clf.predict(x_tr, num_iteration=clf.best_iteration)#.argmax(axis = 1)
    oof_lgb[val_idx,:] = clf.predict(x_va, num_iteration=clf.best_iteration) #
    lgb_importance.append(clf.feature_importance())
    predictions_lgb += clf.predict(X_test)/folds.n_splits
    predictions_lgb2[:,fold_] = clf.predict(X_test).argmax(axis = 1)
    gc.collect()
    
    
from sklearn.metrics import accuracy_score
from scipy import stats

print('raw_data accuracy_score:',accuracy_score(y_train1, oof_lgb.argmax(axis = 1)))

from sklearn.metrics import f1_score
print('val f1',f1_score(y_train1, oof_lgb.argmax(axis = 1), average='macro'))
# from draw import fea_lgb_impt
# fea_lgb_impt(feature_name_,lgb_importance)
named_scores = zip(feature_name_,lgb_importance[0])

print(sorted(named_scores, key=lambda z: z[1], reverse=True))
print(metrics.classification_report(y_train1, oof_lgb.argmax(axis = 1)))



sub = all_label[all_label.type==3][['ship']]
# sub['pred'] = stats.mode(predictions_lgb2,axis = 1)[0]

sub['pred'] = predictions_lgb.argmax(axis = 1)
# pd.DataFrame(oof_lgb).to_csv()
# sub1 = 
type_map = dict(zip(df_train['type'].unique(), np.arange(3)))
type_map_rev = {v:k for k,v in type_map.items()}
# print(sub['pred'].value_counts(1))
# print(sub['pred'].value_counts())
y_ci = y_train1.copy()
y_ci[np.where(y_ci!=2)]=0
y_ci[np.where(y_ci==2)]=1

y_ci_pred = oof_lgb.argmax(axis = 1).copy()
y_ci_pred[np.where(y_ci_pred!=2)]=0
y_ci_pred[np.where(y_ci_pred==2)]=1

# print('val f1 ci',f1_score(y_ci, y_ci_pred))

sub['pred'] = sub['pred'].map(type_map_rev)

sub.to_csv(data_path+'result.csv', index=None, header=None)

# sub = all_label[all_label.type==3][['ship']]

# sub = all_label[all_label.type==3][['ship']]
# sub['pred0'] = predictions_lgb[:,0]
# sub['pred1'] = predictions_lgb[:,1]
# sub['pred2'] = predictions_lgb[:,2]

# sub.to_csv(data_path+'predict_pro.csv',index = False)

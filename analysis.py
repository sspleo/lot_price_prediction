# -*- coding: utf-8 -*-

import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

"""
reader = csv.reader(open('./lot_polygon.csv'), delimiter =',')

lot_polygon = {}
cols = next(reader)
k1 = cols[0]
k2 = cols[1]

for row in reader:
    lot_polygon[row[2]] = {k1:row[0], k2:row[1]}


reader = csv.reader(open('./lot_public_price.csv'), delimiter = ',')

lot_public_price = {}
cols = next(reader)

for i in range(len(cols)-1):
    globals()[f'k{i+1}'] = cols[i]

for row in reader:
    lot_public_price[row[-1]] = {}
    for i in range(len(cols)-1):
        lot_public_price[row[-1]][globals()[f'k{i+1}']] = row[i]
"""

# Convert csv files to dictionaries
def csv_to_dict(name):
    res_dict = {}
    reader = csv.reader(open(f'./{name}.csv'), delimiter = ',')
    cols = next(reader)
    
    for i in range(len(cols)-1):
        globals()[f'k{i}'] = cols[i]
    
    for row in reader:
        res_dict[row[-1]] = {}
        for i in range(len(cols)-1):
            res_dict[row[-1]][globals()[f'k{i}']] = row[i]
    
    return res_dict, cols

lot_polygon, polygon_cols = csv_to_dict('lot_polygon')
lot_public_price, public_price_cols = csv_to_dict('lot_public_price')



fw = open('./deal_lot_refined.csv', 'w')
reader = csv.reader(open('./deal_land.csv'), delimiter =',')

for col in next(reader)[:2]:
    _= fw.write(col+',')

_= fw.write('contract_yr,contract_mo,contract_day,')

for col in public_price_cols[:-2]:
    _= fw.write(col+',')

_= fw.write('uploaded_yr,uploaded_mo,uploaded_day,price_per_area\n')

err_key = []

road_type_to_cat = {'-':0, '8m미만':1, '12m미만':2, '25m미만':3, '25m이상':4}

for row in reader:
    road_type = road_type_to_cat[row[0]]
    lot_ar = row[1]
    contract_yr = row[2][:4]
    contract_mo = row[2][-2:]
    contract_day = row[3]
    price = row[4]
    price_per_area = round(float(price)/float(lot_ar),1)
    key = row[5]
    
    if key in lot_public_price:
        _ = fw.write(f'{road_type},{lot_ar},{contract_yr},{contract_mo},{contract_day},')
        
        for col in public_price_cols[:-2]:
            val = lot_public_price[key][col]
            _= fw.write(f'{val},')
        
        uploaded_dt = lot_public_price[key]['uploaded_dt'].split('-')
        uploaded_yr = uploaded_dt[0]
        uploaded_mo = uploaded_dt[1]
        uploaded_day = uploaded_dt[2]
        
        _= fw.write(f'{uploaded_yr},{uploaded_mo},{uploaded_day},{price_per_area}\n')
    else:
        err_key.append(key)

fw.close()



#- err key check
cnt_pol = 0
for key in err_key:
    if key in lot_polygon:
        cnt_pol += 1
    else:
        pass


cnt_pub = 0
for key in err_key:
    if key in lot_public_price:
        cnt_pub += 1
    else:
        pass

print(cnt_pol, cnt_pub)



#load deal_lot.csv
lot_df = pd.read_csv('./deal_lot.csv')
print(lot_df.head(5))


#correlation plot
f = plt.figure(figsize=(20, 14))
plt.matshow(lot_df.corr(method='pearson'), fignum=f.number)
plt.matshow(lot_df.corr(), fignum=f.number)
plt.xticks(range(lot_df.select_dtypes(['number']).shape[1]), lot_df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(lot_df.select_dtypes(['number']).shape[1]), lot_df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Data Correlation Matrix\n\n', fontsize=24);
plt.savefig('./data_correlation_matrix.png')
plt.show()


#'area' and 'lot_ar' has high correlation of '0.9769'. Hence we might omit 'area'
lot_df.drop('area',axis='columns', inplace=True)

#We do not need uploaded_mo and uploaded_day since they are all '01'
lot_df.drop('uploaded_mo',axis='columns', inplace=True)
lot_df.drop('uploaded_day',axis='columns', inplace=True)
lot_df.dropna()

lot_df.head(5)


#load deal_lot_refined.csv : lot_df - ['latitude', 'longitude']
lot_df_refined = pd.read_csv('./deal_lot_refined.csv')
print(lot_df_refined.head(5))
lot_df_refined.drop('bun', axis = 'columns', inplace=True)
lot_df_refined.drop('ji', axis='columns', inplace=True)
lot_df_refined.drop('area',axis='columns', inplace=True)
lot_df_refined.drop('uploaded_mo',axis='columns', inplace=True)
lot_df_refined.drop('uploaded_day',axis='columns', inplace=True)

cols_df_refined = list(lot_df_refined.columns)[:-1]



lot_df_refined.to_csv('lot_df_processed.csv', index=False)

lot_df_processed = lot_df_refined

cols_df_processed = list(lot_df_processed.columns)[:-1]


#We omit outliers
def get_outliers(dataset, outliers_fraction=0.25):
    clf = svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05, kernel="rbf", gamma=0.1)
    clf.fit(dataset)
    result = clf.predict(dataset)
    return result


training_dataset = lot_df[get_outliers(lot_df, 0.05)==1]
training_dataset

training_dataset = lot_df_processed[get_outliers(lot_df_proceessed, 0.05)==1]
training_dataset


#distplot
sns.set(rc={"figure.figsize": (8, 4)})

plt.subplot(1,2,1)
ax1 = sns.distplot(lot_df_processed['price_per_area'])
plt.subplot(1,2,2)
ax2 = sns.distplot(training_dataset['price_per_area'])

plt.show()


scaler = MinMaxScaler()
print(scaler.fit(training_dataset))
print(scaler.data_max_)
print(scaler.data_min_)

from sklearn.externals import joblib
joblib.dump(scaler, 'MinMaxScaler.save')
scaler = joblib.load('MinMaxScaler.save')



#Check year data fit together.
scaler.data_max_[2] == scaler.data_max_[15]
scaler.data_min_[2] == scaler.data_min_[15]


### Model
X_train, X_val, Y_train, Y_val = train_test_split(training_dataset[cols], training_dataset['price_per_area'], test_size=0.2)
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
Y_train.to_csv('Y_train.csv', index=False)
Y_val.to_csv('Y_val.csv', index=False)

# model = RandomForestRegressor(n_estimators=150, max_features='sqrt', n_jobs=-1)

X_train = pd.read_csv('X_train.csv')
Y_train = pd.read_csv('Y_train.csv')
X_val = pd.read_csv('X_val.csv')
Y_val = pd.read_csv('Y_val.csv')


models = [LinearRegression(),
        RandomForestRegressor(n_estimators=100, max_features='sqrt'),
        KNeighborsRegressor(n_neighbors=6),
        SVR(kernel='linear'),
        LogisticRegression()
        ]

TestModels = pd.DataFrame()
tmp = {}

for model in models:
    # get model name
    m = str(model)
    tmp['Model'] = m[:m.index('(')]
    # fit model on training dataset
    model.fit(X_train, Y_train)
    # predict prices for test dataset and calculate r^2
    tmp['R2_Price'] = r2_score(Y_val, model.predict(X_val))
    # write obtained data
    TestModels = TestModels.append([tmp])

TestModels.set_index('Model', inplace=True)

fig, axes = plt.subplots(ncols=1, figsize=(10, 4))
TestModels.R2_Price.plot(ax=axes, kind='bar', title='R2_Price')
plt.show()




#-- Test Data Set
fw = open('./test_deal_lot.csv', 'w')

reader = csv.reader(open('./test.csv'), delimiter =',')

for col in next(reader)[:2]:
    _= fw.write(col+',')

_= fw.write('contract_yr,contract_mo,contract_day,')

for col in polygon_cols[:-1]:
    _= fw.write(col+',')

for col in public_price_cols[:-2]:
    _= fw.write(col+',')

_= fw.write('uploaded_yr,uploaded_mo,uploaded_day\n')

test_err_key = []

road_type_to_cat = {'-':0, '8m미만':1, '12m미만':2, '25m미만':3, '25m이상':4}

for row in reader:
    road_type = road_type_to_cat[row[0]]
    lot_ar = row[1]
    contract_yr = row[2][:4]
    contract_mo = row[2][-2:]
    contract_day = row[3]
    #price = row[4]
    #price_per_area = round(float(price)/float(lot_ar),1)
    key = row[5]
    
    if key in lot_polygon and key in lot_public_price:
        _ = fw.write(f'{road_type},{lot_ar},{contract_yr},{contract_mo},{contract_day},')
        
        for col in polygon_cols[:-1]:
            val = lot_polygon[key][col]
            _= fw.write(f'{val},')
        
        for col in public_price_cols[:-2]:
            val = lot_public_price[key][col]
            _= fw.write(f'{val},')
        
        uploaded_dt = lot_public_price[key]['uploaded_dt'].split('-')
        uploaded_yr = uploaded_dt[0]
        uploaded_mo = uploaded_dt[1]
        uploaded_day = uploaded_dt[2]
        
        _= fw.write(f'{uploaded_yr},{uploaded_mo},{uploaded_day}\n')
    else:
        test_err_key.append(key)

fw.close()


test_df = pd.read_csv('./test_deal_lot.csv')
test_df.drop('area',axis='columns', inplace=True)
test_df.drop('uploaded_mo',axis='columns', inplace=True)
test_df.drop('uploaded_day',axis='columns', inplace=True)
test_df.dropna()



#- err key check
cnt = 0
for key in test_err_key:
    if key in lot_polygon:
        cnt += 1
    else:
        pass

print(cnt)

cnt = 0
for key in test_err_key:
    if key in lot_public_price:
        cnt += 1
    else:
        pass


print(cnt)
len(test_err_key)
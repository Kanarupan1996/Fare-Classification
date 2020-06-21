from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
%matplotlib inline

ride_df = pd.read_csv("../input/mlproject/train.csv", index_col="tripid") 
rideTest_df = pd.read_csv("../input/mlproject/test.csv", index_col="tripid")
ride_df.head(25)
#rideTest_df.head(25)
#ride_df.describe()

ride_df.dtypes

null_columns = ride_df.columns[ride_df.isnull().any()]
null_columns
null_data = ride_df[null_columns]
null_data[null_data.isnull().any(axis=1)][180:202]
ride_df[null_columns].isnull().sum()

ride_df = ride_df.dropna(how='any',axis=0)

ride_df[null_columns].isnull().sum()

rideTest_df

rideTest_df[rideTest_df.isnull().any(axis=1)].isnull().sum()

ride_df['pickup_hour'] = pd.DatetimeIndex(ride_df['pickup_time']).hour
ride_df['pickup_day'] = pd.DatetimeIndex(ride_df['pickup_time']).weekday
ride_df['drop_hour'] = pd.DatetimeIndex(ride_df['drop_time']).hour
ride_df['drop_day'] = pd.DatetimeIndex(ride_df['drop_time']).weekday
ride_df[0:50]

rideTest_df['pickup_hour'] = pd.DatetimeIndex(rideTest_df['pickup_time']).hour
rideTest_df['pickup_day'] = pd.DatetimeIndex(rideTest_df['pickup_time']).weekday
rideTest_df['drop_hour'] = pd.DatetimeIndex(rideTest_df['drop_time']).hour
rideTest_df['drop_day'] = pd.DatetimeIndex(rideTest_df['drop_time']).weekday
rideTest_df[0:50]

def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    #Define earth radius (km)
    R_earth = 6371
    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    #Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    
    return 2 * R_earth * np.arcsin(np.sqrt(a))
  
ride_df['travel_distance'] = sphere_dist(ride_df['pick_lat'], ride_df['pick_lon'], ride_df['drop_lat'], ride_df['drop_lon'])
rideTest_df['travel_distance'] = sphere_dist(rideTest_df['pick_lat'], rideTest_df['pick_lon'], rideTest_df['drop_lat'], rideTest_df['drop_lon'])
  
ride_df['fare_rate_per_distance'] = (ride_df['fare'] - ride_df['meter_waiting_fare'] - ride_df['additional_fare']) / ride_df['travel_distance']
rideTest_df['fare_rate_per_distance'] = (rideTest_df['fare'] - rideTest_df['meter_waiting_fare'] - rideTest_df['additional_fare']) / rideTest_df['travel_distance']
  
ride_df.label[ride_df.label == 'correct'] = 1
ride_df.label[ride_df.label == 'incorrect'] = 0
ride_df[0:50]
ride_df[50:100]

ride_df.columns

X = ride_df[['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
       'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat',
       'drop_lon', 'fare', 'pickup_hour', 'drop_hour', 'pickup_day', 'drop_day', 'travel_distance','fare_rate_per_distance']]
       
Y = ride_df[['label']].astype(int)
Y[Y.isnull().any(axis=1)]
Y[0:50]

X_train, X_eval, Y_train, Y_eval = train_test_split(
    X,
    Y,
    test_size=0.33,
    shuffle=True,
    stratify=Y,
    random_state=42
)

%%time
model = XGBClassifier(
learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 seed=27)
model.fit(X_train,Y_train)

model.score(X_eval, Y_eval)

%%time
model = XGBClassifier(
learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 seed=27)
model.fit(X,Y)

rideTest_df.columns

rideTest_df = rideTest_df[['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
       'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat',
       'drop_lon', 'fare', 'pickup_hour', 'drop_hour', 'pickup_day', 'drop_day', 'travel_distance', 'fare_rate_per_distance']]
       
test_pred=model.predict_proba(rideTest_df)
test_pred
model.classes_

test_pred

pred = model.predict(rideTest_df)
pred

test_pred[0:50]
pred[0:50]

submission_df = pd.read_csv("../input/mlproject/sample_submission.csv", index_col="tripid")
submission_df

np.testing.assert_array_equal(rideTest_df.index.values, 
                              submission_df.index.values)

# Save predictions to submission data frame
submission_df["prediction"] = pred
submission_df[0:50]

submission_df.to_csv('my_submission.csv', index=True)



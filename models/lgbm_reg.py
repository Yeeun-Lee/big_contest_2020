import lightgbm as lgb
import numpy as np
import pandas as pd
from prep.utils import train_data, test_data

def MAPE(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_lgbm():
    params = {'learning_rate': 0.05,
              'max_depth': 30,
              'boosting': 'gbdt',
              'objective': 'regression',
              'metric': 'mape',
              'is_training_metric': True,
              'num_leaves': 120,
              'feature_fraction': 0.9,
              'bagging_fraction': 0.7,
              'bagging_freq': 5,
              'seed': 123457}
    # get filtered data from prep.utils.train_data
    x_train, x_valid, y_train, y_valid = train_data()
    x_train = x_train.drop(['마더코드', '상품코드'], axis = 1)
    x_valid = x_valid.drop(['마더코드', '상품코드'], axis = 1)

    to_cat = ['상품군', 'week', 'hour', 'weekofyear', 'stage']

    for col in to_cat:
        x_train[col] = x_train[col].astype('category')
        x_valid[col] = x_valid[col].astype('category')

    y_log_train = np.log1p(y_train)
    y_log_valid = np.log1p(y_valid)

    stat_train_ds = lgb.Dataset(x_train, label=y_log_train)
    stat_valid_ds = lgb.Dataset(x_valid, label=y_log_valid)

    # 훈련
    model = lgb.train(params, stat_train_ds, 100000, stat_valid_ds,
                      verbose_eval=100, early_stopping_rounds=200)

    predict_train = model.predict(x_train)
    predict_test = model.predict(x_valid)

    train_mape = MAPE(y_log_train, predict_train)
    test_mape = MAPE(y_log_valid, predict_test)

    print("*** MAPE on Log tranformed labels ***")
    print('- train data MAPE: ', train_mape)
    print('-  test data MAPE: ', test_mape)
    print("=====================================")
    print("** MAPE on exp(origin) Labels ** : ", MAPE(y_valid, np.expm1(predict_test)))

    return model

def pred(train, model):
    test = test_data(train_set = train)
    test = test.drop(['마더코드', '상품코드'], axis = 1)

    to_cat = ['상품군', 'week', 'hour', 'weekofyear', 'stage']

    for col in to_cat:
        test[col] = test[col].astype('category')
    print(test.shape)
    predict_test = model.predict(test)
    return predict_test

def save_file(pred):
    temp = pd.read_excel("prep/data/02_평가데이터/2020 빅콘테스트 데이터분석분야-챔피언리그_2020년 6월 판매실적예측데이터(평가데이터).xlsx",
                         sheet_name='6월편성', skiprows=1)
    temp.loc[temp['노출(분)'].isnull(), '노출(분)'] = \
        temp.loc[temp['노출(분)'].isnull(), '방송일시'].map(temp.loc[temp['노출(분)'].notnull()] \
                                                 .set_index('방송일시')['노출(분)'])
    print(len(temp))
    temp = temp[temp.판매단가.isna() == False][temp.상품군!='무형']
    temp['취급액'] = np.expm1(pred) # original values
    temp.to_csv("prediction.csv", index = False)
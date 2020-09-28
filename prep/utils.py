import numpy as np
import pandas as pd
import time
from tqdm.auto import tqdm
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from . import performance


"""
** 서강패럿 - 김예지, 이예은 **
make validation datasets(Filtering)
1. 기 학습된 데이터셋 내에 test set row의 columns = col_names상품코드가 포함된 경우
    1.1 데이터를 유지하고, 학습데이터에서 판매단가 차가 가장 작은 row의 stat data를 차용한다.
2. (상품코드가 존재하지 않을 때) 마더코드가 존재하는 경우
    2.1 마더코드가 동일한 데이터 row들 중 판매단가 차가 가장 작은 row의 데이터
        (날씨, cpi등의 데이터는 유지)
3. (상품코드, 마더코드가 존재하지 않는 경우)
    3.1 동일 상품군 내 판매단가 차가 가장 작은 row의 데이터
        (날씨, cpi등의 데이터는 유지)
"""
# def DecimalEncoding(df):

def goods_code_encoding(x_train, row, code_set):
    if row['상품코드'] in code_set:
        frame = x_train[x_train['상품코드'] == row['상품코드']]
        frame['diff'] = abs(frame['판매단가'] - row['판매단가'])
        fitrow = frame[frame['diff'] == min(frame['diff'])].iloc[0, :-1]
    else:
        if row['마더코드'] in x_train['마더코드'].unique():
            frame = x_train[x_train['마더코드']==row['마더코드']]
            frame['diff'] =  abs(frame['판매단가'] - row['판매단가'])
            fitrow = frame[frame['diff']==min(frame['diff'])].iloc[0, :-1]

        else:
            frame = x_train[x_train['상품군']==row['상품군']]
            frame['diff'] =  abs(frame['판매단가'] - row['판매단가'])
            fitrow = frame[frame['diff']==min(frame['diff'])].iloc[0, :-1]
    row['avgp'] = fitrow['avgp']
    row['minp'] = fitrow['minp']
    row['maxp'] = fitrow['maxp']
    row['profit/140'] = fitrow['profit/140']
    row['profit-group'] = fitrow['profit-group']

    return row


def train_data(data = None):
    pd.set_option('mode.chained_assignment', None)
    if data == None:
        dataset = performance.load_dataset()
    else:
        dataset = pd.read_csv(os.path.join('prep/data', data))
    dataset = dataset.drop(['방송일시', '상품명', 'date','year', 'month',
                            'time', 'real_date', 'profit/m'], axis = 1)

    _X = dataset.drop(['취급액'], axis = 1)

    print(_X.columns)
    _Y = dataset['취급액']
    x_train, x_valid, y_train, y_valid = train_test_split(_X, _Y,
                                                          test_size=0.1,
                                                          random_state=123457)
    # # filtering logic
    # x_valid = filtering(x_train, x_valid)
    train_goods_set = set(x_train['상품코드'])

    for i in tqdm(range(len(x_valid))):
        row = x_valid.iloc[i]
        x_valid.iloc[i] = goods_code_encoding(x_train, row, train_goods_set)
    print(x_train.shape)
    print(x_valid.shape)
    return x_train, x_valid, y_train, y_valid

def test_data(train_set = None):
    pd.set_option('mode.chained_assignment', None)
    """
    filtering에 사용될 학습한 train data를 같이 넣어줍니다.
    """
    test = performance.load_dataset(test=True)

    test = test.drop(['방송일시', '상품명', '취급액', 'date', 'year', 'month',
                      'time', 'real_date'], axis=1)
    train_set = train_set.drop(['방송일시', '상품명', '취급액', 'date', 'year', 'month',
                                'time', 'real_date', 'profit/m'], axis=1)
    # test_f = filtering(train, test)
    train_goods_set = set(train_set['상품코드'])
    ch_cols = ['avgp', 'maxp', 'minp', 'profit/140', 'profit-group']
    test[ch_cols] = np.nan

    for i in tqdm(range(len(test))):
        row = test.iloc[i]
        test.iloc[i] = goods_code_encoding(train_set, row, train_goods_set)
    print(test.head())
    print(test.columns)
    return test






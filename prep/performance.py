import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from .meta import weather, CPI

"""
** 서강패럿 - 김예지, 이예은 **
실적데이터.xlsx 전처리
1. 요일 처리
2. 실제 date처리(새벽 2시 방송인 경우 전날 방송일로 바꾸어준다, real_date)
3. 휴일 컬럼 추가(IsHoliday, 지속휴일수)
4. 방송 시점 컬럼 추가(stage, 방송 초/중/후반 정보)
"""

def caldate(date, time):
    """
    week : 0 - 월, 1 - 화, 2 - 수 ...
    """
    start = pd.Timestamp('00:00:00').time()
    end = pd.Timestamp('02:00:00').time()
    if time >= start and time <= end:
        date = date - pd.DateOffset(days = 1)
    return date

def data_stat(pf):
    """
    :param df: new performance
    :return: performance df with statistical values
    """
    df = pf.copy()
    df['profit/m'] = df['취급액']/df['노출(분)']
    # print(df.head())
    # mean
    avg = df[['상품코드', 'profit/m']].groupby('상품코드').mean()
    avg.rename(columns={'profit/m': 'avgp'}, inplace=True)
    df = pd.merge(df, avg, on=['상품코드'])

    # max
    maxval = df[['상품코드', 'profit/m']].groupby('상품코드').max()
    maxval.rename(columns={'profit/m': 'maxp'}, inplace=True)
    df = pd.merge(df, maxval, on=['상품코드'])

    # min
    minval = df[['상품코드', 'profit/m']].groupby('상품코드').min()
    minval.rename(columns={'profit/m': 'minp'}, inplace=True)
    df = pd.merge(df, minval, on=['상품코드'])

    # Profit/140
    pf_140 = df[['week', 'hour', 'profit/m']].groupby(['week', 'hour']).mean()
    pf_140.rename(columns={'profit/m': 'profit/140'}, inplace=True)
    df = pd.merge(df, pf_140, on=['week', 'hour'])

    # Profit-group
    pf_g = df[['상품군', 'profit/m']].groupby('상품군').mean()
    pf_g.rename(columns={'profit/m': 'profit-group'}, inplace=True)
    df = pd.merge(df, pf_g, on=['상품군'])

    # *노출(분)
    # df['avgp_mul'] = df['avgp']*df['노출(분)']
    # df['minp_mul'] = df['avgp']*df['노출(분)']
    # df['maxp_mul'] = df['avgp']*df['노출(분)']
    # df['140_mul'] = df['profit/140']*df['노출(분)']
    # df['group_mul'] = df['profit-group']*df['노출(분)']
    df['avgp'] = df['avgp']*df['노출(분)']
    df['minp'] = df['minp']*df['노출(분)']
    df['maxp'] = df['maxp']*df['노출(분)']
    df['profit/140'] = df['profit/140']*df['노출(분)']
    df['profit-group'] = df['profit-group']*df['노출(분)']


    return df

def stage(x):
    if x < 0.34:
        return 'early'
    elif 0.34 <= x < 0.67:
        return 'mid'
    return 'late'

def stage_advanced(x):
    if x['count'] == 1:
        return 'short'
    if x['count'] == 2:
        if x['p'] <= 0.5:
            return 'early'
        else:
            return 'late'
    else:
        return stage(x['p'])

def prime_set(df):
    sp = df[['취급액', 'week', 'hour']].groupby(['week', 'hour']).sum().reset_index()
    sp['취급액'] = sp['취급액'].map(lambda x: x/1000)

    prime = sp[sp['취급액'] > np.percentile(sp['취급액'], 75)]
    prime = list(zip(prime['week'], prime['hour']))
    return prime

def make_prime(week, hour, prime):
    if (week, hour) in prime:
        return 1
    else:
        return 0

def load_dataset(test = False):
    if test == True:
        pf = pd.read_excel("prep/data/02_평가데이터/2020 빅콘테스트 데이터분석분야-챔피언리그_2020년 6월 판매실적예측데이터(평가데이터).xlsx",
                           sheet_name='6월편성', skiprows=1)
    else:
        pf = pd.read_excel("prep/data/01_제공데이터/실적데이터_v1.xlsx", sheet_name = "rawdata_2019(완)",
                         skiprows=1)

    print(len(pf))
    pf.loc[pf['노출(분)'].isnull(), '노출(분)'] = \
        pf.loc[pf['노출(분)'].isnull(), '방송일시'].map(pf.loc[pf['노출(분)'].notnull()] \
                                                 .set_index('방송일시')['노출(분)'])
    # 방송일시 처리
    print("----방송일시 처리----")
    pf['real_date'] = pd.to_datetime(pf['방송일시'].dt.date)
    pf['week'] = pf['방송일시'].dt.dayofweek
    pf['weekofyear'] = pf['방송일시'].dt.week
    pf['year'] = pf['방송일시'].dt.year
    pf['month'] = pf['방송일시'].dt.month
    pf['time'] = pf['방송일시'].dt.time
    pf['hour'] = pf['방송일시'].dt.hour.astype('U')

    pf['date'] = pf.apply(lambda x: caldate(x['real_date'], x['time']), axis=1)
    pf['week'] = pf['date'].dt.dayofweek
    # print(pf.head())

    # add Stage column
    print("----방송 초/중/후반 정보 삽입----")
    sample = pf[['방송일시', 'date', '상품코드', '노출(분)']]
    sample['cumcount'] = sample.groupby(['date', '상품코드'])['노출(분)'].cumcount() + 1
    sample2 = sample.groupby(['date', '상품코드'])['노출(분)'].count().reset_index()
    sample2.rename({'노출(분)': 'count'}, axis=1, inplace=True)
    sample = sample.merge(sample2, on=['date', '상품코드'])
    sample['p'] = sample['cumcount'] / sample['count']

    sample['stage'] = sample.apply(stage_advanced, axis=1)
    pf = pf.merge(sample[['방송일시', '상품코드', 'stage']], on=['방송일시', '상품코드'])
    del sample, sample2


    # prime time
    print("----Prime Time----")
    if test == True:
        with open('prime.txt', 'rb') as fp:
            # test 데이터는 기존 prime.txt 파일을 불러온다.
            prime = pickle.load(fp)
    else:
        prime = prime_set(pf)
        with open("prep/prime.txt", 'wb') as fp:
            pickle.dump(prime, fp)
    pf['prime'] = pf.apply(lambda x: make_prime(x['week'],
                                                  x['hour'],
                                                  prime), axis=1)
    # print(pf.head())

    # 휴일
    print("----Holiday----")
    if test == True:
        pf['IsHoliday'] = np.nan
        pf.loc[(pf['week']==5)|(pf['week']==6), 'IsHoliday'] = 1
        pf['IsHoliday'] = pf['IsHoliday'].fillna(0)

        pf['지속휴일수'] = np.nan
        temp = 0
        for i in range(len(pf)):
            try:
                if pf['IsHoliday'][i] == 1:
                    temp+=1
                else:
                    pf['지속휴일수'][i-1] = temp
                    temp=0 # reset
            except:
                pass
        pf.loc[pf['IsHoliday'] == 0, '지속휴일수']=0
        pf['지속휴일수'] = pf['지속휴일수'].fillna(method='bfill')
    else:
        new_row = {'date': pd.to_datetime('2020-01-01'), '설명': '새해', '요일': '0', 'IsHoliday': 1, '지속휴일수': 1}
        holi = pd.read_csv("prep/data/2019_2020년도_지속_휴일_수.csv", index_col=0,
                           parse_dates=['date'])
        holi = holi.append(new_row, ignore_index=True)
        holi = holi[['date', 'IsHoliday', '지속휴일수']]
        holi.rename(columns={'date': 'real_date'}, inplace=True)
        pf = pf.merge(holi, on=['real_date'])


    # CPI
    print("----CPI----")
    cpi = CPI()
    pf = pf.merge(cpi, on = ['year', 'month'])


    # 날씨
    print("----Weather----")
    wt = weather()
    wt['time'] = pd.to_datetime(wt['time'])
    wt['date'] = pd.to_datetime(wt['time'].dt.date)
    wt['hour'] = wt['time'].dt.hour.astype('U')
    del wt['time']
    wt.rename(columns={'date': 'real_date'}, inplace=True)

    pf = pf.merge(wt, on=['real_date', 'hour'])


    if test == True:
        print("Final Process for test data")
        pf = pf[pf.판매단가.isna() == False][pf.상품군!='무형']
        return pf
    print("Final Process for train data")
    pf = pf[pf.취급액 != 50000][pf.판매단가.isna() == False][pf.상품군!='무형']
    pf = data_stat(pf)
    pf.to_csv("prep/data/final_performance_1.csv", index = False,
              encoding='utf-8')
    return pf
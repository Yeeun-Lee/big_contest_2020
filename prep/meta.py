import pandas as pd
import numpy as np
from datetime import datetime
import os

"""
** 서강패럿 - 김예지, 이예은 **
메타 데이터 전처리(2019.01 ~ 2020.07)
1. CPI(경제지표, KOSIS)
2. 날씨 데이터(기온, 습도, 기상자료개방포털)
    2.1 지역별 영향을 반영하기 위해 월별 지역 인구데이터 활용(KOSIS)
        - 인구 : http://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_1B040A3
        
"""
def region(x, location = None):
    for key, value in location.items():
        if x in value:
            return key
def population():
    pop = pd.read_csv("prep/data/meta/시군구_성별_인구수.csv", encoding='cp949', engine = 'python').T
    pop.columns = pop.iloc[0]
    pop = pop.drop(pop.index[0])
    pop = pop[pop.columns[1:]].reset_index(drop=True)
    pop.columns.name = None
    pop.index = pd.date_range('2019-01', '2020-08', freq='M').strftime('%Y-%m')
    pop = pop.apply(pd.to_numeric)
    pop = pop.apply(lambda x: devide(x, pop['전국']))

    pop = pop.unstack().reset_index()
    pop.columns = ['region', 'month', 'p']
    return pop

def devide(x, total):
    return x/total

def weather(path = "prep/data/meta/weathers"):
    # 시도별 관측소 처리를 위한 사전
    location = {'서울특별시': ['관악산', '서울'],
                '부산광역시': ['부산'],
                '대구광역시': ['대구', '대구(기)'],
                '인천광역시': ['강화', '백령도', '인천'],
                '광주광역시': ['광주'],
                '대전광역시': ['대전'],
                '울산광역시': ['울산'],
                '경기도': ['동두천', '수원', '양평', '이천', '파주'],
                '강원도': ['강릉', '대관령', '동해', '북강릉', '북춘천', '삼척',
                        '속초', '영월', '원주', '인제', '정선군', '철원', '춘천',
                        '태백', '홍천'],
                '충청북도': ['보은', '제천', '청주', '추풍령', '충주'],
                '충청남도': ['금산', '보령', '부여', '서산', '천안', '홍성'],
                '전라북도': ['고창', '고창군', '군산', '남원', '부안', '순창군', '임실',
                         '장수', '전주', '정읍'],
                '전라남도': ['강진군', '고흥', '광양시', '목포', '무안', '보성군',
                         '순천', '여수', '영광군', '완도', '장흥', '주암',
                         '진도(첨찰산)', '진도군', '해남', '흑산도'],
                '경상북도': ['경주시', '구미', '문경', '봉화', '상주', '안동', '영덕',
                         '영주', '영천', '울릉도', '울진', '의성', '청송군', '포항'],
                '경상남도': ['거제', '거창', '김해시', '남해', '밀양', '북창원', '산청',
                         '양산시', '의령군', '진주', '창원', '통영', '함양군', '합천'],
                '제주도': ['고산', '서귀포', '성산', '성산포', '제주'],
                '세종특별자치시': ['세종']
                }
    weather_list = ['weather.csv', 'weather1.csv', 'weather2.csv', 'weather3.csv']
    weathers = pd.DataFrame()
    for data in weather_list:
        weathers = pd.concat([weathers, pd.read_csv(os.path.join(path, data), encoding='cp949')])
    weathers = weathers.interpolate(method='values')
    weathers = weathers.drop(['지점', '강수량(mm)'], axis = 1)

    weathers.columns = ['loc', 'time', 'temp', 'hum']

    weathers['region'] = weathers['loc'].apply(lambda x: region(x, location = location))

    pop_p = population()
    mg = weathers.groupby(['time', 'region']).mean().reset_index(drop=False)
    mg['month'] = mg['time'].apply(lambda x: datetime.strptime(x,
                                                               "%Y-%m-%d %H:%M").strftime("%Y-%m"))
    mg = mg.merge(pop_p, on = ['month', 'region'])
    mg['TEMP'] = mg['temp']*mg['p']
    mg['HUM'] = mg['hum']*mg['p']
    mg = mg[['time', 'region', 'TEMP', 'HUM']]
    mg = mg.groupby(['time']).sum().reset_index(drop=False)
    # 날씨데이터
    return mg

def CPI():
    cpi = pd.read_excel("prep/data/meta/CPI.xlsx", header=0).drop(['시도별'], axis=1).set_index('품목별')
    cpi = cpi.loc['생활물가지수']

    cpi_1 = pd.read_excel("prep/data/meta/CPI_2020.xlsx", header=0).drop(['시도별'], axis=1).set_index('품목별')
    cpi_1 = cpi_1.loc['생활물가지수']
    cpif = pd.DataFrame(pd.concat([cpi, cpi_1])).reset_index().rename(columns = {'index':'date',
                                                                                 '생활물가지수':'cpi'})
    cpif['year'] = pd.to_datetime(cpif['date']).dt.year
    cpif['month'] = pd.to_datetime(cpif['date']).dt.month
    del cpif['date']

    return cpif
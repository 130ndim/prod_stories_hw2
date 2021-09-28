import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder


def prepare_data(data):
    data = data.copy()
    data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'],
                                      format='%Y-%m-%d %H:%M:%S')
    data.drop(['time', 'date', 'id'], axis=1, inplace=True)
    data = data.groupby('deal_id').apply(
        lambda x: x.loc[x['datetime'].idxmin()]).reset_index(drop=True)

    # Разделим данные по платформам и будем обучать отдельные модели под каждую
    data1 = data.loc[data.loc[:, 'platform_id'] == 1]
    data2 = data.loc[data.loc[:, 'platform_id'] == 2]

    # Сделаем ресемплинг по минуте и заполним появившиеся NaN-значения предыдущими
    data1 = data1.set_index('datetime').resample('1Min').aggregate(
        {'lot_size': 'sum', 'price': 'mean'}
    )
    data1.loc[data1.loc[:, 'lot_size'] == 0, 'lot_size'] = np.nan
    valid_dates1 = np.isin(data1.index.date,
                           np.unique(pd.DatetimeIndex(data['datetime']).date))
    data1 = data1[(10 < data1.index.hour) & (data1.index.hour < 13) & valid_dates1].pad()

    data2 = data2.set_index('datetime').resample('1Min').aggregate(
        {'lot_size': 'sum', 'price': 'mean'}
    )
    data2.loc[data2.loc[:, 'lot_size'] == 0, 'lot_size'] = np.nan
    valid_dates2 = np.isin(data2.index.date,
                           np.unique(pd.DatetimeIndex(data['datetime']).date))
    data2 = data2[(10 < data2.index.hour) & (data2.index.hour < 13) & valid_dates2].pad()

    # Сгенерируем новые session_id по часам
    data1['session_id'] = LabelEncoder().fit_transform(data1.index.floor('1H').asi8)
    data2['session_id'] = LabelEncoder().fit_transform(data2.index.floor('1H').asi8)

    return data1, data2

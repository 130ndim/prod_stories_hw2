import argparse
import datetime
from sqlite3 import connect
import time
import warnings

import pandas as pd

from models import Baseline, ARIMAPredictor
from preprocessing import prepare_data


parser = argparse.ArgumentParser()
parser.add_argument(
    '-split_date', '--split_date', dest='split_date',
    type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d')
)
args = parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                            FutureWarning)
    warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                            FutureWarning)

    con = connect('./trade_info.sqlite3')
    data = pd.read_sql(
        """
        SELECT * FROM Chart_data C
        JOIN Trading_session T ON C.session_id=T.id
        WHERE T.trading_type = 'monthly'
        """,
        con
    )

    data1, data2 = prepare_data(data)
    train_data1, test_data1 = (
        data1.loc[data1.index < args.split_date],
        data1.loc[data1.index >= args.split_date]
    )
    train_data2, test_data2 = (
        data2.loc[data2.index < args.split_date],
        data2.loc[data2.index >= args.split_date]
    )

    print('First exchange dataset')
    out1 = test_data1.iloc[1:].copy()
    time1 = time.time()
    base1 = Baseline(train_data1).predict(test_data1.iloc[:-1])
    base_time1 = time.time()
    arima1 = ARIMAPredictor(train_data1, order=(5, 1, 0)).predict(test_data1.iloc[:-1])
    arima_time1 = time.time()

    print(f'Baseline inference took %.2f seconds' % (base_time1 - time1))
    print(f'ARIMA inference took %.2f seconds' % (arima_time1 - base_time1))

    out1['arima'] = arima1['price'].values
    out1['baseline'] = base1['price'].values

    print('Second exchange dataset')
    out2 = test_data2.iloc[1:].copy()
    time2 = time.time()
    base2 = Baseline(train_data2).predict(test_data2.iloc[:-1])
    base_time2 = time.time()
    arima2 = ARIMAPredictor(train_data2, order=(5, 1, 0)).predict(test_data2.iloc[:-1])
    arima_time2 = time.time()

    print(f'Baseline inference took %.2f seconds' % (base_time2 - time2))
    print(f'ARIMA inference took %.2f seconds' % (arima_time2 - base_time2))

    out2['arima'] = arima2['price'].values
    out2['baseline'] = base2['price'].values

    out1.to_csv('data1_out.csv')
    out2.to_csv('data2_out.csv')

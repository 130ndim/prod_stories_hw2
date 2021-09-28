import numpy as np

from statsmodels.tsa.arima_model import ARIMA


class Baseline:
    """Бейзлайн из презентации"""
    def __init__(self, train):
        means = train.groupby(
            [train.index.month, train.index.hour, train.index.minute]
        )['price'].mean().copy()
        self.shifts = (means.loc[np.roll(means.index, -1)] - means.values).reindex(
            means.index).copy()

    def predict(self, X):
        X = X.copy()
        X['price'] = X['price'] + self.shifts.loc[
            zip(X.index.month, X.index.hour, X.index.minute)
        ].values
        return X


class ARIMAPredictor:
    """Обертка над ARIMA из statsmodels"""
    def __init__(self, train, **arima_kws):
        self._train = train.loc[:, 'price']
        self.arima_kws = arima_kws

    def predict(self, X):
        out = X.copy()
        data = self._train.tolist()
        predictions = []
        for point in X['price'].tolist():
            model = ARIMA(data, **self.arima_kws).fit(disp=0)
            pred = model.forecast()[0][0]
            predictions.append(pred)
            data.append(point)

        out['price'] = predictions
        return out

# 用上海二手房价的例子运用线性回归

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


if __name__ == '__main__':
    all_data = pd.read_csv(os.path.join('data', '上海二手房价.csv'))

    prices = all_data['房价（元/平米）'].values.reshape(-1, 1)
    pri_scaler = MinMaxScaler()
    pri_scaler.fit(prices)
    st_prices = pri_scaler.transform(prices)

    floors = all_data['楼层'].values.reshape(-1, 1)
    flo_scaler = MinMaxScaler()
    flo_scaler.fit(floors)
    st_floors = flo_scaler.transform(floors)

    years = all_data['建成年份'].values.reshape(-1, 1)
    year_scaler = MinMaxScaler()
    year_scaler.fit(years)
    st_years = year_scaler.transform(years)

    features = np.stack([st_floors, st_years], axis=-1).squeeze(axis=1)

    k = np.array([1.0, 1.0]).reshape(2, 1)
    b = 0

    epoch = 10
    lr = 0.5

    for e in range(epoch):
        pre = features @ k + b
        mean_loss = ((pre - st_prices) ** 2)

        G = (pre - st_prices) / pre.shape[0]

        delta_k = features.T @ G

        k -= delta_k * lr

        print(mean_loss)



    print("")
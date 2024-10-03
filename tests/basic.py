import numpy as np
from sklearn.pipeline import Pipeline

from preprokit.filters import CustomFilters
import pandas as pd


def test_pandas():
    print("Testing")
    data = {
        'values1': ['3.14', '3.14', '?', '⧼', '3.14', 'hello', '123'],
        'values2': ['3.14', '-', '?', '⧼', '123', 'hello', '123']
    }
    df = pd.DataFrame(data)
    pipeline = Pipeline(
        steps=[
            ("custom filters", CustomFilters(simple=True))
        ]
    )

    print(pipeline.fit_transform(df))

def test_numpy():
    print("Numpy test")
    data = [
            ['3.14', '3.14'],
            ['3.14', '-'],
            ['?', '?'],
            ['⧼', '⧼'],
            ['3.14', '123'],
            ['hello', 'hello'],
            ['123', '123']
    ]
    data = np.array(data)
    pipeline = Pipeline(
        steps=[
            ("custom filters", CustomFilters(simple=True))
        ]
    )

    print(pipeline.fit_transform(data))

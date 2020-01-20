import numpy as np

def smape(y_true, y_pred):
    F = y_pred
    A = y_true.values
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def smape_fast(y_true, y_pred):
    out = 0
    index = 0
    for value in y_true:
        a = value
        b = y_pred[index]
        index +=1
        c = a + b
        if c == 0:
            continue
        out += math.fabs(a - b) / c
        
    out *= (200.0 / index)
    print(index)
    return out
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Load data
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')


def trace_for_all_data(x_train, y_train, vector_like=True):
    if vector_like:
        for i in range(len(x_train)):
            yield (x_train[i]).reshape(2, 1), y_train[i]
    else:
        for i in range(len(x_train)):
            yield x_train[i], y_train[i]

# 1. Compute the mean vectors mi, (i=1,2) of each 2 classes


def get_mi(x_train, y_train):
    n_1 = 0
    # sum_1 = np.zeros(2)
    # sum_2 = np.zeros(2)
    # for i, e in enumerate(y_train):
    #     print(f'dot: ({x_train[i][0]}, {x_train[i][1]}) -> {y_train[i]}')
    #     if e == 0:
    #         sum_1 += x_train[i]
    #         n_1 += 1
    #     else:
    #         sum_2 += x_train[i]
    #     print(f'update sum: s1= {sum_1}, s2= {sum_2}')
    # return sum_1/ n_1, sum_2/ (len(x_train) - n_1)

    sum_ = np.zeros((2, 2, 1))
    for data in trace_for_all_data(x_train, y_train):
        sum_[data[1]] += data[0]
        n_1 += data[1] == 0
        # print(f'data: {data}')
        # print(f'update sum: s1= {sum_[0]}, s2= {sum_[1]}')
        # print(f'n_1: {n_1}')
    return sum_[0] / n_1, sum_[1] / (len(x_train) - n_1)
    # training data:
    # mean vector of class 1: [ 1.3559426  -1.34746216]
    # mean vector of class 2: [-1.29735587  1.29096203]

m1, m2 = get_mi(x_train, y_train)
print(f"mean vector of class 1:\n{m1}\nmean vector of class 2:\n{m2}\n")
# =================================================================

# 2. Compute the Within-class scatter matrix SW


def get_SW(x_train, y_train, m=None):
    '''
    m: [ m1: 2x1 vector, m2: 2x1 vector ]
    '''
    if m is None:
        m = get_mi(x_train, y_train)

    # Si = sum( (xi - m)(xi - m)T )
    # 2x2 matrix, if xi is 2-D data
    S = np.zeros((2, 2, 2))
    for data in trace_for_all_data(x_train, y_train):
        sub = data[0] - m[data[1]]
        S[data[1]] += sub @ sub.T
        # print(f'   X: {data[0]}')
        # print(f'mean: {m[data[1]]}')
        # print(f' sub: {sub}')
        # print(f'sub @ subT: {sub@sub.T}')
        # print(f'S[{data[1]}]: {S[data[1]]}')
    return S[0] + S[1]

sw = get_SW(x_train, y_train, (m1, m2))
assert sw.shape == (2, 2)
print(f"Within-class scatter matrix SW:\n{sw}\n")
# =================================================================

# 3. Compute the Between-class scatter matrix SB


def get_SB(m1, m2):
    sub = m2 - m1
    return sub @ sub.T

sb = get_SB(m1, m2)
assert sb.shape == (2, 2)
print(f"Between-class scatter matrix SB:\n{sb}\n")
# =================================================================

# 4. Compute the Fisher’s linear discriminant


def get_W(SW, m1, m2):
    w = inv(SW) @ (m2 - m1)
    # |w| = sqrt( wTw )
    # print(f'origin w: {w}')
    w = w / (w.T @ w)**(1/2)
    # print(f' after w: {w}')
    return w

w = get_W(sw, m1, m2)
assert w.shape == (2, 1)
print(f" Fisher’s linear discriminant:\n{w}\n")
# =================================================================

# 5. Project the test data by linear discriminant
#    to get the class prediction by nearest-neighbor rule
#    and calculate the accuracy score


def predict(x_train, y_train, w, x_test):
    '''
    x_test: [
        [x0, x1],
        [x0, x1],
        ...
    ]
    y_test: [
        1, 0, 0, 1, 1, ...
    ]
    w: [
        [w0],
        [w1]
    ]
    return [
        1, 0, 0, 1, 1
    ]
    x_train 投影到 w 上的長度(正負)差異
    x_train @ w = [
        [1.342],
        [-0.023],
        ...
    ]
    '''
    x_train_project = x_train @ w
    x_test_project = x_test @ w

    check_table = np.append(x_train_project,
                            y_train.reshape((-1, 1)),
                            axis=1)
    # check table:
    # [[project val, catgory],
    #  [project val, catgory]]
    # print(check_table)

    # sort by [proj val, ]
    check_table = check_table[np.argsort(check_table[:, 0])]
    # find no. of x_test( by project val to w )
    search_result = np.searchsorted(check_table[:, 0], x_test_project)
    # [1 3 5], 2 -> place in pos "1" -> check distance to "0" and "1"
    # print(f'check:\n{check_table}')
    # print(f'resul:\n{search_result}')

    y_pred = []
    for i, e in enumerate(search_result):
        e = e[0]  # U 夠麻煩...
        # x_test[i] between check table [e-1] to [e]
        # print(f'{x_test_project[i][0]} - {check_table[e-1][0]} vs.')
        # print(f'{check_table[e][0]} - {x_test_project[i][0]}')
        if x_test_project[i][0] - check_table[e-1][0] < \
           check_table[e][0] - x_test_project[i][0]:
            # if distance to e-1 is closer than
            #    distance to e
            y_pred += [check_table[e-1][1]]
        else:
            y_pred += [check_table[e][1]]
    return np.array(y_pred)

y_pred = predict(x_train, y_train, w, x_test)


def accuracy_score(y_test, y_pred):
    '''
    y_test: [1, 0, 0, ...]
    y_pred: [1, 1, 0, ...]

    correct / all
    (y_test == y_pred) / all

    not correct: abs(y_test - y_pred)
    (all-abs(y_test-y_pred)) / all
    '''
    n = len(y_test)
    return (n-sum(abs(y_test-y_pred))) / n

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of test-set {acc}")
# =================================================================

# 6. Plot the
# 1) best projection line on the training data and show the slope
#    and intercept on the title
#    (you can choose any value of intercept for better visualization)
# 2) colorize the data with each class
# 3) project all data points on your projection line.
# Your result should look like this image


def partition(x_train, y_train):
    '''
    return [
        [x0, x1],
        [x0, x1],
        ...
        [x0, x1]
    ]
    '''
    collect_0 = []
    collect_1 = []
    for x, y in trace_for_all_data(x_train, y_train, vector_like=False):
        if y == 0:
            # plt.plot([x[0]], [x[1]], 'b.')
            collect_0 += [x]
        else:
            # plt.plot([x[0]], [x[1]], 'r.')
            collect_1 += [x]
    collect_0 = np.array(collect_0)
    collect_1 = np.array(collect_1)
    return collect_0, collect_1


def Plot_point(x_train, y_train):
    collect_0, collect_1 = partition(x_train, y_train)
    # plt.plot(collect_0[0][0], collect_0[0][1], 'b.')
    plt.plot(collect_0.T[0], collect_0.T[1], 'bo')
    plt.plot(collect_1.T[0], collect_1.T[1], 'ro')


def Plot_line(w, b, from_x, to_x):
    # y_dist / x_dist
    a = w[1][0] / w[0][0]
    # y = a x + b
    plt.title(f'ProjectionLine: w={a}, b={b}')
    plt.plot(
        [from_x, to_x],
        [a * from_x + b, a * to_x + b],
        'k-'
    )


def Plot_point_on_line(x_train, y_train, w, b, from_x, to_x):
    collect_0, collect_1 = partition(x_train, y_train)
    # x0, x1 = collect_0[0]
    # project = (x0*w[0][0] + (x1-b)*w[1][0]) * w
    # point = [project[0][0], project[1][0] + b]
    # plt.plot(*point, 'b.')

    # plt.plot([x0, point[0]], [x1, point[1]], 'b--')
    # # a = (x1-x0)*w[0] + (point[1] - point[0])*w[1]
    # print()

    # ================================================

    c_0 = []
    c_1 = []
    pairs = []
    for x0, x1 in collect_0:
        # project = np.array([[x0], [x1-b]]).dot(w) * w
        project = (x0*w[0][0] + (x1-b)*w[1][0]) * w
        point = [project[0][0], project[1][0] + b]
        c_0 += [point]
        pairs += [[[x0, x1], point]]
    for x0, x1 in collect_1:
        # project = np.array([[x0], [x1-b]]).dot(w) * w
        project = (x0*w[0][0] + (x1-b)*w[1][0]) * w
        point = [project[0][0], project[1][0] + b]
        c_1 += [point]
        pairs += [[[x0, x1], point]]

    c_0 = np.array(c_0)
    c_1 = np.array(c_1)
    pairs = np.array(pairs)

    plt.plot(c_0.T[0], c_0.T[1], 'bo')
    plt.plot(c_1.T[0], c_1.T[1], 'ro')
    for p1, p2 in pairs:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'c:')


def Plot_all(x_train, y_train, w, b, from_x, to_x):
    # set x, y size
    plt.figure(figsize=(8, 8), dpi=80)
    plt.xlim(-6, 4)
    plt.ylim(-5, 5)

    Plot_point(x_test, y_test)
    Plot_line(**line_information)
    Plot_point_on_line(x_test, y_test, **line_information)
    plt.show()

line_information = {
    'w': w,
    'b': -3.5,
    'from_x': -6,
    'to_x': 4
}

Plot_all(x_train, y_train, **line_information)

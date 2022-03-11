import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

η = 0.001  # learning rate

train_df = pd.read_csv("train_data.csv")
x_train, y_train = train_df['x_train'], train_df['y_train']

# wired initial weight:
# β0, β1=[
#     (-2.8878, -0.7567),
#     (-1.4769, -0.7439),
#     (-0.265, -1.3396)
# ]


def MSE(β0, β1, x_data, y_data):
    # y = β0 + β1 * x
    # error = 0
    # for x, y in zip(x_data, y_data):
    #     error += (β0 + β1*x - y) ** 2
    return sum((β0 + β1*x_data - y_data) ** 2) / x_data.size


def MAE(β0, β1, x_data, y_data):
    # y = β0 + β1 * x
    # error = 0
    # for x, y in zip(x_data, y_data):
    #     error += abs(β0 + β1*x - y)
    return sum(abs(β0 + β1*x_data - y_data)) / x_data.size


def f(β0, β1, X):
    return β0 + β1*X


def plot_line(β0, β1, X, Y):
    x_min = X.min() - 1
    x_max = X.max() + 1
    # x_dist = int(x_max - x_min) + 1
    # dots = np.linspace(int(x_min)-1, int(x_max)+1, x_dist*3)

    plt.plot(X, Y, '.')
    plt.plot([x_min, x_max], f(β0, β1, np.array([x_min, x_max])))


def MSE_train(x_train, y_train):
    # IN:   [x_train], [y_train]
    # OUT:  [iter_loss], β0, β1
    n = x_train.shape[0]
    β0, β1 = np.random.normal(0, 1, 2)  # μ, σ, shape
    loss = MSE(β0, β1, x_train, y_train)
    loss_record = [loss]
    pre_loss = loss + 1
    # 1st derivation
    # β0 2/n ∑(β0+β1*X_i-Y_i)
    # β1 2/n ∑(β0+β1*X_i-Y_i)*β1
    while(abs(loss - pre_loss) > 0.000001):
        # print("η:", η)
        sum_β0_β1_X_Y = sum(β0 + β1 * x_train - y_train)
        β0 -= η * 2 * sum_β0_β1_X_Y / n
        β1 -= η * 2 * sum_β0_β1_X_Y * β1 / n
        pre_loss = loss
        loss = MSE(β0, β1, x_train, y_train)
        loss_record += [loss]
        print(loss, β0, β1)
    return loss_record, β0, β1


def MAE_train(x_train, y_train):
    # IN:   [x_train], [y_train]
    # OUT:  [iter_loss], β0, β1
    n = x_train.shape[0]
    β0, β1 = np.random.normal(0, 1, 2)  # μ, σ, shape
    loss = MAE(β0, β1, x_train, y_train)
    loss_record = [loss]
    pre_loss = loss + 1
    # 1st derivation
    # β0 2/n ∑(+1 if β0+β1*X_i-Y_i > 0
    #           0 if β0+β1*X_i-Y_i ==0
    #          -1 if β0+β1*X_i-Y_i < 0)
    # β1 2/n ∑(+1 if β0+β1*X_i-Y_i > 0
    #           0 if β0+β1*X_i-Y_i ==0
    #          -1 if β0+β1*X_i-Y_i < 0) * β1
    while(abs(loss - pre_loss) > 0.000001):
        sum_β0_β1_X_Y = sum(
            1 if e > 0 else -1
            for e in (β0 + β1 * x_train - y_train)
        )  # 假裝 β0 + β1 * x_train - y_train == 0 沒發生
        β0 -= η * 2 * sum_β0_β1_X_Y / n
        β1 -= η * 2 * sum_β0_β1_X_Y * β1 / n

        pre_loss = loss
        loss = MAE(β0, β1, x_train, y_train)
        loss_record += [loss]
        print(loss, β0, β1)
    return loss_record, β0, β1

# ----------------------------------------------------------------

loss, β0, β1 = MSE_train(x_train, y_train)
# print("loss:", loss[-1])
# print("β0, β1: ", β0, β1)
# plot_line(β0, β1, x_train, y_train)
b = plt.figure(2)
plt.title('MSE')
plt.plot(range(len(loss)), loss)


loss, β0, β1 = MAE_train(x_train, y_train)
c = plt.figure(3)
plt.title('MAE')
plt.plot(range(len(loss)), loss)
plt.show()

# ----------------------------------------------------------------

# TEST
test_data = pd.read_csv("test_data.csv")
x_test, y_test = test_data['x_test'], test_data['y_test']

# print("test data error:", MSE(β0, β1, x_test, y_test))
# print("test data error:", MAE(β0, β1, x_test, y_test))

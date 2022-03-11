import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score

K_for_kfold = 10  # kfold
Q5only = True
η = 0.001  # HW1 need

# sample code from matplotlib:
# https://matplotlib.org/stable/gallery/images_contours_and_fields/
#         image_annotated_heatmap.html


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# ## Load data
x_train = np.load("dataset/x_train.npy")
y_train = np.load("dataset/y_train.npy")
x_test = np.load("dataset/x_test.npy")
y_test = np.load("dataset/y_test.npy")

# 550 data with 300 features
print(x_train.shape)
# It's a binary classification problem
print(np.unique(y_train))

# Question 1
# K-fold data partition: Implement the K-fold cross-validation function.


def cross_validation(x_train, y_train, k=5):
    ind = np.arange(x_train.shape[0])
    np.random.shuffle(ind)
    mod_k = x_train.shape[0] % k
    part_size = x_train.shape[0] // k
    # seperate
    kfold = []
    for valid_i in range(k):
        train_part = []
        valid_part = None
        from_i = 0
        to_i = 0
        for i in range(k):
            # part i: [0~ part_size], [part_size, 2*part_size], ...
            # validation: valid_i, else: i except valid_i
            from_i = to_i
            step_size = part_size + (i < mod_k)
            to_i = from_i + step_size

            if i == valid_i:
                valid_part = ind[from_i: to_i]
            else:
                train_part += [ind[from_i: to_i]]
        # TRAIN data IS list of numpy arr!!
        # change to np array
        train_part = np.hstack(train_part)
        kfold += [[train_part, valid_part]]
    return kfold

kfold_data = cross_validation(x_train, y_train, k=K_for_kfold)

assert len(kfold_data) == 10  # should contain 10 fold of data
assert len(kfold_data[0]) == 2  # each element should contain train, valid fold
assert kfold_data[0][1].shape[0] == 55  # The number of data in each validation

with open(r'Q1 kfold.txt', 'w') as f:
    print('output Q1 kfold result to /Q1 kfold.txt')
    for (train_id, valid_id) in kfold_data:
        train_id.sort()
        valid_id.sort()
        print(f'- Train: {train_id}', file=f)
        print(f'- Valid: {valid_id}', file=f)
        print('', file=f)

# Question 2


def gen(begin, end):
    '''
    generate numbers from begin to end, (*= 10)
    gen_C = gen(0.01, 10000.0) -> [0.01, 0.1, ..., 10000.0]
    gen_γ = gen(0.0001, 1000.0) -> [0.0001, ..., 1000.0]
    '''
    while begin != end:
        yield begin
        begin *= 10
    yield begin

if not Q5only:
    data = np.zeros((7, 8))  # 7 rows, 8 cols
    best_param = (None, None)  # (C, γ)
    best_score = 0

    pool_γ = list(gen(0.0001, 1000.0))
    pool_C = list(gen(0.01, 10000.0))
    # 逐行 由左往右掃描(抓 score)
    for i, C in enumerate(pool_C):
        for j, γ in enumerate(pool_γ):
            print(f'start testing with (C, γ) = ({C: >10.5}, {γ: >10.5})')
            score = 0
            for (train_id, valid_id) in kfold_data:
                clf = SVC(C=C, kernel='rbf', gamma=γ)
                clf.fit(x_train[train_id], y_train[train_id])
                result = clf.predict(x_train[valid_id])
                score += accuracy_score(result, y_train[valid_id])

            score /= K_for_kfold
            data[i, j] = score
            if score > best_score:
                best_score = score
                best_param = (C, γ)

    print(f'Best C: {best_param[0]}, γ: {best_param[1]}')
    print(f'with score: {best_score}\n')

    # Question 3
    im, cbar = heatmap(
        data,
        [f'C: {e}' for e in pool_C],
        [f'γ: {e}' for e in pool_γ],
        cmap='YlGn',
        cbarlabel='Accuracy Score'
    )
    texts = annotate_heatmap(im, valfmt='{x:.2f}')

    plt.tight_layout()
    plt.show()

    # Question 4

    best_model = SVC(C=best_param[0], kernel='rbf', gamma=best_param[1])
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    print("Accuracy score: ", accuracy_score(y_pred, y_test))

###############################################################################
# Question 5
# Compare the performance of each model you have implemented from HW1
# code from HW1


def MSE(x_data, y_data, β0=None, β1=None, value_only=False):
    '''
    value_only -> cmp(x_data, y_data)
    else -> cmp(β1 * x_data + β0, y_data)
    '''
    if value_only:
        return np.sum((x_data - y_data) ** 2) / x_data.shape[0]
    else:
        return np.sum((β0 + β1*x_data - y_data) ** 2) / x_data.size
        # return np.sum((β0 + β1*x_data - y_data) ** 2) / x_data.size


def MSE_train(x_train, y_train):
    # IN:   [x_train], [y_train]
    # OUT:  [iter_loss], β0, β1
    n = x_train.shape[0]
    β0, β1 = np.random.normal(0, 1, 2)  # μ, σ, shape
    loss = MSE(x_train, y_train, β0, β1)
    loss_record = [loss]
    pre_loss = loss + 1

    num_i = 0
    while abs(loss - pre_loss) > 0.00000001 and num_i < 3000:
        # print("η:", η)
        β0_β1_X_Y = β0 + β1 * x_train - y_train
        β0 -= η * 2 * (np.sum(β0_β1_X_Y) / n)
        β1 -= η * 2 * (np.sum(β0_β1_X_Y * x_train) / n)
        pre_loss = loss
        loss = MSE(x_train, y_train, β0, β1)
        loss_record += [loss]
        # print(loss, β0, β1)
        num_i += 1
    return loss_record, β0, β1

# ### HW1
train_df = pd.read_csv("dataset/train_data.csv")
x_train = train_df['x_train'].to_numpy().reshape(-1)
x_train_for_SVR = x_train.reshape(-1, 1)
y_train = train_df['y_train'].to_numpy().reshape(-1)

test_df = pd.read_csv("dataset/test_data.csv")
x_test = test_df['x_test'].to_numpy().reshape(-1)
x_test_for_SVR = x_test.reshape(-1, 1)
y_test = test_df['y_test'].to_numpy().reshape(-1)
# -----
K_for_kfold = 5
kfold_data = cross_validation(x_train, y_train, k=K_for_kfold)

with open(r'Q5-1 kfold.txt', 'w') as f:
    print('output Q5-1 kfold result to /Q5-1 kfold.txt')
    for (train_id, valid_id) in kfold_data:
        train_id.sort()
        valid_id.sort()
        print(f'- Train: {train_id}', file=f)
        print(f'- Valid: {valid_id}', file=f)
        print('', )

# -----

data = np.zeros((7, 8))  # 7 rows, 8 cols
best_param = (None, None)  # (C, γ)
best_error = 1000

pool_γ = list(gen(0.0001, 1000.0))
pool_C = list(gen(0.01, 10000.0))
# 逐行 由左往右掃描(抓 score)
for i, C in enumerate(pool_C):
    for j, γ in enumerate(pool_γ):
        print(f'start testing with (C, γ) = ({C: >10.5}, {γ: >10.5})')
        # FIND LOWEST ERROR
        error = 0
        for (train_id, valid_id) in kfold_data:
            clf = SVR(C=C, kernel='rbf', gamma=γ)
            clf.fit(x_train_for_SVR[train_id], y_train[train_id])
            result = clf.predict(x_train_for_SVR[valid_id])
            error += MSE(result, y_train[valid_id], value_only=True)

        error /= K_for_kfold
        data[i, j] = error
        if error < best_error:
            best_error = error
            best_param = (C, γ)

print(f'Best C: {best_param[0]}, γ: {best_param[1]}')
print(f'with error: {best_error}\n')
# -----
im, cbar = heatmap(
    data,
    [f'C: {e}' for e in pool_C],
    [f'γ: {e}' for e in pool_γ],
    cmap='YlGn',
    cbarlabel='MSE error'
)
texts = annotate_heatmap(im, valfmt='{x:.2f}')

plt.tight_layout()
plt.show()
# -----

best_model = SVR(C=best_param[0], kernel='rbf', gamma=best_param[1])
best_model.fit(x_train_for_SVR, y_train)
y_pred = best_model.predict(x_train_for_SVR)
print("MSE error :", MSE(y_pred, y_train, value_only=True))

# -----
# HW1 solution: Gradient decent on MSE

loss, β0, β1 = MSE_train(x_train, y_train)
y_pred = best_model.predict(x_test_for_SVR)

print("Square error of Linear regression:", MSE(x_test, y_test, β0, β1))
print("Square error of SVM regresssion model:",
      MSE(y_pred, y_test, value_only=True))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEBUG = False
DEBUGpred = False
DEBUGrandomforest = False

EXE_Question_List = {
    'Q2.1': True,
    'Q2.2': True,
    'Q3': True,
    'Q4.1': True,
    'Q4.2': True
}

# Load Data

x_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")
# add attribute y
# x_train['y'] = y_train

x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")

# Question 1: Criterion


def gini(sequence):
    # sequence: [yi, yi, ...], yi ∈ {0, 1}
    n = len(sequence)
    labels, counts = np.unique(sequence, return_counts=True)

    probabilities = counts / n
    return 1 - np.sum(probabilities ** 2)


def entropy(sequence):
    # sequence: [yi, yi, ...], yi ∈ {0, 1}
    n = len(sequence)
    labels, counts = np.unique(sequence, return_counts=True)

    probabilities = counts / n
    return -1 * np.sum(probabilities * np.log2(probabilities))

Criterion = {
    'gini': gini,
    'entropy': entropy
}

dataT = np.array([1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2])
print("Gini of data is ", gini(dataT))
print("Entropy of data is ", entropy(dataT))

# Question 2


class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterionFunction = Criterion[criterion]
        self.max_depth = max_depth
        self.feature_names = None
        self.root = None
        self.used_attribute = set()
        # choose ONLY useful attribute when PREDICTION
        self.feature_instance = None

    def fit(self, X, y):
        self.feature_names = list(X.columns)

        WholeTrueSeries = pd.Series(data=([True]*len(X)))
        X_copy = X.copy()
        y_copy = y.copy()
        X_copy['y'] = y_copy

        self.root = TreeNode(X_copy, y_copy, WholeTrueSeries, self, 0)
        self.root.BuildTree()
        return None

    def SetFeatureInstance(self):
        '''
        set value self.feature_instance
        AFTER fit(X, y)
        FI[ attr ] =
            N*( Criterion_ALL ) - N_0*( Criterion_0 ) - N_1*( Criterion_1 )
        where N0, N1 is split by Attr

        pd index = [True, True, False, True, ...]
        N = index.sum()
        '''
        if self.root is None:
            print("[ERROR] DO DecitionTree.fit(X, y) before SetFI()")
            return
        FI = {}  # Empty Dictionary

        NodeTODO = [self.root]
        while NodeTODO:
            Node = NodeTODO.pop()

            if Node.leftNode:
                # have child
                # no child -> no split -> no FI
                if Node.split_attr not in FI:
                    FI[Node.split_attr] = 0

                NodeTODO += [Node.leftNode]
                NodeTODO += [Node.rightNode]

                FI[Node.split_attr] += sum(Node.index) * Node.criterion
                FI[Node.split_attr] -= sum(Node.leftNode.index) *\
                    Node.leftNode.criterion
                FI[Node.split_attr] -= sum(Node.rightNode.index) *\
                    Node.rightNode.criterion

        self.feature_instance = FI

    def predict(self, X):
        '''
        input: X-> dataframe of 1 or more rows

        return: np array
        '''
        X = X[list(self.used_attribute)]
        # choose ONLY used attribute

        X = X.to_dict('records')
        # X -> list of dictionaries
        pred = []
        for x in X:
            # x: dictionary

            Node = self.root
            while Node.predict is None:
                if float(x[Node.split_attr]) < Node.split_threshold:
                    if DEBUGpred:
                        print(f'x[{Node.split_attr}] = {x[Node.split_attr]}' +
                              f' < {Node.split_threshold}')
                    Node = Node.leftNode
                else:
                    if DEBUGpred:
                        print(f'x[{Node.split_attr}] = {x[Node.split_attr]}' +
                              f' >= {Node.split_threshold}')
                    Node = Node.rightNode
            # come to leaf node

            pred += [Node.predict]
        return np.array(pred)

    def Show(self):
        '''
        show shape of tree
        Left {Middle} Right
        '''
        self.root.Show()


class TreeNode():
    def __init__(self, X, y, index, DT: DecisionTree, depth):
        # INITIALIZE value
        self.X = X
        self.y = y
        self.index = index
        self.DT = DT
        # DT.criterion, DT.max_depth
        self.depth = depth
        # self.criterionF = Criterion[criterion]
        # self.max_depth = max_depth
        self.split_attr = None
        self.split_threshold = None
        # < threshold | threshold <=
        self.predict = None

        self.leftNode = None
        self.rightNode = None

        # compute THIS node criterion
        self.criterion = self.DT.criterionFunction(
            np.array(self.X[self.index]['y'].values).reshape((-1))
        )

    def SplitAttribute(self):
        '''
        set attrbute BEST split
                     self.attr, threshold [0, threshold) [threshold, n]
        '''

        # partition: find best attr-threshold
        MinCriterionVal = 10000
        MinAttr = None
        MinSplitNo = 1
        MinSplitThreshold = None
        cF = self.DT.criterionFunction
        for attr in self.DT.feature_names:
            # sorted data = sort data by THIS attr val
            data = self.X[[attr, 'y']][self.index]
            # data: cols=[attr, y], rows=[index]

            SortedData = data.sort_values(by=attr)
            SortedAttrVal = np.array(SortedData[attr].values).reshape(-1)
            Sortedy = np.array(SortedData['y'].values).reshape(-1)

            # SortedData: 0101 array, {ys} sorted by attr value
            N = len(SortedData)

            for i in range(1, N):
                # S[0: 1], S[1, N]
                # ...
                # S[0: N-1], S[N-1: N]
                # skip same value
                # can't split
                # attr: [..., 1.23 | 1.23, ...]
                #    y: [...,    0 |    1, ...]
                if SortedAttrVal[i] == SortedAttrVal[i-1]:
                    continue

                split1, split2 = Sortedy[0: i], Sortedy[i: N]
                # split1_IND, split2_IND =
                #     self.X[attr< THRESHOLD ], self.x[attr>= THRESHOLD]
                # split1, split2 = self.y[split1_IND], self.y[split2_IND]
                # split1,split2 =
                #     np.array(split1.values), np.array(split2.values)
                # same result

                CriterionVal = cF(split1) + cF(split2)
                # smaller -> purer
                if CriterionVal < MinCriterionVal:
                    MinCriterionVal = CriterionVal
                    MinAttr = attr
                    MinSplitNo = i
                    # MinSplitThreshold = SortedData[MinAttr][MinSplitNo]
                    MinSplitThreshold = SortedAttrVal[MinSplitNo]
            # BEST split:
            # S[0: i], S[i, N]

        self.split_attr = MinAttr
        self.split_threshold = MinSplitThreshold
        if MinAttr is None:
            print('[ERROR]: No best split found')
            print()

    def BuildTree(self):
        # max depth
        # set THIS group to one class
        if self.DT.max_depth is not None and self.depth == self.DT.max_depth:
            # set to one group
            class_1 = self.X['y'][self.index].sum()
            class_0 = len(self.X['y'][self.index]) - class_1
            if class_0 > class_1:
                self.predict = 0
            else:
                self.predict = 1
            if DEBUG:
                print(f'STOP because depth limit: {self.DT.max_depth}')
                for i in range(self.depth):
                    print('    ', end='')
                print(f'{self.depth}> ', end='')
                print(' '*25 + f' - {"MAX DEPTH":<10}', end='')
                print('    '*(self.DT.max_depth-self.depth) +
                      f'{self.criterion}')

            return

        # all belong to same class
        if self.criterion == 0:
            self.predict = int(self.y[self.index].values[0])
            if DEBUG:
                print('STOP because all same class')
                for i in range(self.depth):
                    print('    ', end='')
                print(f'{self.depth}> ', end='')
                print(' '*25 + f' - {"SAME CLASS":<10}', end='')
                print('    '*(self.DT.max_depth or 5-self.depth) +
                      f'{self.criterion}')

            return

        self.SplitAttribute()
        # set self.split_attr, self.split_threshold
        self.DT.used_attribute.add(self.split_attr)
        # see DecitionTree.used_attribute for more information

        if DEBUG:
            for i in range(self.depth):
                print('    ', end=f'')
            print(f'{self.depth}> ', end='')
            print(f'{self.split_attr:>25} - {self.split_threshold:<10}',
                  end='')
            print('    '*(self.DT.max_depth or 5-self.depth) +
                  f'{self.criterion}')

        # split Data to two group
        index_l = self.index &\
            (self.X[self.split_attr] < self.split_threshold)
        index_r = self.index &\
            (self.X[self.split_attr] >= self.split_threshold)

        # index_l = self.X[self.index][self.split_attr] < self.split_threshold
        # 不能這樣寫因為 前面會把 index=False 的欄位吃掉

        self.leftNode = TreeNode(self.X, self.y, index_l,
                                 self.DT, self.depth + 1)
        self.leftNode.BuildTree()
        self.rightNode = TreeNode(self.X, self.y, index_r,
                                  self.DT, self.depth + 1)
        self.rightNode.BuildTree()

    def Show(self):
        '''
        Node:
            1. leaf: show class( predict ), criterion
            2. have child: show split method, criterion
        '''
        if self.leftNode:
            # have child
            self.leftNode.Show()
            print(self)
            self.rightNode.Show()
        else:
            print(self)

    def __str__(self):
        # show ONLY this node
        to_return = ''
        to_return += '  '*self.depth
        to_return += f'{self.depth}> '
        to_return += '  '*(self.DT.max_depth-self.depth)
        to_return += f'Crit: {self.criterion:<6.3f}'

        if self.leftNode:
            # have child
            # show ONLY this node: threshold
            to_return += f'{self.split_attr:>25} - {self.split_threshold:<10}'
        else:
            # leaf node
            to_return += f'VALUE: {self.predict:<3}'

        return to_return


def accuracy(pred, real):
    '''
    pred, real: np array of shape (N)

    pred - real: 0-> same, +-1-> diff
    '''
    return (len(pred) - sum(abs(pred-real))) / len(pred)

# Question 2.1

clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
clf_depth10 = DecisionTree(criterion='gini', max_depth=10)

if EXE_Question_List['Q2.1']:
    clf_depth3.fit(x_train, y_train)
    clf_depth3_pred = clf_depth3.predict(x_test)
    print(f'{"Accuracy clf_depth3:":<25}',
          accuracy(clf_depth3_pred, np.array(y_test.values).reshape(-1)))

    clf_depth10.fit(x_train, y_train)
    clf_depth10_pred = clf_depth10.predict(x_test)
    print(f'{"Accuracy clf_depth10:":<25}',
          accuracy(clf_depth10_pred, np.array(y_test.values).reshape(-1)))

# Question 2.2

clf_gini = DecisionTree(criterion='gini', max_depth=3)
clf_entropy = DecisionTree(criterion='entropy', max_depth=3)

if EXE_Question_List['Q2.2']:
    clf_gini.fit(x_train, y_train)
    clf_gini_pred = clf_gini.predict(x_test)
    print(f'{"Accuracy clf_gini:":<25}',
          accuracy(clf_gini_pred, np.array(y_test.values).reshape(-1)))

    clf_entropy.fit(x_train, y_train)
    clf_entropy_pred = clf_entropy.predict(x_test)
    print(f'{"Accuracy clf_entropy:":<25}',
          accuracy(clf_entropy_pred, np.array(y_test.values).reshape(-1)))


# Qustion 3
def PlotFeatureInstance(FI):
    '''
    FI: dict
    '''
    plt.barh(y=[e.replace(' ', '\n') for e in FI],
             width=FI.values(), height=0.5)
    plt.title('Feature Importance')
    plt.show()

if EXE_Question_List['Q3'] and EXE_Question_List['Q2.1']:
    clf_depth10.SetFeatureInstance()
    PlotFeatureInstance(clf_depth10.feature_instance)


# Question 4
class RandomForest():
    def __init__(self, n_estimators, max_features,
                 boostrap=True, criterion='gini', max_depth=None):
        '''
        n_estimators: number of DecisionTrees
        max_features: # of features to use in DecisionTree
        '''
        self.n_estimators = n_estimators
        self.max_features = int(max_features)
        self.boostrap = boostrap
        # Bool always True?
        self.criterion = criterion
        self.max_depth = max_depth

        self.decision_trees = []

    def fit(self, X, y):
        AllAttr = np.array(X.columns)
        if self.boostrap:
            for i in range(self.n_estimators):
                # Grow N DecisionTrees
                print(f'Grow tree: {i}')
                np.random.shuffle(AllAttr)

                AttrSelect = AllAttr[:self.max_features]
                if DEBUGrandomforest:
                    attrselectlist = [f'{e: <25}' for e in AttrSelect]
                    attrselectlist = '| '.join(attrselectlist)
                    print(f'select Attr: {attrselectlist}')
                RowSelect = np.random.randint(len(X.index), size=len(X.index))
                # may select same row in RowSelect

                # select attrs
                X_select = X[AttrSelect]  # need to modify rows

                # select rows
                X_numpy = X_select.to_numpy()
                y_numpy = y.to_numpy()
                X_numpy = np.array([X_numpy[i] for i in RowSelect])
                y_numpy = np.array([y_numpy[i] for i in RowSelect])

                # to_DataFrame()
                X_select = pd.DataFrame(X_numpy, columns=AttrSelect)
                y_select = pd.DataFrame(y_numpy, columns=['0'])

                DT = DecisionTree(criterion=self.criterion,
                                  max_depth=self.max_depth)
                DT.fit(X_select, y_select)
                if DEBUGrandomforest:
                    print('calc accuracy')
                    print('Accuracy(test) :', accuracy(DT.predict(x_test),
                          np.array(y_test.values).reshape(-1)))

                self.decision_trees += [DT]
        else:
            for i in range(self.n_estimators):
                # Grow N DecisionTrees
                print(f'Grow tree: {i}')
                np.random.shuffle(AllAttr)

                AttrSelect = AllAttr[:self.max_features]
                if DEBUGrandomforest:
                    attrselectlist = [f'{e: <25}' for e in AttrSelect]
                    attrselectlist = '| '.join(attrselectlist)
                    print(f'select Attr: {attrselectlist}')
                # RowSelect =
                #     np.random.randint(len(X.index), size=len(X.index))
                # may select same row in RowSelect

                # select attrs
                X_select = X[AttrSelect]  # need to modify rows

                # select rows
                # X_numpy = X_select.to_numpy()
                # y_numpy = y.to_numpy()
                # X_numpy = np.array([X_numpy[i] for i in RowSelect])
                # y_numpy = np.array([y_numpy[i] for i in RowSelect])

                # to_DataFrame()
                # X_select = pd.DataFrame(X_numpy, columns=AttrSelect)
                # y_select = pd.DataFrame(y_numpy, columns=['0'])
                y_select = y

                DT = DecisionTree(criterion=self.criterion,
                                  max_depth=self.max_depth)
                DT.fit(X_select, y_select)
                if DEBUGrandomforest:
                    print('calc accuracy')
                    print('Accuracy(test) :', accuracy(DT.predict(x_test),
                          np.array(y_test.values).reshape(-1)))

                self.decision_trees += [DT]

    def predict(self, X):
        predictSum = np.zeros(len(X.index))
        for DT in self.decision_trees:
            predictSum += DT.predict(X)

        predictSum /= self.n_estimators

        pred = np.zeros(len(X.index), dtype=int)
        pred[predictSum < 0.5] = 0
        pred[predictSum >= 0.5] = 1

        return pred

# Question 4.1
clf_10tree = RandomForest(
    n_estimators=10,
    max_features=np.sqrt(x_train.shape[1])
)
clf_100tree = RandomForest(
    n_estimators=100,
    max_features=np.sqrt(x_train.shape[1])
)

if EXE_Question_List['Q4.1']:
    clf_10tree.fit(x_train, y_train)
    clf_10tree_pred = clf_10tree.predict(x_test)
    print(f'{"Accuracy clf_10tree:":<25}',
          accuracy(clf_10tree_pred, np.array(y_test.values).reshape(-1)))

    clf_100tree.fit(x_train, y_train)
    clf_100tree_pred = clf_100tree.predict(x_test)
    print(f'{"Accuracy clf_100tree:":<25}',
          accuracy(clf_100tree_pred, np.array(y_test.values).reshape(-1)))

# Question 4.2
clf_random_features = RandomForest(
    n_estimators=10,
    max_features=np.sqrt(x_train.shape[1])
)
clf_all_features = RandomForest(
    n_estimators=10,
    max_features=x_train.shape[1]
)

if EXE_Question_List['Q4.2']:
    clf_random_features.fit(x_train, y_train)
    clf_random_features_pred = clf_random_features.predict(x_test)
    print(f'{"Accuracy clf_random_features:":<25}', accuracy(
          clf_random_features_pred, np.array(y_test.values).reshape(-1)))

    clf_all_features.fit(x_train, y_train)
    clf_all_features_pred = clf_all_features.predict(x_test)
    print(f'{"Accuracy clf_all_features:":<25}',
          accuracy(clf_all_features_pred, np.array(y_test.values).reshape(-1)))

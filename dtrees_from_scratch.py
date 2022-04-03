class DecisionTree():

    def __init__(self, X, y, min_samples_leaf=5, max_depth=6, idxs=None):
        assert max_depth >= 0, 'max_depth must be nonnegative'
        assert min_samples_leaf > 0, 'min_samples_leaf must be positive'
        self.min_samples_leaf, self.max_depth = min_samples_leaf, max_depth
        if isinstance(y, pd.Series): y = y.values
        if idxs is None: idxs = np.arange(len(y))
        self.X, self.y, self.idxs = X, y, idxs
        self.n, self.c = len(idxs), X.shape[1]
        self.value = np.mean(y[idxs]) # node's prediction value
        self.best_score_so_far = float('inf') # initial loss before split finding
        if self.max_depth > 0:
            self._maybe_insert_child_nodes()
            
    def _maybe_insert_child_nodes(self):
        for j in range(self.c): 
            self._find_better_split(j)
        if self.is_leaf: # do not insert children
            return 
        x = self.X.values[self.idxs,self.split_feature_idx]
        left_idx = np.nonzero(x <= self.threshold)[0]
        right_idx = np.nonzero(x > self.threshold)[0]
        self.left = DecisionTree(self.X, self.y, self.min_samples_leaf, 
                                  self.max_depth - 1, self.idxs[left_idx])
        self.right = DecisionTree(self.X, self.y, self.min_samples_leaf, 
                                  self.max_depth - 1, self.idxs[right_idx])
    
    @property
    def is_leaf(self): return self.best_score_so_far == float('inf')
    
    def _find_better_split(self, feature_idx):
        x = self.X.values[self.idxs,feature_idx]
        y = self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y, sort_x = y[sort_idx], x[sort_idx]
        sum_y, n = y.sum(), len(y)
        sum_y_right, n_right = sum_y, n
        sum_y_left, n_left = 0., 0
    
        for i in range(0, self.n - self.min_samples_leaf):
            y_i, x_i, x_i_next = sort_y[i], sort_x[i], sort_x[i + 1]
            sum_y_left += y_i; sum_y_right -= y_i
            n_left += 1; n_right -= 1
            if  n_left < self.min_samples_leaf or x_i == x_i_next:
                continue
            score = - sum_y_left**2 / n_left - sum_y_right**2 / n_right + sum_y**2 / n
            if score < self.best_score_so_far:
                self.best_score_so_far = score
                self.split_feature_idx = feature_idx
                self.threshold = (x_i + x_i_next) / 2
                
    def __repr__(self):
        s = f'n: {self.n}'
        s += f'; value:{self.value:0.2f}'
        if not self.is_leaf:
            split_feature_name = self.X.columns[self.split_feature_idx]
            s += f'; split: {split_feature_name} <= {self.threshold:0.3f}'
        return s
    
    def predict(self, X):
        return np.array([self._predict_row(row) for i, row in X.iterrows()])
    
    def _predict_row(self, row):
        if self.is_leaf: 
            return self.value
        child = self.left if row[self.split_feature_idx] <= self.threshold \
                else self.right
        return child._predict_row(row)
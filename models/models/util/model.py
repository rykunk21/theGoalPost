from .dependencies import *
from .dataManager import DataManager


class Model(RandomForestRegressor):
    """
    This class is a machine learning model that uses 
    random forest regression to make predictions 
    based on input data.
    """
    def __init__(self, dataPath, label=None, options={'n_estimators':100}) -> None:
        """
        Constructor method that initializes the 
        model with a path to the data and a dictionary 
        of options.
        """
        import random
        super().__init__(n_estimators=options['n_estimators'], random_state=random.randint(1,100))
        self.dataManager = DataManager(dataPath, label)
        


    def scatter(self):
        """
        creates a scatter plot of the features and labels of the input data
        """
    
        import seaborn as sns
       

        sns.set(style="ticks")

        df = self.dataManager.loadData()
        num_cols = df.select_dtypes(include=np.number).columns
        num_cols = [col for col in num_cols if col != self.dataManager.label]
        
        ncols = 3
        nrows = np.ceil(len(num_cols) / ncols).astype(int)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 7))
        ax = ax.flatten()
        
        for i, col in enumerate(num_cols):
            sns.scatterplot(x=df[col], y=df[self.dataManager.label], ax=ax[i])
            ax[i].set_xlabel(col)
            ax[i].set_ylabel(self.dataManager.label)
            
        plt.tight_layout()
        plt.show()


    def plotLearningCurves(self) -> None:
        """
        Method that plots the learning curves for the model
        """
        
        train_sizes, train_scores, test_scores = learning_curve(super(), 
                                                self.dataManager.trainFeatures(), self.dataManager.trainLabels(),
                                                cv=5, scoring="neg_mean_squared_error"
                                                )

        train_scores_mean = np.mean(-train_scores, axis=1)
        test_scores_mean = np.mean(-test_scores, axis=1)
        plt.plot(train_sizes, np.sqrt(train_scores_mean), "r-+", linewidth=2, label="train")
        plt.plot(train_sizes, np.sqrt(test_scores_mean), "b-", linewidth=3, label="test")
        plt.legend(loc="upper right", fontsize=14)
        plt.xlabel("Training set size", fontsize=14)
        plt.ylabel("RMSE", fontsize=14)
        plt.title("Learning Curves", fontsize=16)
        plt.show()


    def stdDev(self):
        """
        returns the standard deviation of the differences between 
        the predicted values and the actual values 
        of the test data.
        """
        preds = self.predict(self.dataManager.testFeatures())
        diff = np.array(preds) - np.array(self.dataManager.testLabels())
        return np.std(diff)


    def predict(self, input):
        self.fit(self.dataManager.trainFeatures(), self.dataManager.trainLabels())

        try:
            return super().predict(input)
        except NotFittedError:
            self.fit(self.dataManager.trainFeatures(), self.dataManager.trainLabels())
            return super().predict(input)

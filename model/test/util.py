##
# @file util.py
# @author Ryan Kunkel
#

import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError

import requests
from bs4 import BeautifulSoup


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


class DataManager:
    """
    This class is responsible for preprocessing the 
    input data before it is fed to the machine learning 
    model.
    """
    
    def __init__(self, data, label=None, testSize=0.2, pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy = "median")),
            ('std_scaler', StandardScaler())
        ])) -> None:
        """
        initializes the Data Manager object with a path 
        to the data and an optional data processing 
        pipeline.
        """
        import random
        if isinstance(data, str):
            self.data = pd.read_csv(data, header=0)
        elif isinstance(data, dict):
            self.data = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data

        self.pipeline = pipeline

        if not label is None:
            self.labelCol = label
            self.features = self.data.drop(self.labelCol, axis=1)
            
        else:
            self.features = None

        self.testSize = testSize
        self.randomState = random.randint(1,100)

        

    def numericalize(self):
        """
        Numericalize the features
        """
      
        for col in self.features.columns:
            if self.features[col].dtype == 'object':
                onehot = pd.get_dummies(self.features[col])
                self.features.drop(col, axis=1, inplace=True)
                self.features = self.features.join(onehot)


    def labels(self) -> pd.DataFrame:
        """
        returns the column of the data with the current label
        """
        return self.data()[self.labelCol]


    def featuresScaled(self) -> pd.DataFrame:
        """
        scales the features based on the pipeline
        """

        return self.pipeline.fit_transform(self.features)


    def trainFeatures(self) -> pd.DataFrame:
        """
        returns the features of the training data
        """
        trainFeatures, _, _, _ = train_test_split(
            self.featuresScaled(), self.labels(), test_size=self.testSize, random_state=self.randomState
        )
        return trainFeatures


    def testFeatures(self) -> pd.DataFrame:
        """
        Method that returns the features of the test data.
        """
        _, testFeatures, _, _  = train_test_split(
            self.featuresScaled(), self.labels(), test_size=self.testSize, random_state=self.randomState
        )
        return testFeatures


    def trainLabels(self) -> pd.DataFrame:
        """
        the labels of the training data
        """
        _, _, trainLabels, _ = train_test_split(
            self.featuresScaled(), self.labels(), test_size=self.testSize, random_state=self.randomState
        )
        return trainLabels


    def testLabels(self) -> pd.DataFrame:
        """
        returns the labels of the test data
        """
        _, _, _, testLabels = train_test_split(
            self.featuresScaled(), self.labels(), test_size=self.testSize, random_state=self.randomState
        )
        return testLabels


    def scale(self, frame) -> pd.DataFrame:

        return self.pipeline.fit(frame)



class NCAABModel(Model):

    """
    Model class for NCAAB
    """

    def __init__(self, dataPath, options={'n_estimators':100}) -> None:
        """
        initalizar for NCAAB model
        """
        super().__init__(dataPath,'TmScore', options)
        self.droppedCols = [*'TeamName,W/L,Opp,G,Date,TEAM_FG,TEAM_FGA,TEAM_3P,TEAM_3PA,TEAM_FT,TEAM_FTA,OPP_FG,OPP_FGA,OPP_3P,OPP_3PA,OPP_FT,OPP_FTA'.split(',')]


    def getTeamRows(self, team) -> pd.DataFrame:
        """
        gets all the rows with TeamName == team
        """
        return self.dataManager.data[self.dataManager.data['TeamName'] == team]

    def teamInput(self, team) -> pd.DataFrame:
        """
        Given a team, return the following 
        """
        raw = self.getTeamRows(team).drop(columns=self.droppedCols, axis=1).drop(self.dataManager.labelCol, axis=1)
        for col in raw.columns:
            if raw[col].dtype == 'object':
                onehot = pd.get_dummies(raw[col])
                raw.drop(col, axis=1, inplace=True)
                raw = raw.join(onehot)
                
        return raw.columns

    def predictScore(self, team1, team2):
        pass

    def predict(self, team):
        """
        Make a prediction
        """
        pass



class Scraper:
    """
    Sraper for team object
    Global team mapping?

    """
    def __init__(self, link=None) -> None:
        self.link = link
    

    def getTable(self):

        def parseLine(line):
            for i, entry in enumerate(line[5:]):
                try:
                    data[5 + i] = float(entry)
                except ValueError:
                    return False
            return True

        fp = requests.get(self.link)
        soup = BeautifulSoup(fp.text, 'html.parser')

        table = soup.find(id='div_sgl-basic').table.tbody

        dataTable = []

        for child in table.findChildren('tr'):

            if not child.get('class') is None and 'thead' in child.get('class'):
                continue    

            data = list(map(lambda x: x.text, list(child.findChildren(['th','td']))))
            # col 23 is unnecessary
            data.pop(23)
            data[2] = {'@': '@', 'N':'N', '':'H'}[data[2]]

            if not parseLine(copy.deepcopy(data)):
                continue

            dataTable.append(data)

        return dataTable



class BetUtil:
    import numpy as np

    class BetSlip:

        """
        Class that represents a betslip contiaining the following information
        Team1, Team2
        
        """
        def __init__(self):
            pass

    class Line: 
        def __init__(self, line) -> None:
            if isinstance(line, str):
                sign, value = (line[0], line[1:])
                self.sign = sign
                self.value = int(value)

        def toProb(self):
            if self.sign == '+':
                return 100 / (100 + self.value)
            elif self.sign == '-':
                return self.value / (100 + self.value)

        def __repr__(self):
            return f'{self.sign}{self.value}'

        def __str__(self):
            return self.__repr__(self)

    class Distribution:
        
        def __init__(self, mean=0, std=1) -> None:
            self.mean = mean
            self.std = std

        def probGreaterThan(self, other):
            from scipy.special import erf

            return 1/2* (1 + erf((self.mean - other.mean)/(np.sqrt(2)*np.sqrt(self.std**2+other.std**2))))

    def __init__(self) -> None:
        raise Exception('Constructed a static class')

    
    

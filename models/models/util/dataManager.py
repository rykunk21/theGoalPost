from dependencies import *


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



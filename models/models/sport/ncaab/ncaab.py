from models.util.dependencies import *
from models.util.model import Model

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
        Given a team, return the following input
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


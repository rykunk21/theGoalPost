import unittest
from util import DataManager, NCAABModel
import pandas as pd

def checkDfEqual(df1, df2):
    # Check if the dataframes have the same shape
    if df1.shape != df2.shape:
        print('UNEQUAL SHAPE')
        return False
    
    # Check if the column names are the same
    if not df1.columns.equals(df2.columns):
        print('UNEQUAL COLUMNS')
        return False
    
    # Check if the data in each cell is the same
    if not (df1.values == df2.values).all():
        print("UNEQUAL DATA")
        return False
    
    # If all checks pass, the dataframes are equal
    return True


class TestDataManager(unittest.TestCase):

    def test_init(self):
        DataManager('./scores.csv')

    def test_loadData(self):
        

        data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
            'Age': [25, 30, 35, 40, 45],
            'Gender': ['F', 'M', 'M', 'M', 'F'],
            'Salary': [50000, 60000, 70000, 80000, 90000]
        }

        df = pd.DataFrame(data)
        
        manager = DataManager(df)
        self.assertTrue(checkDfEqual(manager.data.head(), df.head()))

    

    def test_featuresScaled(self):
        pass

    def test_scale(self):
        pass


class TestNcaaModel(unittest.TestCase):

    def test_init(self):
        NCAABModel('./scores.csv')

    def test_teamRows(self):

        model = NCAABModel('./scores.csv')
        teamRows = model.getTeamRows('oregon')
        
        self.assertTrue((teamRows['TeamName'] == 'oregon').all())
        
        
    def test_teamInput(self):
        features = ['OppScore', 'TEAM_FG%', 'TEAM_3P%', 'TEAM_FT%', 'TEAM_ORB', 'TEAM_TRB', 'TEAM_AST', 'TEAM_STL', 'TEAM_BLK', 'TEAM_TOV', 'TEAM_PF', 'OPP_FG%', 'OPP_3P%', 'OPP_FT%', 'OPP_ORB', 'OPP_TRB', 'OPP_AST', 'OPP_STL', 'OPP_BLK', 'OPP_TOV', 'OPP_PF', '@', 'H', 'N']
        
        model = NCAABModel('./scores.csv')

        self.assertTrue((model.teamInput('oregon').array == features).all())

    def test_predictScore(self):
        pass

if __name__ == "__main__":
    unittest.main()
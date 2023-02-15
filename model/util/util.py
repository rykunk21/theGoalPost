##
# @file util.py
# @author Ryan Kunkel
#
import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import requests
from bs4 import BeautifulSoup
import copy


class Preprocessor:
    """
    Preprocessor object
    """
    def __init__(self) -> None:
        pass

class Scraper:
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

    
    

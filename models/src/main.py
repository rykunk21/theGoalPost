from util.dependencies import *


header = [
    'G', 'Date', 'Court', 'Opp', 
    'W/L', 'Tm', 'Opp', 'TEAM_FG', 'TEAM_FGA', 
    'TEAM_FG%', 'TEAM_3P', 'TEAM_3PA', 'TEAM_3P%',
    'TEAM_FT', 'TEAM_FTA', 'TEAM_FT%', 'TEAM_ORB', 
    'TEAM_TRB', 'TEAM_AST', 'TEAM_STL', 'TEAM_BLK',
    'TEAM_TOV', 'TEAM_PF', 'OPP_FG', 
    'OPP_FGA', 'OPP_FG%', 'OPP_3P', 'OPP_3PA',
    'OPP_3P%', 'OPP_FT', 'OPP_FTA', 'OPP_FT%', 
    'OPP_ORB', 'OPP_TRB', 'OPP_AST', 'OPP_STL',
    'OPP_BLK', 'OPP_TOV', 'OPP_PF'
]

def test():
    print('This is a successful test!')

def getRange(id):

    with open('links.txt', 'r') as fp:
        start = 30 * id
        end = start + 30
        links = [link.rstrip() for link in fp.readlines()][start:end]
    
    return links


def getTable(link):

    def parseLine(line):
        for i, entry in enumerate(line[5:]):
            try:
                data[5 + i] = float(entry)
            except ValueError:
                return False
        return True
    
    fp = requests.get(link) 
    soup = BeautifulSoup(fp.text, 'html.parser')

    table = soup.find(id='div_sgl-basic').table.tbody

    dataTable = []
    for child in table.findChildren('tr'):

        if not child.get('class') is None and 'thead' in child.get('class'):
            continue

        data = list(map(lambda x: x.text, list(child.findChildren(['th','td']))))
        data.pop(23)
        data[2] = {'@':'@', 'N':'N', '':'H'}[data[2]]

        if not parseLine(copy.deepcopy(data)):
            continue

        dataTable.append(data)

    return dataTable


def scrape():

    id = int(input('Enter the ID of this machine: '))
    links = getRange(id)

    with open(f'./resources/scores({id}).csv', 'w', newline='') as fp:
        writer = csv.writer(fp)

        for i, link in enumerate(links):
            teamName = link.split('schools/')[1].split('/')[0]
            req = requests.get(link)
            if req.ok:
                print(f'WROTE TEAM {i} TO CSV')
                print('')
                table = getTable(link)
                for row in table:
                    writer.writerow([teamName] + row)
            else:
                print(req.reason)


if __name__ == '__main__':
    scrape()
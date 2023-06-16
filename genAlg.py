import pandas as pd
import numpy as np
import random
import warnings
from matplotlib import pyplot as plt
from itertools import zip_longest

warnings.simplefilter(action='ignore')

def calcFitness(c, df, badDist_list):
    return (df['sitting_dist'] + c*(1 - df[badDist_list].sum(axis=1)/4)) / (1+c)


def swap_oneCoordinate(df, df1, coorList, row1, row2):
    idx = random.randrange(12) #random coordinate
    df1.iloc[row1][coorList[idx]], df1.iloc[row2][coorList[idx]] = df.iloc[row2][coorList[idx]].copy(), df.iloc[row1][coorList[idx]].copy()
    return df1


def swap_XYZ(df, df1, axes, row1, row2):
    idx = axes[random.randrange(3)]
    tmpList = []
    for i in range(1,5):
        tmpList.append(f'{idx}{i}')
    df1.iloc[row1][tmpList], df1.iloc[row2][tmpList] = df.iloc[row2][tmpList].copy(), df.iloc[row1][tmpList].copy()
    return df1


def swap_1234(df, df1, axes, row1, row2):
    id = random.randrange(4)+1
    tmpList = []
    for i in axes:
        tmpList.append(f'{i}{id}')
    df1.iloc[row1][tmpList], df1.iloc[row2][tmpList] = df.iloc[row2][tmpList].copy(), df.iloc[row1][tmpList].copy()
    return df1


def column_wise_sum(rows):
    columns = zip_longest(*rows, fillvalue=0)
    return [sum(col)/len(rows) for col in columns]


def avg(lst):
    return sum(lst) / len(lst)


csvDf = pd.read_csv('dataset-HAR-PUC-Rio.csv', sep=';')

#metrics
population = 20
c = 0.1
pc = 0.6 
pm = 0.01 

axes = ['x', 'y', 'z']
nums = list(range(1,5))
coord = []

populationDf = pd.DataFrame()

for i in axes:
    for j in nums:
        coord.append(f'{i}{j}')

#normalize
for i in coord:
    csvDf[i] = (csvDf[i]-csvDf[i].min()) / (csvDf[i].max()-csvDf[i].min())#"""

meanOfOtherPositionsDf = csvDf.loc[csvDf['class'] != 'sitting'].groupby('class')[coord].mean().reset_index()
csvDf = csvDf.loc[csvDf['class'] == 'sitting']
csvDf = csvDf[coord]

bestSittingDf = csvDf.mean()
bestSittingDf = bestSittingDf.values.flatten().tolist()
bestSittingDf = [ '%.6f' % elem for elem in bestSittingDf ]
bestSittingDf.insert(0, 'sitting')

repsBest = []
repsGenerations = []

for repetitions in range(10):
    for i in coord:
        populationDf[i] = np.random.uniform(csvDf[i].min()/10000, csvDf[i].max()*10000, size=population)

    meanOfOtherPositionsDf.loc[4] = bestSittingDf

    meanOfOtherPositionsDf[coord] = meanOfOtherPositionsDf[coord].astype('float64')

    badDist_list = []
    shouldContinue = True
    allAvgsFitnesses = []
    allBestFitnesses = []
    numOfNotImprovedBest = 0
    prevBestFitness = 0.4

    for generation in range(1001):

        #calculate cos() distance from each ideal stance
        for i in meanOfOtherPositionsDf['class'].unique():
            if generation == 0:
                if i != 'sitting':
                    badDist_list.append(f'{i}_dist')
            A = populationDf.loc[:, coord[0]: coord[-1]]
            B = meanOfOtherPositionsDf.loc[meanOfOtherPositionsDf['class'] == i].squeeze().drop(['class'], axis=0)

            populationDf[f'{i}_dist'] = (A * B).sum(axis=1) / (((A**2).sum(axis=1) * (B**2).sum()) ** 0.5)

        populationDf['fitness'] = calcFitness(c, populationDf, badDist_list)

        populationDf = populationDf.drop(badDist_list+['sitting_dist'], axis=1).sort_values('fitness', ascending=True).reset_index(drop=True)
        totalSuitabiliity = populationDf['fitness'].sum()
        populationDf['p'] = (populationDf['fitness'] / totalSuitabiliity).cumsum()

        #termination
        allAvgsFitnesses.append(populationDf['fitness'].mean())
        allBestFitnesses.append(populationDf['fitness'].max())

        percentImprovementOfBest = (populationDf.iloc[-1]['fitness'] - prevBestFitness) / prevBestFitness

        if percentImprovementOfBest < 0.1:
            numOfNotImprovedBest += 1
        else:
            numOfNotImprovedBest = 0
        
        if numOfNotImprovedBest == 3:
            repsGenerations.append(generation)
            print(f'Rep: {repetitions}. Only small improvements for best individual. Generation: {generation}')
            break

        if generation == 1000:
            repsGenerations.append(generation)
            print(f'Rep: {repetitions}. Last generation reached')

        tmpDf = pd.DataFrame()


        #selection
        for rot in range(population):
            sel = random.random()
            tmpDf = pd.concat([tmpDf, populationDf.loc[populationDf['p'] > sel].head(1).astype('float64')])
        tmpDf = tmpDf.sort_values('fitness')

        populationDf = tmpDf.copy().drop(['p', 'fitness'], axis=1) #populationDf now contains only selected individuals


        #breeding
        populationDf['r_breed'] = np.random.uniform(0, 1, size=population)
        populationDf = populationDf.reset_index()
        populationDf['r_breed'].iloc[-1] = 1
        toBreedDf = populationDf.loc[populationDf['r_breed'] < pc].drop(['r_breed'], axis = 1)
        #print(populationDf)
        populationDf = populationDf.drop(populationDf.loc[populationDf['r_breed'] <= pc].index)

        bredDf = toBreedDf.copy()

        for i in range(0, toBreedDf.shape[0]):
            try:
                bredDf = swap_XYZ(toBreedDf, axes, i, random.randrange(toBreedDf.shape[0]-1))
            except: pass

        if toBreedDf.shape[0] % 2: #if number of rows is odd then breed the last one with a random one from the breeding list
            bredDf = swap_XYZ(toBreedDf, bredDf, axes, toBreedDf.shape[0]-1, random.randrange(toBreedDf.shape[0]-2))


        populationDf = populationDf.drop(['r_breed'], axis=1).reset_index(drop = True)

        populationDf = pd.concat([populationDf, bredDf]).sort_index().reset_index(drop = True).drop(['index'], axis=1)

        populationDf.iloc[0: bredDf.shape[0]] = bredDf

        #mutations
        populationDf['r_mutate'] = np.random.uniform(0, 1, size=population)
        populationDf['r_mutate'].iloc[-populationDf.shape[0] // 10:] = 1 #last 10% of individuals will not be mutated. That's called elitism
        randCol = coord[random.randrange(12)]

        x1 = 0.1
        x2 = 0.3
        if random.randint(0, 1) == 0:
            x1, x2 = -x1, -x2
        populationDf.loc[populationDf['r_mutate'] < pm, randCol] = populationDf.loc[populationDf['r_mutate'] < pm][randCol] + \
        np.random.uniform(x1, x2, size=populationDf.loc[populationDf['r_mutate'] < pm].shape[0]) * populationDf.loc[populationDf['r_mutate'] < pm][randCol]
        
        populationDf = populationDf.drop(['r_mutate'], axis=1)

    repsBest.append(allBestFitnesses)

avgOfBest = column_wise_sum(repsBest)
print(avgOfBest)
print(avg(repsGenerations))

plt.plot(avgOfBest, label = 'best')
plt.show()
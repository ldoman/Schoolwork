"""
Recorded stats for 1 million runs each:
Method 1: cw% = 33.3089, rob% = 33.34, ties% = 33.3511
Method 2: cw% = 33.3499, rob% = 33.3195, ties% = 33.3306
Method 3: cw% = 33.3593, rob% = 33.3025, ties% = 33.3382
"""

import random as rn

#values of rock, paper, scissors
r,p,s = 0,1,2
#dictionary e.g., rock beats scissors
ws = {r:s, p:r, s:p}
lose_scenario = [1, 2, 0]
nogames = int(input("Number of games? "))

totgames = 0
compwins = 0
robwins = 0
ties = 0

gamehistory = []
counts = {0:0, 1:0, 2:0}

def max_count():
    return max(counts, key = counts.get)
    
while totgames < nogames:
    if not gamehistory:
        print("rand")
        rob = rn.randrange(0,3,1)
    else:
        rob = lose_scenario[gamehistory[-1][1]] # Method 1: Change move to winning move of comp's last move 
        #rob = lose_scenario[gamehistory[0][1]] # Method 2: Change move to winning move of comp's first move 
        #rob = lose_scenario[max_count()] # Method 3: Change move to winning move of comp's most used move
    comp = rn.randrange(0,3,1)
    counts[comp] = counts[comp] + 1
    gamehistory.append([rob, comp])

    print("Rob: {0}, Comp: {1}".format(rob, comp))

    if ws[comp] == rob:
       compwins += 1
    elif ws[rob] == comp:
       robwins += 1
    else:
       ties += 1
    totgames += 1

v = list(map(lambda x: 100*x/totgames, [compwins, robwins, ties]))
print("Stats\ncw% = {0}, rob% = {1}, ties% = {2}".format(*v))
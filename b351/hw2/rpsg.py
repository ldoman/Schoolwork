"""
Recorded stats for 100,000 runs each:
Rob bets: min(rob_pot, comp_pot, 10) cw% = 49.851, rob% = 50.149
Rob bets: min(rob_pot, comp_pot, 15) cw% = 49.879, rob% = 50.121
Rob bets: min(rob_pot, comp_pot, 20) cw% = 49.85, rob% = 50.15
Rob bets: min(rob_pot, comp_pot, 25) cw% = 50.014, rob% = 49.986

1 Million runs:
Rob bets: min(rob_pot, comp_pot, 10) cw% = 49.9995, rob% = 50.0005
"""

import random as rn

# Get max used move - for rob strategy method 3
def max_count():
    return max(counts, key = counts.get)

# Check win conditions
def check_win(comp_pot, rob_pot):
    ret = False
    if comp_pot == 0 and rob_pot == 0: # TODO: Is this even reachable?
        print("Tie")
        ret = True
    if comp_pot == 0:
        print("Comp wins")
        ret = True
    if rob_pot == 0:
        print("Rob wins")
        ret = True
    
    return ret

def play():
    # Values of rock, paper, scissors
    r,p,s = 0,1,2

    # Win/Lose scenarios
    ws = {r:s, p:r, s:p}
    lose_scenario = [1, 2, 0]

    totgames = 0
    compwins = 0
    robwins = 0
    ties = 0

    # Define pot amounts
    comp_pot = 100
    rob_pot = 100

    gamehistory = []
    bet_history = []
    counts = {0:0, 1:0, 2:0}
    
    # Loop until empty pot
    while True:
        if check_win(comp_pot, rob_pot):
            break

        # Rob decision
        if not gamehistory:
            rob = rn.randrange(0,3,1)
        else:
            rob = lose_scenario[gamehistory[-1][1]] # Method 1: Change move to winning move of comp's last move 
            #rob = lose_scenario[gamehistory[0][1]] # Method 2: Change move to winning move of comp's first move 
            #rob = lose_scenario[max_count()] # Method 3: Change move to winning move of comp's most used move
        
        # Comp decision
        comp = rn.randrange(0,3,1)

        # Take turns making the initial bet and always call opponents bet
        if len(gamehistory) % 2 == 0:
            if comp_pot == 1:
                comp_bet = 1
            else:
                comp_bet = min(rob_pot, rn.randrange(1,comp_pot,1))
            rob_bet = comp_bet
        else:
            rob_bet = min(rob_pot, comp_pot, 10) # Rob is not a very smart gambler
            comp_bet = rob_bet
        
        # Record history
        gamehistory.append([rob, comp])
        print("Rob: {0}, Comp: {1}".format(rob, comp))

        # Determine winner and update pots
        if ws[comp] == rob:
           compwins += 1
           rob_pot = rob_pot - rob_bet
           comp_pot = comp_pot + comp_bet
        elif ws[rob] == comp:
           robwins += 1
           rob_pot = rob_pot + rob_bet
           comp_pot = comp_pot - comp_bet
        else:
           ties += 1
        totgames += 1

    v = list(map(lambda x: 100*x/totgames, [compwins, robwins, ties]))
    print("Stats\ncw% = {0}, rob% = {1}, ties% = {2}".format(*v))

    return 0 if comp_pot == 0 else 1 # Return 0 if Rob wins else 1

if __name__ == '__main__':
    # Rob wins @ 0. Comp wins @ 1
    stats = [0,0]
    games = 1000000
    
    for i in range(0, games):
        winner = play()
        stats[winner] = stats[winner] + 1
    
    v = list(map(lambda x: 100*x/games, [stats[1],stats[0]]))
    print("Stats\ncw% = {0}, rob% = {1}".format(*v))
    
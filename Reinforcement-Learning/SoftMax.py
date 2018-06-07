# bandit-starter.py
# roncale-olivier/25-mar-2017
#
# The skeleton of program to run experiments with softmax action selection for n-armed
# bandits.
#
# This is an implementation of the bandit problem discussed by Sutton and Barto. The
# problem is stated here:
#
# https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node16.html
#
# The Parameters, Class definition, and Plotting sections are NOT my code.
# These belong to Dr. Simon Parsons at kcl. All main functions embodying the loops of the 
# bandit learner are mine. This was an exercise in coding to solidify understanding of the
# typical bandit learner algorithm. All code has been annotated extensively, regardless of origin,
# in order to facilitate understanding. 

# This is the soft-max implementation of bandit learning.


import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Parameters
# =============================================================================

epsilon     = 0.1  # learning parameter
num_actions = 10   # number of arms on the bandit
iterations  = 20 # number of times we repeat the experiment
plays       = 1000 # number of plays within each experiment


# =============================================================================
# # Class definition
# =============================================================================

# Each class 
class bandit_problem:

    # Initialization
    def __init__ (self):
        # Create the actual reward for each of the actions. This is a draw from
        # a normal distribution with mean 0 and variance (and standard deviation) 1.
        self.q_star = []
        for k in range(num_actions):
            self.q_star.append(np.random.randn())

    # When an action is selected, provide the payoff for the action.
    #
    # This is a draw from a normal distribution with mean q_star and
    # variance 1. That is the same as q_star plus a draw from a normal
    # distribution with mean 0 and variance 1
    def execute_action(self, action):
        return self.q_star[pick] + np.random.randn()
    
    
# =============================================================================
# Functions
# =============================================================================
        

# Soft max function     
def softmax(x, tau):
#    Compute softmax values for each sets of scores in x.
#    Where X is the list of rewards.

    y = []
    prob = []
    
    for i in x:
        y.append(np.exp(float(i)/float(tau)))
    
    total = np.sum(y)
    
    for i in y:
        prob.append(i/total)
        
    return prob 




# Setup lists to collect metric data
average_rewards = []
proportions_optimal_action = []


## Main loop --- repeated for each iteration (i.e., each game). Each game is specified by 
# a number of actions (inner loop)
for i in range(iterations):

    # Create a new bandit with num_actions arms (see parameters)
    bp = bandit_problem()


    ## Inner loop --- repeated for each play in each iteration
    
    # Variable to collect data on each play. We reinitialise for each
    # iteration.
    rewards = []
    actions = []
    # Will keep track of average rewards
    avg_reward = np.zeros(num_actions)
    # Will keep track of number of times an action has been played
    num_actions_played = np.zeros(num_actions)
    
    # Will allow us to initialize the inner loop if we do not have an action yet
    # i.e., we have not played yet.
    play = False
    
    # Calculating proportionality of games to be played. Used for altering tau.
    # Linear decrease from 1 to 0.1. This helps ensure better convergence.
    tau_values = np.linspace(1, 0.1, plays)
    
    
    for j in range(plays):
        
        # Initializing tau based on number of games played. 
        tau = tau_values[j]
        
        ####################################################
        
        # Softmax computation
        probabilities = softmax(avg_reward, tau)
        
        # Pick choice depending on probabilities as calculated by softmax
        pick = int(np.random.choice(range(num_actions), 1, p = probabilities))
        
        ####################################################
        
        # Make the action and reap the reward:
        reward_from_action = bp.execute_action(pick)
        
        # Remember what action we took and what reward we got
        actions.append(pick)
        rewards.append(reward_from_action)
    
        # Update action counter
        num_actions_played[pick] += 1
        
    
        # Calculate the average reward incrementally. First check if an existing average is present.
        # If present, use incremental update mean formula
        if num_actions_played[pick] != 1:
            Qk = avg_reward[pick]
            k = num_actions_played[pick]
            r = reward_from_action
            
            new_mean = Qk + ((r - Qk) / k)
            # Note k is (k+1) as the number of actions was already updated prior to the calculation.
            
            avg_reward[pick] = new_mean # Updating the mean
            
        # Else set that value as the average
        else: 
            avg_reward[pick] = reward_from_action
        
        print pick # Examine the printout as the loop runs. You should see convergence soon. 
            
    
    
        # Pick the best reward
        pick = avg_reward.argmax()


    # End of inner loop
    
    # Compute metrics over a run (game).
    counter = []
    average_reward = []
    proportion_optimal_action = []

    # Find the true optimal action
    optimal_action = bp.q_star.index(max(bp.q_star))
    
    
    for j in range(plays):
        counter.append(j)
        total_reward = 0
        optimal_action_count = 0
        for i in range(j):
            total_reward += rewards[i] # Total up the rewards
            if actions[i] == optimal_action: # Calculate the proportion of optimal action selected
                optimal_action_count += 1
        
        # Calculates the average reward for each step of a play in a particular game. Len = max 1000             
        average_reward.append(total_reward/(j+1))
        proportion_optimal_action.append(float(optimal_action_count)/(j+1)) # Avg optimal action selected

    # Stash metrics for later analysis
    average_rewards.append(average_reward)
    proportions_optimal_action.append(proportion_optimal_action)


# End of main loop


# =============================================================================
# Averaging Results
# =============================================================================

# This computes the averages of rewards per steps and optimal action proportion over
# all x games. 
averaged_reward = []
averaged_proportion = []
for i in range(plays): # For each game...
    total_reward_per_step = 0
    total_proportion_per_step = 0
    for j in range(iterations):
        total_reward_per_step += average_rewards[j][i]
        total_proportion_per_step += proportions_optimal_action[j][i]
        
    averaged_reward.append(total_reward_per_step/iterations)
    averaged_proportion.append(float(total_proportion_per_step)/iterations)


# Make proportion a percentage
for i in range(len(averaged_proportion)):
    averaged_proportion[i] = averaged_proportion[i]*100
    


# =============================================================================
# Plotting
# =============================================================================
    
# Plot parameters are optimised for figures that are used in slides.
plt.subplot(1, 2, 1)
plt.title("Epsilon = " + str(epsilon), fontsize=20)
plt.plot(counter, averaged_reward, color = 'blue', linewidth = 4)
plt.xlabel("Plays", fontsize=20 )
plt.ylabel("Average reward", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0, 1.5)
plt.xlim(0, plays)

plt.subplot(1, 2, 2)
plt.title("Epsilon = " + str(epsilon), fontsize=20)
plt.plot(counter, averaged_proportion, color = 'green', linewidth=4)
plt.xlabel("Plays", fontsize=20 )
plt.ylabel("Percent optimal action", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0, 100)
plt.xlim(0, plays)

plt.tight_layout()
plt.show()

# Note: Change the parameters in the parameter section to view different results. 

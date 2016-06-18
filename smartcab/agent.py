import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
from collections import defaultdict

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""


    def __init__(self, env, epsilon=1, alpha=1, gamma=0.5, qTable = defaultdict(int)):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.qTable = qTable
        self.epoch = 1
        self.total_reward = 0
        self.explored_states = 0
       
    def max_action(self, state):
        max_reward = -float("inf")
        max_action = None
        possible_actions = ['forward', 'left', 'right', None]
        for action in possible_actions:
            reward = self.qTable[(state, action)]
            #Find maximum
            if reward > max_reward:
                max_reward = reward
                max_action = action
        return max_action

    def max_reward(self, state):
        max_reward = -float("inf")
        possible_actions = ['forward', 'left', 'right', None]
        for action in possible_actions:
            reward = self.qTable[(state, action)]
            #Find maximum
            if reward > max_reward:
                max_reward = reward
        return max_reward

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.epoch += 1
        self.epsilon = 1 / float(self.epoch)
        self.alpha = 1 / float(self.epoch)
        #print "Epsilon: {}".format(self.epsilon)
        if self.epoch == 100:
            #table = pd.Series(self.qTable, index=self.qTable.keys())
            total_reward_list.append(self.total_reward)
            destination_reached_list.append(self.env.get_dest_reached())
            explored_states.append(self.explored_states)
            #failed_trial.append(self.env.get_trial_failed())
            #table["total_reward"] = self.total_reward
            #table["destination_reached"] = self.env.get_dest_reached()
            #table.to_csv("qtable_gamma_0.5_run_"+str(i)+".csv")

    def update(self, t):
        #print "epsilon: {}".format(self.epsilon)
        #print "alpha: {}".format(self.alpha)
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        #print "next_waypoint: {}".format(self.next_waypoint)
        inputs = self.env.sense(self)
        
        # decreasing gamma as a function of deadline
        deadline = self.env.get_deadline(self)
        #print "deadline: {}".format(deadline)
        #print "t: {}".format(t)
        if t == 0:
            self.deadline_start = deadline
        #print "decreased_gamma: {}".format((float(deadline)/self.deadline_start)*self.gamma)

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        state_i = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        #print "state_i: {}".format(state_i)
        
        # TODO: Select action according to your policy
        if (random.random() < self.epsilon) and self.epoch < 50: # epsilon-greedy method
            action = random.choice([None, 'forward', 'left', 'right'])
            self.explored_states += 1
            #print "random_action: {}".format(action)
        else: #choose best action from Q(s,a) values
            action = self.max_action(state_i)
            #print "chosen_action: {}".format(action)
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward
        #print "reward: {}".format(reward)

        # Sense again and set state
        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        state_ii = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        #print "state_ii: {}".format(state_ii)

        # TODO: Learn policy based on state, action, reward        
        self.qTable[state_i, action] = (1 - self.alpha)*self.qTable[state_i, action] + self.alpha*(reward + (float(deadline)/self.deadline_start)*self.gamma*self.max_reward(state_ii))
        #self.qTable[state_i, action] = self.qTable[state_i, action] + self.alpha * (reward)
        #print "qTable: {}".format(self.qTable)
        #print "LearningAgent.update(): qTable = {}, deadline = {}, inputs = {}, action = {}, reward = {}, epsilon = {}, t = {}".format(self.qTable, deadline, inputs, action, reward, self.epsilon, t)  # [debug]
    
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track    

    # Now simulate it
    sim = Simulator(e, update_delay=None)  # reduce update_delay to speed up simulation
    sim.run(n_trials=101)  # press Esc or close pygame window to quit
    

if __name__ == '__main__':
    total_reward_list = []
    destination_reached_list = []
    #failed_trial = []
    explored_states = []
    
    for i in range(1):
        run()
        
    data = pd.DataFrame({'Mean Reward' : total_reward_list, 
                         'Pass Rate' : destination_reached_list,
                         'Learning Rate' : 1,
                         'Discout Factor' : 0.5,
                         'Epsilon' : 1,
                         #'Failed Trial' : failed_trial,
                         'Explores States' : explored_states})
    data.to_csv("data_alpha_0.5_gamma_0.5_epsilon_0.5.csv", index=False)

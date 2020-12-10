## Reinforcement Learning Using Q-Learning in a Predator/Prey Agent-Based Model with NetLogo and PyTorch 

Agent-based modeling is way of designing computational models from the perspective of individual autonomous agents to study how the behavior of systems as a whole are determined by the interactions between agents and other elements of the system. [NetLogo](https://ccl.northwestern.edu/netlogo/) is a programming language and modeling environment created and maintained by the [CCL Lab](https://ccl.northwestern.edu/) at [Northwestern University](http://www.northwestern.edu/). Predator/Prey systems are an ecological phenomenon that can be easily modeled using agent-based modeling (ABM.) Typically in these models, the behaviors of the predator and prey agents are determined by fairly simple heuristics (often time pure randomness.) The aim of this project is to embedded reinforcement learning models into both the predator agents and the prey agents in a predator/prey ABM in NetLogo. NetLogo includes a Python extension which allows us to interact with external Python code from within a NetLogo model-and more importantly to use machine learning libraries such as PyTorch. We have created a few different NetLogo models and a Q-learning model in PyTorch that we use in this project to conduct our experiments.
 
### NetLogo Models

We used a single agent ABM to calibrate our reinforcement learning model. A single mouse agent is placed in an environment with a fixed number of randomly placed fruit and posion vials. The agent gets a positive reward for landing on fruit and a negative reward for landing on poison. In each time step the agent can choose to turn right or left by 20 degrees or move forward.

The following is a demonstration, but just with a manually programmed agent heuristic.

__Press the _setup_ and then the _go_ button.__
 
<div style="overflow: hidden; width: 100%; height: 700px"><iframe src="dummy-model-demo.html" width="100%" height="100%" style="border:none; margin-top:-150px;" scrolling="no">
</iframe></div>

The second model we used involved multiple predator and prey agents. The hawk agents and the mice agents can both perform only 3 actions as in the prior model: turn right or left by 20 degrees and move forward. The hawks recieve a positive reward when they get within a certain distance of a mouse. Mice recieve a negative reward when they get too close to a hawk and then are moved to a new random location in the world.

Below is a demonstration that in which mice and hawks act purely randomly.

__Press the _setup_ and then the _go_ button.__

<div style="overflow: hidden; width: 100%; height: 700px"><iframe src="pred-prey-dummy-demo.html" width="100%" height="100%" style="border:none; margin-top: -150px;" scrolling="no">
</iframe></div>

### Reinforcement Learning Models

To embed reinforcement learning into the agents in our ABMs we designed an Agent class in Python that we access from NetLogo with the NetLogo Python Extension. In fact, we developed a number of different Agent classes to experiment with differen styles of reinformcement learning. However, all of these variations centered around one method of reinforcement learning: Q-learning.

#### Q-Learning

In Q-learning we try to find a state-action value function for an agent. The state-action value function gives us a value for performing each action in the set of possible actions available to the agent given a state. The agent's policy can then be determined by chosing an action in a given state based on the values of the state-action function. We can do this deterministically (always chosing the maximum valued action) or probabilitistically (choose an action with a probability proportional to its value.) In deep q-learning, we train a deep net (ANN) to appoximate the state-action value function. We do this by passing reward signals from the agent's environment into the deep learning model. 




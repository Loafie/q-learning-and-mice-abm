## Reinforcement Learning Using Q-Learning in a Predator/Prey Agent-Based Model with NetLogo and PyTorch 

Agent-based modeling is way of designing computational models from the perspective of individual autonomous agents to study how the behavior of systems as a whole are determined by the interactions between agents and other elements of the system. [NetLogo](https://ccl.northwestern.edu/netlogo/) is a programming language and modeling environment created and maintained by the [CCL Lab](https://ccl.northwestern.edu/) at [Northwestern University](http://www.northwestern.edu/). Predator/Prey systems are an ecological phenomenon that can be easily modeled using agent-based modeling (ABM.) Typically in these models, the behaviors of the predator and prey agents are determined by fairly simple heuristics (often time pure randomness.) The aim of this project is to embedded reinforcement learning models into both the predator agents and the prey agents in a predator/prey ABM in NetLogo. NetLogo includes a Python extension which allows us to interact with external Python code from within a NetLogo model-and more importantly to use machine learning libraries such as PyTorch. We have created a few different NetLogo models and a Q-learning model in PyTorch that we use in this project to conduct our experiments.
 
### NetLogo Models

We used a single agent ABM to calibrate our reinforcement learning model. A single mouse agent is placed in an environment with a fixed number of randomly placed fruit and posion vials. The agent gets a positive reward for landing on fruit and a negative reward for landing on poison. In each time step the agent can choose to turn right or left by 20 degrees or move forward.

The following is a demonstration, but just with a manually programmed agent heuristic.

__Press the _setup_ and then the _go_ button.__
 
<div style="width: 900px; height: 600px; overflow: hidden"><iframe src="dummy-model-demo.html" width="100%" height="100%" style="border:none; margin-top:-150px;" scrolling="no">
</iframe></div>

The second model we used involved multiple predator and prey agents. The hawk agents and the mice agents can both perform only 3 actions as in the prior model: turn right or left by 20 degrees and move forward. The hawks recieve a positive reward when they get within a certain distance of a mouse. Mice recieve a negative reward when they get too close to a hawk and then are moved to a new random location in the world.

Below is a demonstration that in which mice and hawks act purely randomly.

__Press the _setup_ and then the _go_ button.__

<div style="width: 900px; height: 600px; overflow: hidden"><iframe src="pred-prey-dummy-demo.html" width="100%" height="100%" style="border:none; margin-top: -150px;" scrolling="no">
</iframe></div>

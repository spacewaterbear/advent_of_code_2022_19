# AdventofCode 2022, 19, using RL

This is a solution to the 19th day of Advent of Code 2022, using the reinforcement learning approach.
Here are the problem explanations:

_The wind has changed direction enough to stop sending lava droplets toward you, so you and the elephants exit the cave. As you do, you notice a collection of geodes around the pond. Perhaps you could use the obsidian to create some geode-cracking robots and break them open?
To collect the obsidian from the bottom of the pond, you'll need waterproof obsidian-collecting robots. Fortunately, there is an abundant amount of clay nearby that you can use to make them waterproof.
In order to harvest the clay, you'll need special-purpose clay-collecting robots. To make any type of robot, you'll need ore, which is also plentiful but in the opposite direction from the clay.
Collecting ore requires ore-collecting robots with big drills. Fortunately, you have exactly one ore-collecting robot in your pack that you can use to kickstart the whole operation.
Each robot can collect 1 of its resource type per minute. It also takes one minute for the robot factory (also conveniently from your pack) to construct any type of robot, although it consumes the necessary resources available when construction begins.
The robot factory has many blueprints (your puzzle input) you can choose from, but once you've configured it with a blueprint, you can't change it. You'll need to work out which blueprint is best._


Please check out the [Advent of Code 2022, day 19](https://adventofcode.com/2022/day/19) website for the full enonciation of the problem.
## Q-learning

### Environment

The env is store in a dictionary with key as state and value as actions.

- the state is a tuple of 8 digits, representing : 
  - the 4 first digits are the quantity of material : ore, clay, obsidian, geode
  - the 4 last digits are the quantity of robots that collect the material : ore_robot, clay_robot, obsidian_robot, geode_robot

- the action is a tuple of 5 digits, representing the action to do nothing or to build a robots :
  - do_nothing, ore_robot, clay_robot, obsidian_robot, geode_robot (see `config.action_mapping`)

### Training & Evaluation informations
- The evaluation metrics are saved in neptune.ai
- You can search for hyperparametization

## Result

- Q Learning : Even with 10 millions episodes, the agent is not able to find the optimal solution :
    - for the first example, it succeed at best to collect 8 geodes instead of 9
    - for the second example, it succeed at best to collect 10 geodes instead of 12

## Setup

- install the requirements : `pip install -r requirements.txt`
- rename template.env to .env and fill the variables

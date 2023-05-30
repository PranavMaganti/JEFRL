# js-rl

Enhancing Javascript engine fuzzing with the use of Reinforcement Learning

To run the code, you need to first install [poetry](https://python-poetry.org/) and then run the following commands which setup a virtual environment and install all the dependencies:

```bash
poetry shell
poetry install
python3 src/main.py
```

## Project Timeline

-   [x] Implement a way to run a JavaScript program through an engine from Python and receive outputs such as status, coverage, etc (Target: Late January)
-   [x] Design and implement a way to take a JavaScript Abstract Syntax Tree (AST) along with a mutation as input and return a mutated AST (Target: Mid-Late February)
-   [x] Implement Environment which can take an action from the RL agent, modify the current JavaScript AST based on this action, run the program through the engine, and return the reward along with the new state. (Target: Mid-Late March)
-   [x] Implement a DQN agent which takes a JavaScript AST as input and then outputs the desired action to be taken (Target: Early April)
-   [x] Make mutations smarter by taking into account the variable/functions in scope (Target: Mid April)
-   [x] Add more unit tests for each module due to errors arising during execution (Target: Late April)
-   [x] Come up with a comprehensive evaluation metric for the fuzzers to allow for easy comparision, eg. coverage in a certain time period of execution or number of crashes (Target: Early May)
-   [x] Implement a transformer model using similar technique to montage to tokenize/embed the AST directly instead of converting to code (Target: Late May)
-   [ ] Execute fuzzer on several versions of JavaScript engines and use the evaluation metric to tune reward function and hyperparameters (Target: Early June)
-   [ ] Compare results of RL agent with a random agent and other fuzzers in terms of bugs found in older versions and coverage gained per time period (Target: Early June)
-   [ ] Write a report (Target: Mid June)

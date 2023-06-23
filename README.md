# JEFRL: JavaScript Engine Fuzzing using Reinforcement Learning

## Prerequisites

Before running the code you will need to build the JavaScript engine by running the following commands (docker required):

```bash
cd engines/v8
docker build -t v8-build .
docker run -it -v .:/v8 v8-build /v8/build.sh
sudo chown -R $USER:$USER . # to change ownership of the files created by docker back to the user
```

Additionally, to run the code, you need to first need a working version of python3.11 on your machine and you will then need to install [poetry](https://python-poetry.org/).  We use poetry to initialise all our Python dependencies and virtual environment. To setup poetry run the following in the root of our project :

```
poetry shell
poetry install
```

## Running the fuzzer

As mentioned in the [Report](https://github.com/vanpra/js-rl/blob/main/write-up.pdf), JEFRL uses a three stage training process which includes: pre-training of the AST Transformer, fine-tuning of the AST Transformer, and finally, running the fuzzer. In addition to this we also have a pre-training step to process all the data required for all of these steps. 

### Stage 0: Pre-processing

For our training we made use of [DIE-corpus](https://github.com/sslab-gatech/DIE-corpus), which includes seed inputs from various different JavaScript engine test suites and also past vulnerability PoCs. First start by cloning the DIE-corpus into the folder `corpus` as `DIE`:

```bash
git clone git@github.com:sslab-gatech/DIE-corpus.git corpus/DIE
```

After this we can run our pre-processing stage using the following command:

```
python3 src/preprocessing.py
```

This will save all the necessary file to the directory `data`

### Stage 1: Pre-training AST Transformer

To run pre-training a machine with a powerful GPU is required. We used an NVIDIA A30 to pre-train and fine-tune our transformer. To run the pre-training use the following command:

```
python3 src/pretraining.py
```

This will create new folder in `out/finetuning` with the name being the timestamp at which you started training. The name of this folder is required for the next stage of training.

### Stage 2: Fine-tuning AST Transformer 

We can run the fine-tuning step by using the following command:

```
python3 src/finetuning.py --pretraining-path <path-to-pretraing-out-folder> --pretraing-step <step-of-the-pretraining-to-use>
```

This will create new folder in `out/finetuning` with the name being the timestamp at which you started fine-tuning. The name of this folder is required for the next stage.

### Stage 3: Running the fuzzer

For this step we no longer require a powerful GPU as our transformer is now frozen. We can run the fuzzer using the following command:
```
python3 src/main.py --finetuning-path <path-to-finetuning-out-folder> --finetuning-step <step-of-the-finetuning-to-use>
```

The final model along with run results for every stage are store in `out/rl-training`

## Project Timeline

-   [x] Implement a way to run a JavaScript program through an engine from Python and receive outputs such as status, coverage, etc (Target: Late January)
-   [x] Design and implement a way to take a JavaScript Abstract Syntax Tree (AST) along with a mutation as input and return a mutated AST (Target: Mid-Late February)
-   [x] Implement Environment which can take an action from the RL agent, modify the current JavaScript AST based on this action, run the program through the engine, and return the reward along with the new state. (Target: Mid-Late March)
-   [x] Implement a DQN agent which takes a JavaScript AST as input and then outputs the desired action to be taken (Target: Early April)
-   [x] Make mutations smarter by taking into account the variable/functions in scope (Target: Mid April)
-   [x] Add more unit tests for each module due to errors arising during execution (Target: Late April)
-   [x] Come up with a comprehensive evaluation metric for the fuzzers to allow for easy comparision, eg. coverage in a certain time period of execution or number of crashes (Target: Early May)
-   [x] Implement a transformer model using similar technique to montage to tokenize/embed the AST directly instead of converting to code (Target: Late May)
-   [x] Execute fuzzer on several versions of JavaScript engines and use the evaluation metric to tune reward function and hyperparameters (Target: Early June)
-   [x] Compare results of RL agent with a random agent and other fuzzers in terms of bugs found in older versions and coverage gained per time period (Target: Early June)
-   [x] Write a report (Target: Mid June)

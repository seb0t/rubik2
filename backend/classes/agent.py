from backend.classes.cube import RubikEnv
from typing import Dict, Any
from collections import deque
import numpy as np
import json
import os


class RubikAgent:
    def __init__(
            self,
            env : RubikEnv,
            alpha: float = 0.001,
            gamma: float = 0.95,
            epsilon: float = 1.0,
            epsilonDecay: float = 0.995,
            epsilonMin: float = 0.1,
            memorySize: int = 100000,
            batchSize: int = 64,
            qTable : Dict[Any, Any] = {}
        ) -> None:
    
        """
        Initializes the Rubik's Cube agent with the given parameters.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.memorySize = memorySize
        self.batchSize = batchSize
        self.qTable = qTable
        self.memory : deque[Any] = deque(maxlen=memorySize)
        self.tempMemory :dict[str, Any] = {}

        self.stateDim = env.state.size
        self.actionDim = env.actionMap.__len__()

        # metrics
        self.rewardsHistory : list[float] = []
        self.lossHistory : list[float] = []
        self.epsilonHistory : list[float] = []

    def encodeState(self , state : np.ndarray) -> Any:
        """
        Encodes the current state of the Rubik's Cube into a unique integer representation.
        """
        encodedState :str = ""
        for value in state.flatten():
            if value != 0:
                encodedState += str(int(value))

        try:
            if len(encodedState) != 54:
                raise ValueError("Encoded state length is not 54.")
            return int(encodedState)
        except ValueError as ve:
            print(f"Error encoding state: {ve}")
            return None
        
    def choseActionFromQTable(self, epsilonOverride: float | bool | None = None, verbose: int = 0, state: np.ndarray | None = None) -> int:
        """
        This function selects the action to execute based on the current state using the Q-Table.
        It generates a random value between 0 and 1 and compares it with epsilon.

        - A higher epsilon favors exploration (random actions), allowing the agent to discover new strategies.
        - A lower epsilon favors exploitation (best-known actions), optimizing for maximum reward based on learned values.

        If the current state does not yet have an entry in the Q-map, a new entry is created with all initial values set to 0.

        Returns the index of the chosen action from the action_map:

        Parameters:
        - state (np.ndarray, optional): The current state of the cube.
        - epsilonOverride (float | bool | None, optional): If specified, this value will override self.epsilon.
        - verbose (int): Verbosity level for debugging (0 = silent).

        Returns:
        - int: The index of the chosen action.
        """

        currentEpsilon = self.epsilon if epsilonOverride is None else epsilonOverride
        randomEpsilon = np.random.rand()

        if verbose > 1:
            print(f"Current Epsilon: {currentEpsilon}, Random Epsilon: {randomEpsilon}")

        if randomEpsilon < currentEpsilon:
            return self.choseRandomActionId(verbose)

        else:
            if state is None:
                state = self.env.state
            encodedState = self.encodeState(state)
            if encodedState not in self.qTable:
                if verbose > 1:
                    print(f"State {encodedState} not in Q-table. New entry created.")
                self.addStateToQtable(encodedState)

            return self.choseBestActionId(verbose, encodedState) 

    def choseRandomActionId(self, verbose : int = 0) -> int:
        actionIndex : int = np.random.randint(0, self.actionDim)
        if verbose > 0:
            print(f"Chose random action: {actionIndex}")
        return actionIndex

    def addStateToQtable(self, encodedState : int):
        self.qTable[encodedState] = np.zeros(self.actionDim)

    def choseBestActionId(self, verbose: int, encodedState: int) -> int:
        if encodedState not in self.qTable:
            print(f"State {encodedState} not found in Q-table. Adding it now.")
            self.addStateToQtable(encodedState)

        qValues = self.qTable[encodedState]
        bestActionIdList: list[int] = np.flatnonzero(qValues == np.max(qValues)).tolist()
        if len(bestActionIdList) > 1:
            bestActionIndex = np.random.choice(bestActionIdList)
            if verbose > 1:
                print(f"Multiple best actions found. Randomly selected action: {bestActionIndex}")
        else:
            bestActionIndex = bestActionIdList[0]
            if verbose > 1:
                print(f"Chose best action: {bestActionIndex}")
        return bestActionIndex 
    
    def updateQtable(
            self,
            currentState: np.ndarray,
            nextState: np.ndarray,
            actionId: int,
            reward: float
            ) -> None:
        """
        Updates the Q-table for the current state-action pair using the Q-learning update rule:

            Q(s, a) ← Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]

        Where:
        - Q(s, a): The current Q-value for state s and action a.
        - α (alpha): The learning rate, determining the update speed. A higher alpha makes the agent adapt quickly to new information, while a lower alpha makes learning more stable but slower.
        - r: The reward received after taking action a in state s.
        - γ (gamma): The discount factor, weighting the importance of future rewards. A higher gamma makes the agent prioritize long-term rewards, while a lower gamma focuses on immediate rewards.
        - s': The next state after taking action a.
        - max Q(s', a'): The maximum Q-value for the next state s' over all possible actions a'.

        Parameters:
        - currentState (np.ndarray): The current state (s).
        - nextState (np.ndarray): The next state (s') after taking the action.
        - actionId (int): The action taken (a).
        - reward (float): The reward received (r).

        Returns:
        - None
        """

        currentStateEncoded = self.encodeState(currentState)
        nextStateEncoded = self.encodeState(nextState)

        if currentStateEncoded not in self.qTable:
            self.addStateToQtable(currentStateEncoded)
        if nextStateEncoded not in self.qTable:
            self.addStateToQtable(nextStateEncoded)

        bestNextStateActionId: int = self.choseBestActionId(0, nextStateEncoded)
        tdTarget: float = reward + self.gamma * self.qTable[nextStateEncoded][bestNextStateActionId]
        tdDelta: float = tdTarget - self.qTable[currentStateEncoded][actionId]
        self.qTable[currentStateEncoded][actionId] += self.alpha * tdDelta

    def train(
            self,
            numEpisodes: int = 1000,
            maxStepsPerEpisode: int = 50,
            resetEpsilon: bool = True,
            minTimeShuffle: int = 1,
            maxTimeShuffle: None | int = 5,
            verbose: int = 0,
            epsilonOverride: float | bool | None = None,
        ) -> dict[Any, Any]:
        """
        Trains the agent using the Q-learning algorithm.
        """
        
        if minTimeShuffle < 1:
            raise ValueError("minTimeShuffle must be at least 1.")
        if maxTimeShuffle is None:
            maxTimeShuffle = minTimeShuffle +1

        history : dict[Any, Any] = {}

        if resetEpsilon:
            self.epsilon = 1.0

        for shuffleNum in range(minTimeShuffle, maxTimeShuffle+1):
            solvedCases : list[Any] = []
            history[shuffleNum] = {}
            history[shuffleNum]["solvedCasesNumber"] = 0
            history[shuffleNum]["unsolvedCasesNumber"] = 0
            history[shuffleNum]["percentSolved"] = 0.0

            for episode in range(numEpisodes):
                self.env.reset()
                self.env.shuffle(shuffleNum)
                startingEncodedState = self.encodeState(self.env.state)
                totalReward : float = 0
                done : bool = False

                for step in range(maxStepsPerEpisode):
                    currentState = self.env.state.copy()
                    actionId = self.choseActionFromQTable(epsilonOverride, verbose, currentState)
                    _, nextState, reward, done = self.env.step(actionId)
                    self.updateQtable(currentState, nextState, actionId, reward)
                    totalReward += reward

                    if done:
                        if verbose > 0:
                            print(f"Episode {episode+1}/{numEpisodes} solved in {step+1} steps with reward {totalReward}.")
                        solvedCases.append({startingEncodedState : step+1})
                        history[shuffleNum]["solvedCasesNumber"] += 1
                        break

                if not done:
                    if verbose > 0:
                        print(f"Episode {episode+1}/{numEpisodes} not solved. Total reward: {totalReward}.")
                    history[shuffleNum]["unsolvedCasesNumber"] += 1

                # Decay epsilon
                if self.epsilon > self.epsilonMin:
                    self.epsilon *= self.epsilonDecay

            totalCases = history[shuffleNum]["solvedCasesNumber"] + history[shuffleNum]["unsolvedCasesNumber"]
            if totalCases > 0:
                history[shuffleNum]["percentSolved"] = (history[shuffleNum]["solvedCasesNumber"] / totalCases) * 100.0
            if verbose > 0:
                print(f"Shuffle {shuffleNum}: Solved {history[shuffleNum]['solvedCasesNumber']} out of {totalCases} cases ({history[shuffleNum]['percentSolved']:.2f}%).")
        
        return history
    
    def solveCube(self, maxSteps: int = 100, verbose: int = 0) -> list[str]:
        """
        Solves the Rubik's Cube using the learned Q-table.

        Parameters:
        - maxSteps (int): Maximum number of steps to attempt solving the cube.
        - verbose (int): Verbosity level for debugging (0 = silent).

        Returns:
        - list[int]: A list of action indices taken to solve the cube.
        """
        stepsTaken : list[str] = []
        done : bool = False

        for step in range(maxSteps):
            currentState = self.env.state.copy()
            actionId = self.choseBestActionId(verbose, self.encodeState(currentState))
            _, _, reward, done = self.env.step(actionId)
            stepsTaken.append(self.env.actionMap[actionId])

            if verbose > 0:
                print(f"Step {step+1}: Action {actionId}, Reward: {reward}")

            if done:
                if verbose > 0:
                    print(f"Cube solved in {step+1} steps.")
                break

        if not done and verbose > 0:
            print("Failed to solve the cube within the maximum allowed steps.")

        return stepsTaken
    
    def saveQtableToFile(self, folder: str = "../data/qtable") -> None:
        """
        Saves the Q-table to a file inside the specified folder. If the file already exists,
        prompts the user for confirmation before overwriting.
        """
        

        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # Find the next available file name
        file_index = 1
        filepath = os.path.join(folder, f"qtable_{file_index}.json")
        while os.path.exists(filepath):
            overwrite = input(f"File {filepath} already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite == 'y':
                break
            elif overwrite == 'n':
                file_index += 1
                filepath = os.path.join(folder, f"qtable_{file_index}.json")
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

        # Save the Q-table to the file
        with open(filepath, 'w') as f:
            json.dump({str(k): v.tolist() for k, v in self.qTable.items()}, f)

    def loadQtableFromFile(self, filepath: str) -> None:
        """
        Loads the Q-table from the specified file.
        """
        with open(filepath, 'r') as f:
            qtable_data = json.load(f)
            self.qTable = {int(k): np.array(v) for k, v in qtable_data.items()}

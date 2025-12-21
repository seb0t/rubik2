from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def showCube(states: Union[np.ndarray, list[np.ndarray]]):
    """
    Plot a list of 2D arrays as colored squares.
    Each array represents a Rubik's cube state.
    """
    color_map = {
        0: "none",
        1: "white",
        2: "red",
        3: "green",
        4: "blue",
        5: "orange",
        6: "yellow"
    }

    # Ensure states is a list of 2D arrays
    if isinstance(states, np.ndarray):
        states = [states]  # Wrap single 2D array in a list

    num_states = len(states)
    fig, axes = plt.subplots(1, num_states, figsize=(num_states * 5, 5), facecolor="black")

    if num_states == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    for ax, state in zip(axes, states):
        if len(state.shape) != 2:  # Validate that state is 2D
            raise ValueError(f"Each state must be a 2D array. Got shape {state.shape} instead.")
        rows, cols = state.shape
        for i in range(rows):
            for j in range(cols):
                value = state[i, j]
                if value != 0:
                    ax.add_patch(Rectangle((j, rows - i - 1), 1, 1, color=color_map[value], ec="gray"))
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("black")

    plt.show()
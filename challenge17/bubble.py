import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import numpy as np
from matplotlib.animation import PillowWriter

class ProcessingElement:
    def __init__(self, value):
        self.value = value

    def compare_and_swap(self, neighbor):
        if self.value > neighbor.value:
            self.value, neighbor.value = neighbor.value, self.value

def systolic_bubble_sort_animated(arr):
    N = len(arr)
    pes = [ProcessingElement(val) for val in arr]
    states = [[pe.value for pe in pes]]

    for pass_num in range(N):
        if pass_num % 2 == 0:
            for i in range(0, N - 1, 2):
                pes[i].compare_and_swap(pes[i + 1])
        else:
            for i in range(1, N - 1, 2):
                pes[i].compare_and_swap(pes[i + 1])
        states.append([pe.value for pe in pes])
    
    return states

# Example array
arr = [5, 1, 4, 2, 3]
states = systolic_bubble_sort_animated(arr)

# Modified animation where x-axis is labeled 1 to 5
fig, ax = plt.subplots()
bar_rects = ax.bar(range(1, len(arr)+1), states[0], align="center", color="skyblue")
ax.set_ylim(0, max(arr) + 1)
ax.set_xticks(range(1, len(arr)+1))
ax.set_xlabel("Processing Elements (PEs)")
ax.set_title("Systolic Array Bubble Sort Animation")

def update(frame):
    for rect, height in zip(bar_rects, states[frame]):
        rect.set_height(height)
    ax.set_ylabel(f"Pass {frame}")

ani = animation.FuncAnimation(fig, update, frames=len(states), repeat=False, interval=800)
ani.save('animation.gif', writer=PillowWriter(fps=10))
plt.show()

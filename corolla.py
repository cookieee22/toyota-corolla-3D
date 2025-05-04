import numpy as np
import matplotlib.pyplot as plt
import math

# --- Car Model Data (3D Coordinates) ---
D = np.array([
    [-6.5, -6.5, -6.5, -6.5, -2.5, -2.5, -0.75, -0.75,
     3.25, 3.25, 4.5, 4.5, 6.5, 6.5, 6.5, 6.5],
    [-2, -2, 0.5, 0.5, 0.5, 0.5, 2, 2,
     2, 2, 0.5, 0.5, 0.5, 0.5, -2, -2],
    [-2.5, 2.5, 2.5, -2.5, -2.5, 2.5, -2.5, 2.5,
     -2.5, 2.5, -2.5, 2.5, -2.5, 2.5, 2.5, -2.5],
    [1]*16
])

# --- Edge Connections ---
connections = [
    (0, 1), (0, 3), (0, 15), (1, 2), (1, 14), (2, 3), (2, 5), (3, 4),
    (4, 5), (4, 6), (5, 7), (6, 7), (6, 8), (7, 9), (8, 9), (8, 10),
    (9, 11), (10, 11), (10, 12), (11, 13), (12, 13), (12, 15), (13, 14), (14, 15)
]

# --- Helper Functions ---
def get_projection_matrix(b, c, d):
    return np.array([
        [1, 0, -b/d, 0],
        [0, 1, -c/d, 0],
        [0, 0, 0,    0],
        [0, 0, -1/d, 1]
    ])

def rotate_y(theta):
    t = math.radians(theta)
    return np.array([
        [math.cos(t), 0, math.sin(t), 0],
        [0, 1, 0, 0],
        [-math.sin(t), 0, math.cos(t), 0],
        [0, 0, 0, 1]
    ])

def rotate_z(theta):
    t = math.radians(theta)
    return np.array([
        [math.cos(t), -math.sin(t), 0, 0],
        [math.sin(t),  math.cos(t), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def zoom(factor):
    return np.diag([factor, factor, factor, 1])

def project(D, transform, P):
    T = transform @ D
    PD = P @ T
    PD /= PD[3]
    return PD[0], PD[1]

# --- Projection Matrices ---
P_default = get_projection_matrix(0, 10, 25)
P_shifted = get_projection_matrix(-5, 10, 10)

# --- Setup 5 Rows x 2 Columns (One for each Question) ---
fig, axs = plt.subplots(5, 2, figsize=(14, 25))
axs = axs.flatten()
index = 0

# --- Q1: Projection with Center (-5, 10, 10) ---
x, y = project(D, np.eye(4), P_shifted)
ax = axs[index]
for a, b in connections:
    ax.plot([x[a], x[b]], [y[a], y[b]], 'k-', linewidth=1)
ax.scatter(x, y, color='blue', s=10)
ax.set_title("Projection with Center (-5,10,10)")
ax.axis('equal'); ax.axis('off')
index += 1

# --- Q2: Projection with Center (0, 10, 25) ---
x, y = project(D, np.eye(4), P_default)
ax = axs[index]
for a, b in connections:
    ax.plot([x[a], x[b]], [y[a], y[b]], 'k-', linewidth=1)
ax.scatter(x, y, color='blue', s=10)
ax.set_title("Projection with Center (0,10,25)")
ax.axis('equal'); ax.axis('off')
index += 1

# --- Q3: 30° Y Rotation + Projection (side-by-side) ---
for title, transform in [("Original Projection", np.eye(4)),
                         ("30° Y Rotation + Projection", rotate_y(30))]:
    x, y = project(D, transform, P_default)
    ax = axs[index]
    for a, b in connections:
        ax.plot([x[a], x[b]], [y[a], y[b]], 'k-', linewidth=1)
    ax.scatter(x, y, color='blue', s=10)
    ax.set_title(title)
    ax.axis('equal'); ax.axis('off')
    index += 1

# --- Q4: 45° Z Rotation + Projection (side-by-side) ---
for title, transform in [("Original Projection", np.eye(4)),
                         ("45° Z Rotation + Projection", rotate_z(45))]:
    x, y = project(D, transform, P_default)
    ax = axs[index]
    for a, b in connections:
        ax.plot([x[a], x[b]], [y[a], y[b]], 'k-', linewidth=1)
    ax.scatter(x, y, color='blue', s=10)
    ax.set_title(title)
    ax.axis('equal'); ax.axis('off')
    index += 1

# --- Q5: 150% Zoom + Projection (side-by-side) ---
for title, transform in [("Original Projection", np.eye(4)),
                         ("150% Zoom + Projection", zoom(1.5))]:
    x, y = project(D, transform, P_default)
    ax = axs[index]
    for a, b in connections:
        ax.plot([x[a], x[b]], [y[a], y[b]], 'k-', linewidth=1)
    ax.scatter(x, y, color='blue', s=10)
    ax.set_title(title)
    ax.axis('equal'); ax.axis('off')
    index += 1

plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
plt.show()

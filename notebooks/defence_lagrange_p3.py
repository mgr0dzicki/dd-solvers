import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update(
    {
        "mathtext.fontset": "cm",
        # "font.family": "serif",
        "axes.titlesize": 14,
        "font.size": 12,
    }
)

p = 3
# Reference triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.0, 1.0])

# Generate barycentric nodes (i, j, k) / p where i+j+k = p
nodes = []
indices = []
for i in range(p + 1):
    for j in range(p + 1 - i):
        k = p - i - j
        alpha, beta, gamma = i / p, j / p, k / p
        # Map barycentric to Cartesian on triangle A,B,C
        point = alpha * A + beta * B + gamma * C
        nodes.append(point)
        indices.append((i, j, k))

nodes = np.array(nodes)

# Plot
fig, ax = plt.subplots(figsize=(3.2, 3.2))
# triangle edges
tri_x = [A[0], B[0], C[0], A[0]]
tri_y = [A[1], B[1], C[1], A[1]]
ax.plot(tri_x, tri_y, linewidth=1.5, color="black")

# scatter nodes
ax.scatter(nodes[:, 0], nodes[:, 1], s=60, zorder=5, color="black")

ax.set_title(r"$\mathbb{P}_p$", fontsize=40)
ax.text(
    0.5,
    -0.3,
    r"$u|_T(\mathbf{x})=\sum_{|\alpha|\leq p} {\lambda}_{T,\alpha}\,\mathbf{x}^{\alpha}$",
    ha="center",
    fontsize=25,
)

ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.axis("off")

# Save as SVG
out_path = Path("../docs/defence-media/lagrange_p3.svg")
fig.savefig(out_path, format="svg", bbox_inches="tight", transparent=True)
print("Saved:", out_path.resolve())

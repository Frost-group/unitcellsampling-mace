import sys
import numpy as np
import matplotlib.pyplot as plt


def get_slice(grid, axis="z", index=None):
    if index is None:
        index = grid.shape[{"x": 0, "y": 1, "z": 2}[axis]] // 2

    if axis == "x":
        plane = grid[index, :, :]
        xlabel, ylabel = "y", "z"
    elif axis == "y":
        plane = grid[:, index, :]
        xlabel, ylabel = "x", "z"
    else:
        plane = grid[:, :, index]
        xlabel, ylabel = "x", "y"

    return plane, index, xlabel, ylabel


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_npy_grid.py grid.npy [axis] [index]")
        print("Example: python plot_npy_grid.py NaTaCl6_Na_probe_mace_grid.npy z 2")
        sys.exit(1)

    gridfile = sys.argv[1]
    axis = sys.argv[2].lower() if len(sys.argv) > 2 else "z"
    index = int(sys.argv[3]) if len(sys.argv) > 3 else None

    grid = np.load(gridfile)
    print("Loaded:", gridfile)
    print("Shape:", grid.shape)
    print("Dtype:", grid.dtype)

    # Convert giant placeholder values into NaN for plotting
    grid = grid.astype(float)
    grid[grid > 1e100] = np.nan

    finite = np.isfinite(grid)
    print("Finite points:", np.count_nonzero(finite), "/", grid.size)
    if np.any(finite):
        print("Finite min:", np.nanmin(grid))
        print("Finite max:", np.nanmax(grid))

    plane, index, xlabel, ylabel = get_slice(grid, axis=axis, index=index)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(plane, origin="lower", cmap="viridis")
    plt.colorbar(im, label="Energy")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{axis}-slice at index {index}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
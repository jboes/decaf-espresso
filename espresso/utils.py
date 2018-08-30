import numpy as np


def get_max_empty_space(atoms, edir=3):
    """Assuming periodic boundary conditions, finds the largest
    continuous segment of free, unoccupied space and returns
    its midpoint in scaled coordinates (0 to 1) in the edir
    direction (default z).

    Parameters
    ----------
    atoms : Atoms object
        Unit cell to find the maximum distance for.
    edir : int (1 | 2 | 3)
        Direction to search for the maximum distance in.

    Returns
    -------
    max_distance : float
        The scaled maximum distance between atoms in the unit cell
        in resepcts to edir.
    """
    position_array = atoms.get_scaled_positions()[..., edir - 1]
    position_array.sort()
    differences = np.diff(position_array)
    differences = np.append(
        differences,
        position_array[0] + 1 - position_array[-1])
    max_diff_index = np.argmax(differences)
    if max_diff_index == len(position_array) - 1:
        max_distance = (position_array[0] + 1 +
                        position_array[-1]) / 2. % 1
    else:
        max_distance = (position_array[max_diff_index] +
                        position_array[max_diff_index + 1]) / 2.

    return max_distance

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Provides routines for reading from and writing to VMD-readable XYZ format files.
Main functions: save, load
"""

import logging
import numpy as np

# Configure a logger for the library.
# The user of the library can then configure the root logger to see these messages.
logger = logging.getLogger(__name__)

def _format_xyz_line(atom_symbol: str, coords: np.ndarray, fmt: str = "15.10f") -> str:
    """
    Formats a single atom's data into one line for an XYZ file.

    This is an internal helper function that handles 1D, 2D, or 3D coordinates,
    padding with zeros where necessary.

    Parameters
    ----------
    atom_symbol : str
        The chemical symbol of the atom (e.g., 'H', 'C', 'O').
    coords : np.ndarray
        A 1D numpy array containing the atom's coordinates (x, y, z).
        The size can be 1, 2, or 3.

    Returns
    -------
    str
        A formatted string with aligned columns.
    """
    # Right-align atom symbol in a 2-character space.
    # Format coordinates to a width of 15 with 10 decimal places.
    line = f"{atom_symbol:<2s} "
    if coords.size == 3:
        line += f"{coords[0]:{fmt}} {coords[1]:{fmt}} {coords[2]:{fmt}}"
    elif coords.size == 2:
        line += f"{coords[0]:{fmt}} {coords[1]:{fmt}} {0.0:{fmt}}"
    elif coords.size == 1:
        line += f"{coords[0]:{fmt}} {0.0:{fmt}} {0.0:{fmt}}"
    else:
        # This case should be prevented by the validation in the `save` function.
        raise ValueError(f"Coordinate array for an atom must have size 1, 2, or 3, but got {coords.size}")
    return line


def save(
    filepath: str,
    coords: np.ndarray,
    atom_symbols: list[str] | np.ndarray,
    comment: str | list[str] | None = None,
    append: bool = False
) -> None:
    """
    Saves atomic coordinates to a file in the aligned XYZ format.

    This function handles both single-frame (2D coordinate array) and
    multi-frame (3D coordinate array) data. It can also process coordinates
    specified in 1D, 2D, or 3D, padding with zeros as needed.

    Parameters
    ----------
    filepath : str
        The path to the file to be written.
    coords : np.ndarray
        The atomic coordinates.
        - For a single frame: shape must be (N, D) where N is the number of
          atoms and D (dimensions) is 1, 2, or 3.
        - For multiple frames: shape must be (T, N, D) where T is the
          number of frames.
    atom_symbols : list[str] or np.ndarray
        A list or array of atom symbols of length N.
    comment : str or list[str], optional
        A string to be placed on the comment line (line 2) of the file.
        For multi-frame data, a list of strings can be provided, with one
        comment per frame. If None, a default comment is generated.
        Defaults to None.
    append : bool, optional
        If True, data will be appended to an existing file instead of
        overwriting it. Defaults to False.

    Raises
    ------
    ValueError
        If the shapes of the coordinates and atom symbols are incompatible.
    """
    logger.debug(f"Attempting to save data to '{filepath}' with append={append}.")
    coords = np.asarray(coords)
    atom_symbols = np.asarray(atom_symbols)
    num_atoms = len(atom_symbols)

    if coords.ndim not in [2, 3] or coords.shape[-1] not in [1, 2, 3]:
        msg = (
            f"Coordinates array `coords` must be 2D (N, D) or 3D (T, N, D), "
            f"where D is 1, 2, or 3. Received shape: {coords.shape}"
        )
        logger.error(msg)
        raise ValueError(msg)

    if coords.shape[-2] != num_atoms:
        msg = (
            f"The number of atoms in coordinates ({coords.shape[-2]}) does not "
            f"match the number of atom symbols ({num_atoms})."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Standardize to 3D array for consistent processing
    is_multi_frame = coords.ndim == 3
    if not is_multi_frame:
        coords = np.expand_dims(coords, axis=0)

    num_frames = coords.shape[0]

    # Process comments
    comments = []
    if comment is None:
        comments = [f"Frame {t}" for t in range(num_frames)]
    elif isinstance(comment, str):
        comments = [comment] * num_frames
    elif isinstance(comment, list) and len(comment) == num_frames:
        comments = comment
    else:
        msg = "`comment` must be a string or a list of strings with the same length as the number of frames."
        logger.error(msg)
        raise ValueError(msg)

    write_mode = 'a' if append else 'w'
    with open(filepath, write_mode) as f:
        for i in range(num_frames):
            f.write(f"{num_atoms}\n")
            f.write(f"{comments[i]}\n")
            for j in range(num_atoms):
                line = _format_xyz_line(atom_symbols[j], coords[i, j])
                f.write(f"{line}\n")
    
    logger.info(f"Successfully saved {num_frames} frame(s) to '{filepath}'.")


def load(filepath: str, return_symbols: bool = False, return_all: bool = False):
    """
    Reads coordinates from an XYZ file.

    This function can handle single-frame or multi-frame XYZ files. If the file
    contains only one frame, the returned coordinate array is squeezed to 2D.

    Parameters
    ----------
    filepath : str
        The path of the XYZ file to read.
    return_symbols : bool, optional
        If True, also returns the list of atom symbols. Defaults to False.
    return_all : bool, optional
        If True, also returns atom symbols and comment lines. This option
        overrides `return_symbols`. Defaults to False.

    Returns
    -------
    numpy.ndarray or tuple
        - By default: returns the coordinate array `coords`.
          - Single-frame file: (N, 3)
          - Multi-frame file: (T, N, 3)
        - If `return_symbols` is True: returns `(coords, atom_symbols)`.
        - If `return_all` is True: returns `(coords, atom_symbols, comments)`.

    Raises
    ------
    IOError
        If the file format is incorrect, the file cannot be read, or the
        atom count is inconsistent with the file content.
    """
    logger.debug(f"Attempting to load data from '{filepath}'.")
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.error(f"File not found at path: {filepath}")
        raise

    # Clean trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()

    frames_data = []
    i = 0
    while i < len(lines):
        try:
            num_atoms = int(lines[i].strip())
            frame_lines = lines[i : i + num_atoms + 2]
            if len(frame_lines) < num_atoms + 2:
                raise IndexError
            frames_data.append(frame_lines)
            i += num_atoms + 2
        except (ValueError, IndexError):
            msg = (
                f"XYZ file '{filepath}' is malformed or truncated near line {i + 1}."
            )
            logger.error(msg)
            raise IOError(msg)

    if not frames_data:
        logger.warning(f"File '{filepath}' is empty or contains no valid frames.")
        if return_all:
             return np.array([]), np.array([]), []
        elif return_symbols:
             return np.array([]), np.array([])
        else:
             return np.array([])

    num_atoms_first_frame = int(frames_data[0][0])
    
    # Parse atom symbols (from the first frame only)
    atom_symbols = np.array([line.split()[0] for line in frames_data[0][2:]])

    # Parse coordinates and comments
    all_coords = np.empty((len(frames_data), num_atoms_first_frame, 3))
    comments = []
    for frame_idx, frame in enumerate(frames_data):
        comments.append(frame[1].strip())
        coord_lines = [line.split()[1:4] for line in frame[2:]]
        all_coords[frame_idx] = np.array(coord_lines, dtype=float)

    # Squeeze dimension if only one frame is present
    final_coords = all_coords[0] if all_coords.shape[0] == 1 else all_coords
    logger.info(f"Successfully loaded {len(frames_data)} frame(s) from '{filepath}'.")

    if return_all:
        final_comments = comments[0] if len(comments) == 1 else comments
        return final_coords, atom_symbols, final_comments
    if return_symbols:
        return final_coords, atom_symbols
    return final_coords


if __name__ == '__main__':

    TEST_FILENAME = 'vmd_test_pep8.xyz'

    # Define single-frame test data (3D and 2D)
    symbols = ['O', 'H', 'H']
    coords_3d = np.array([
        [0.0000, 0.0000, 0.1173],
        [0.0000, 0.7572, -0.4692],
        [0.0000, -0.7572, -0.4692]
    ])
    coords_2d = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.866]
    ])
    comment_single = "Single water molecule"

    # --- 1. Test `save` function ---
    logger.info("--- Testing `save` function ---")
    
    # Test saving 3D coordinates
    save(TEST_FILENAME, coords_3d, symbols, comment=comment_single)
    
    # Test appending with 2D coordinates (will be padded with z=0.0)
    save(TEST_FILENAME, coords_2d, symbols, comment="A 2D triangle", append=True)

    # --- 2. Test `load` function ---
    logger.info("\n--- Testing `load` function ---")

    try:
        # Test loading all contents
        coords_loaded, symbols_loaded, comments_loaded = load(TEST_FILENAME, return_all=True)
        
        logger.info(f"Load successful! Shape of loaded coords: {coords_loaded.shape}")
        logger.info(f"Loaded atom symbols: {symbols_loaded}")
        logger.info(f"Loaded comments: {comments_loaded}")
        
        # --- 3. Verification ---
        logger.info("\n--- Verifying data consistency ---")
        assert coords_loaded.shape == (2, 3, 3)
        assert np.all(symbols_loaded == symbols)
        
        # Verify first frame (originally 3D)
        assert np.allclose(coords_loaded[0], coords_3d)
        
        # Verify second frame (originally 2D, now padded)
        coords_2d_padded = np.hstack([coords_2d, np.zeros((3, 1))])
        assert np.allclose(coords_loaded[1], coords_2d_padded)

        logger.info("✅ Verification successful! Data was saved and loaded correctly.")

    except (IOError, ValueError, AssertionError) as e:
        logger.error(f"A test failed: {e}")

    # --- 4. Preview file content ---
    logger.info(f"\n--- Content of '{TEST_FILENAME}' ---")
    with open(TEST_FILENAME, 'r') as f:
        # Use print() here for direct output of file content for visual check
        print(f.read())
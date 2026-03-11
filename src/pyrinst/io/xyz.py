#!/usr/bin/env python3
"""
Provides routines for reading from and writing to VMD-readable XYZ format files.
Main functions: save, load
"""

import logging
import re
from io import TextIOWrapper

import numpy as np

from pyrinst.utils.units import ANGSTROM, Length

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


def lines(atom_symbols: list[str] | np.ndarray, coords: np.ndarray, fmt: str = "15.10f") -> str:
    """
    Generates xyz coordinate lines without the header.

    This function formats atomic coordinates as a multi-line string in xyz format,
    but without the atom count and comment lines that are typically present in
    standard xyz files. This is useful for embedding xyz-formatted coordinates
    into input files for ab initio programs.

    Parameters
    ----------
    coords : np.ndarray
        The atomic coordinates in a.u.
        - Shape must be (N, D) where N is the number of atoms and
          D (dimensions) is 1, 2, or 3.
        - Coordinates will be automatically converted from a.u. to Angstroms.
    atom_symbols : list[str] or np.ndarray
        A list or array of atom symbols of length N.
    fmt : str, optional
        Format string for the coordinates (e.g., "15.10f" for width 15
        with 10 decimal places). Defaults to "15.10f".

    Returns
    -------
    str
        A multi-line string containing the formatted xyz coordinate lines,
        with each line in the format "SYMBOL  x  y  z".
        The string does NOT end with a trailing newline.

    Raises
    ------
    ValueError
        If the shapes of the coordinates and atom symbols are incompatible,
        or if coords is not a 2D array.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> symbols = ['H', 'H']
    >>> print(lines(coords, symbols))
    H    0.0000000000   0.0000000000   0.0000000000
    H    1.0000000000   0.0000000000   0.0000000000
    """
    logger.debug("Formatting xyz lines without header.")
    coords = np.asarray(coords) * Length(1, "au").get("A")
    atom_symbols = np.asarray(atom_symbols)
    num_atoms = len(atom_symbols)

    if coords.ndim != 2 or coords.shape[-1] not in [1, 2, 3]:
        msg = f"Coordinates array `coords` must be 2D (N, D), where D is 1, 2, or 3. Received shape: {coords.shape}"
        logger.error(msg)
        raise ValueError(msg)

    if coords.shape[0] != num_atoms:
        msg = (
            f"The number of atoms in coordinates ({coords.shape[0]}) does not "
            f"match the number of atom symbols ({num_atoms})."
        )
        logger.error(msg)
        raise ValueError(msg)

    lines = []
    for j in range(num_atoms):
        line = _format_xyz_line(atom_symbols[j], coords[j], fmt=fmt)
        lines.append(line)

    result = "\n".join(lines)
    logger.debug(f"Successfully formatted {num_atoms} coordinate lines.")
    return result


def save(
    filepath: str,
    coords: np.ndarray,
    atom_symbols: list[str] | np.ndarray,
    comment: str | list[str] | None = None,
    append: bool = False,
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
        The atomic coordinates in a.u.
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
    coords = np.asarray(coords) * Length(1, "au").get("A")
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

    write_mode = "a" if append else "w"
    with open(filepath, write_mode) as f:
        for i in range(num_frames):
            f.write(f"{num_atoms}\n")
            f.write(f"{comments[i]}\n")
            for j in range(num_atoms):
                line = _format_xyz_line(atom_symbols[j], coords[i, j])
                f.write(f"{line}\n")
    logger.info(f"Successfully saved {num_frames} frame(s) to '{filepath}'.")


def _read_frame(
    f: TextIOWrapper,
    num_atoms: int,
    read_coords: bool,
    energy_pattern: str | None | bool,
) -> tuple[list[str], list[list[float]], str | float | None]:
    """
    Helper function to read a single frame from an XYZ file iterator.

    Parameters
    ----------
    f : TextIOWrapper
        Iterator over lines of the file.
    num_atoms : int
        Number of atoms in the frame.
    read_coords : bool
        Whether to parse and return coordinates.
    energy_pattern : str, None, or bool
        If True, reads comments. If a string, extracts energy via regex.
        If False/None, skips comments.

    Returns
    -------
    tuple
        (frame_symbols, [frame_coords], [comment_val])
    """
    comment_line = next(f).strip()

    comment_val = None
    if energy_pattern:
        if isinstance(energy_pattern, str):
            match = re.search(energy_pattern, comment_line)
            if match:
                comment_val = float(match.group(1))
        else:
            comment_val = comment_line

    frame_coords = []
    frame_symbols = []
    for _ in range(num_atoms):
        atom_line = next(f).split()

        frame_symbols.append(atom_line[0])

        if read_coords:
            frame_coords.append([float(x) for x in atom_line[1:4]])

    return frame_symbols, [frame_coords], [comment_val]


def load(filepath: str | list[str], read_coords: bool = True, energy_pattern: str | None | bool = True):
    """
    Reads coordinates and optionally extracts energy from an XYZ file or a series of XYZ files.

    This function can handle single-frame or multi-frame XYZ files, as well as
    a list of such files. It reads frame by frame to save memory.

    Parameters
    ----------
    filepath : str or list[str]
        The path of the XYZ file(s) to read.
    read_coords : bool, optional
        If True, reads and returns the coordinate array. Defaults to True.
    energy_pattern : str or bool or None, optional
        If True, reads and returns the full comment lines.
        If a string, extracts information from the comment lines using regex and returns it
        as floats instead of returning the full comment strings.
        If False/None, skips reading comments. Defaults to True.

    Returns
    -------
    tuple
        Always returns a tuple `(atom_symbols, coords, comments)`.
        - `atom_symbols` is always returned as a 1D numpy array.
        - `coords` is returned if `read_coords` is True, otherwise `None`.
          - Single-file: squeezed if 1 frame (N, 3), else (T, N, 3)
          - Multi-file: stacked array of shape (num_files, ...)
        - `comments` is returned if `energy_pattern` is truthy, otherwise `None`.
          - If `energy_pattern` is True: returns comments (string, list of strings, or list of lists).
          - If `energy_pattern` is a string: returns energies as a numpy array.
            - For single file: (T,) array
            - For multi-file: (T, num_files) array

    Raises
    ------
    OSError
        If the file format is incorrect, the file cannot be read, or the
        atom count is inconsistent with the file content.
    """
    is_single_file = isinstance(filepath, str)
    filepaths = [filepath] if is_single_file else filepath

    atom_symbols = None
    all_file_coords = []
    all_file_comments = []

    for path in filepaths:
        logger.debug(f"Attempting to load data from '{path}'.")

        file_coords = []
        file_comments = []

        with open(path) as f:
            for line in f:
                num_atoms: int = int(line.strip())

                try:
                    atom_symbols, frame_coords, comment_val = _read_frame(f, num_atoms, read_coords, energy_pattern)
                except Exception as e:
                    raise OSError(f"Error reading frame {len(file_coords)} from '{path}'.") from e

                file_coords += frame_coords
                file_comments += comment_val

        all_file_coords.append(file_coords)
        all_file_comments.append(file_comments)

        count = len(file_coords) if read_coords else (len(file_comments) if file_comments else "unknown")
        print(f"Successfully loaded {count} frame(s) from '{path}'.")

    ret_coords = np.atleast_2d(np.squeeze(np.array(all_file_coords))) * ANGSTROM
    ret_comments = np.array(all_file_comments, dtype=float if isinstance(energy_pattern, str) else str)
    if is_single_file and energy_pattern:
        ret_comments = ret_comments[0]
    if not read_coords:
        ret_coords = None
    if not energy_pattern:
        ret_comments = None
    return atom_symbols, ret_coords, ret_comments


if __name__ == "__main__":
    TEST_FILENAME = "vmd_test_pep8.xyz"

    # Define single-frame test data (3D and 2D)
    symbols = ["O", "H", "H"]
    coords_3d = np.array([[0.0000, 0.0000, 0.1173], [0.0000, 0.7572, -0.4692], [0.0000, -0.7572, -0.4692]])
    coords_2d = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
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
        symbols_loaded, coords_loaded, comments_loaded = load(TEST_FILENAME)

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

    except (OSError, ValueError, AssertionError) as e:
        logger.error(f"A test failed: {e}")

    # --- 4. Preview file content ---
    logger.info(f"\n--- Content of '{TEST_FILENAME}' ---")
    with open(TEST_FILENAME) as f:
        # Use print() here for direct output of file content for visual check
        print(f.read())

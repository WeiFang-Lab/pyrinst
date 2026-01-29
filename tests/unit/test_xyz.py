import numpy as np
import pytest

from pyrinst.io import xyz


SYMBOLS = ['O', 'H', 'H']
COORDS_3D = np.array([
    [0.0000, 0.0000, 0.1173],
    [0.0000, 0.7572, -0.4692],
    [0.0000, -0.7572, -0.4692]
])
COORDS_2D = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, 0.866]
])
COMMENT_SINGLE = "Single water molecule"
COMMENT_MULTI = ["Frame 1: 3D water", "Frame 2: 2D triangle"]


def test_save_and_load_roundtrip(tmp_path):
    """Tests if saving and then loading a file results in the original data.

    This is an integration test that covers the main functionality of both
    the `save` and `load` functions working together. It mimics the logic
    from the original `if __name__ == '__main__'` block.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A built-in pytest fixture that provides a temporary directory path.
    """
    # 1. Setup: Define a temporary file path
    test_filepath = tmp_path / "roundtrip.xyz"

    # 2. Action: Save a first frame, then append a second frame
    xyz.save(test_filepath, COORDS_3D, SYMBOLS, comment=COMMENT_MULTI[0])
    xyz.save(test_filepath, COORDS_2D, SYMBOLS, comment=COMMENT_MULTI[1], append=True)

    # 3. Action: Load the data back
    coords_loaded, symbols_loaded, comments_loaded = xyz.load(
        test_filepath, return_all=True
    )

    # 4. Verification: Assert that the loaded data is correct
    assert coords_loaded.shape == (2, 3, 3)
    assert np.array_equal(symbols_loaded, np.array(SYMBOLS))
    assert comments_loaded == COMMENT_MULTI

    # Verify the first frame (was 3D)
    assert np.allclose(coords_loaded[0], COORDS_3D)

    # Verify the second frame (was 2D, should be padded with zeros)
    coords_2d_padded = np.hstack([COORDS_2D, np.zeros((3, 1))])
    assert np.allclose(coords_loaded[1], coords_2d_padded)


def test_load_single_frame_squeezes_dimension(tmp_path):
    """Tests that loading a file with a single frame returns a 2D array.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory path provided by pytest.
    """
    test_filepath = tmp_path / "single_frame.xyz"

    # Action: Save just one frame
    xyz.save(test_filepath, COORDS_3D, SYMBOLS, comment=COMMENT_SINGLE)

    # Action: Load the data back (default return)
    coords_loaded = xyz.load(test_filepath)

    # Verification: The first dimension for frames should be squeezed.
    assert coords_loaded.shape == (3, 3)
    assert np.allclose(coords_loaded, COORDS_3D)


def test_save_raises_valueerror_on_mismatched_shapes():
    """Tests that `save` raises a ValueError for incompatible shapes.

    This test ensures that the input validation within the `save` function
    is working correctly.
    """
    # Symbols for 3 atoms, but coordinates for only 2
    mismatched_coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    with pytest.raises(ValueError, match="does not match the number"):
        xyz.save("dummy.xyz", mismatched_coords, SYMBOLS)


def test_load_raises_ioerror_on_malformed_file(tmp_path):
    """Tests that `load` raises an IOError for a malformed file.

    This test checks the error handling of the `load` function when it
    encounters a file that does not conform to the XYZ format.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory path provided by pytest.
    """
    test_filepath = tmp_path / "malformed.xyz"

    # Create a file where the atom count in the header is wrong.
    malformed_content = "10\nThis header is wrong\nO 0 0 0\nH 1 1 1\n"
    test_filepath.write_text(malformed_content)

    with pytest.raises(IOError, match="is malformed or truncated"):
        xyz.load(test_filepath)


def test_lines():
    """Tests that lines generates coordinate lines without headers.
    
    This test verifies that the function returns a properly formatted string
    containing xyz coordinate lines without the atom count and comment lines.
    """
    # Action: Generate xyz lines for a simple water molecule
    xyz_text = xyz.lines(SYMBOLS, COORDS_3D)
    
    # Verification: The result should be a string with lines (no header)
    assert isinstance(xyz_text, str)
    
    # Split into lines and verify we have the correct number
    lines = xyz_text.split('\n')
    assert len(lines) == 3  # Three atoms
    
    # Verify that each line starts with the correct atom symbol
    for i, line in enumerate(lines):
        assert line.startswith(SYMBOLS[i])
    
    # Verify the string does not contain header lines
    assert not xyz_text.startswith('3')  # No atom count
    assert COMMENT_SINGLE not in xyz_text  # No comment line


def test_lines_can_be_written_to_file(tmp_path):
    """Tests that the output of lines can be written to a file.
    
    This test verifies that the function output is suitable for writing
    directly to files, which is a common use case for ab initio programs.
    
    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory path provided by pytest.
    """
    test_filepath = tmp_path / "coords_only.txt"
    
    # Action: Generate xyz lines and write to file
    xyz_text = xyz.lines(SYMBOLS, COORDS_3D)
    with open(test_filepath, 'w') as f:
        f.write(xyz_text)
    
    # Verification: Read back and verify content
    content = test_filepath.read_text()
    assert content == xyz_text
    
    # Verify the file has exactly 3 lines (no header)
    lines = content.strip().split('\n')
    assert len(lines) == 3

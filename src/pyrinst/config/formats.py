"""
Global formatting constants for the application.

This module provides a centralized configuration for string formats used in
screen output, logging, and file writing.
"""

import numpy as np
import math

class Formats:
    """
    A namespace for string formatting constants.
    """

    MOMENTUM_OF_INERTIA = ".4f"
    ROTATIONAL_CONSTANT = ".4f"
    
    ENERGY = ".8f"
    GRADIENT = ".8f"
    GRAD_NORM = ".4e"

    FREQUENCY = ".4f"

    TEMPERATURE = ".4f"
    BETA = ".4f"  # inverse temperature

    PARTITION_FUNCTION = ".4e"
    LOG_PARTITION_FUNCTION = ".4f"
    RATE = ".4e"
    LOG_RATE = ".4f"

    BN = '.4f'
    ACTION = '.8f'
    TUNNELING_FACTOR = '.4e'

    STEP_INDEX = "3d"  # Integer, 5 characters wide

    MAX_LINE_LENGTH = 88
    ARRAY_SPACE = ' ' * 4


def format_array(
    arr: np.ndarray,
    fmt: str = ".4f",
    sep: str = Formats.ARRAY_SPACE,
    edge_items: int = 6,
    linewidth: int | None = 88,
    wrap_indent: str = ''
) -> str:
    """
    Formats a 1D array intelligently with guaranteed column alignment.

    This function provides a robust, all-in-one solution for pretty-printing
    1D NumPy arrays. It automatically handles alignment for any format
    specifier (e.g., '.4f', '.3e') by using a two-pass approach to find
    the maximum required width.

    Truncation is determined by `edge_items`: if the array has more than
    `edge_items * 2` elements, it shows the first and last `edge_items`
    elements, separated by an ellipsis.

    Parameters
    ----------
    arr : np.ndarray
        The 1D array to format.
    fmt : str, optional
        The format specifier for each element. Defaults to ".4f".
    sep : str, optional
        Separator between array elements, by default ' '.
    edge_items : int, optional
        Number of items to show at the beginning and end of a truncated
        array. Defaults to 6.
    linewidth : int or None, optional
        The maximum width of a line before wrapping. If None, no wrapping.
        Defaults to 88.
    wrap_indent : str, optional
        The indentation string for lines after the first one when wrapping
        occurs. Defaults to an empty string.

    Returns
    -------
    str
        The formatted, aligned, and wrapped string representation.

    Raises
    ------
    ValueError
        If `arr` is not a 1D array or if `edge_items` is not positive.
    """
    if arr.ndim != 1:
        raise ValueError("This function is designed for 1D arrays only.")
    if not isinstance(edge_items, int) or edge_items <= 0:
        raise ValueError("edge_items must be a positive integer.")

    # --- Step 1: Determine which elements to process ---
    max_items = edge_items * 2
    is_truncated = len(arr) > max_items
    if is_truncated:
        elements_to_process = np.concatenate([arr[:edge_items], arr[-edge_items:]])
    else:
        elements_to_process = arr

    # --- Step 2 (Pass 1): Find the maximum width required ---
    max_elem_width = 0
    
    for num in elements_to_process:
        try:
            formatted_num_str = f"{num:{fmt}}"
        except ValueError:
            # Re-raise with a more informative message
            raise ValueError(f"Invalid format specifier '{fmt}' for value of type {type(num)}.")
            
        if len(formatted_num_str) > max_elem_width:
            max_elem_width = len(formatted_num_str)
            
    if is_truncated:
        ellipsis_width = len('...')
        if ellipsis_width > max_elem_width:
            max_elem_width = ellipsis_width

    # --- Step 3: Generate the final list of equal-width string elements ---
    align_fmt = f"{{:>{max_elem_width}}}" 
    ellipsis_fmt = f"{{:^{max_elem_width}}}"

    elements_to_render = []
    
    def _format_and_align(num):
        return align_fmt.format(f"{num:{fmt}}")

    if not is_truncated:
        elements_to_render = [_format_and_align(num) for num in arr]
    else:
        head = [_format_and_align(num) for num in arr[:edge_items]]
        ellipsis = ellipsis_fmt.format('...')
        tail = [_format_and_align(num) for num in arr[-edge_items:]]
        elements_to_render = head + [ellipsis] + tail
        
    # --- Step 4: Join and wrap the final string elements ---
    if linewidth is None:
        return sep.join(elements_to_render)

    output_parts = []
    current_line_length = 0
    for i, elem_str in enumerate(elements_to_render):
        is_first_on_line = (i == 0) or (not output_parts) or (output_parts[-1] == wrap_indent)
        sep_len = 0 if is_first_on_line else len(sep)

        if current_line_length + sep_len + len(elem_str) > linewidth:
            output_parts.append('\n')
            output_parts.append(wrap_indent)
            current_line_length = len(wrap_indent)
            is_first_on_line = True
            sep_len = 0

        if not is_first_on_line:
            output_parts.append(sep)
            current_line_length += len(sep)
        
        output_parts.append(elem_str)
        current_line_length += len(elem_str)

    return "".join(output_parts)

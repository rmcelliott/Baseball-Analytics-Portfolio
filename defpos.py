"""
    Project Title: Oakland Ballers Defensive Positioning Cards

    Author: Riley Elliott

    Date:   9/18/2024
"""

import os
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.font_manager import FontProperties
import numpy as np
from pathlib import Path
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch


def append_column(matrix, new_column_values, column_index):
    """
    Appends a new column at the specified index in the matrix, initializing any missing columns if necessary.

    :param matrix: The original matrix (list of lists).
    :param new_column_values: The values to be inserted into the new column.
    :param column_index: The index at which the new column should be inserted.
    :return: None
    """
    # Warning if values are missing
    if len(matrix) != len(new_column_values):
        raise ValueError("The number of rows in the new column must match the number of rows in the matrix.")

    # Ensure all rows have enough columns
    for row in matrix:
        while len(row) < column_index:
            row.append(None)  # Append placeholder values

    # Insert each value from new_column_values to the specified column index in the matrix
    for i in range(len(matrix)):
        matrix[i].insert(column_index, new_column_values[i])


def get_last_name(name_list):
    """
    Re-formats a list of names.

    :param name_list: (list) a list of full names
    :return: (list) a list of last names, all capitalized
    """
    new_list = []

    for name in name_list:
        first_last = name.split()
        new_name = first_last[-1].upper()
        new_list.append(new_name)

    return new_list


def truncate_text(text, max_width, renderer, fontproperties):
    """
    Truncates the given text to fit within the specified maximum width.

    :param text: (str) The text to be truncated.
    :param max_width: (int) The maximum width of the cell
    :param renderer: An object used to calculate the width of the text
    :param fontproperties: Font information used to render the text
    :return: (str) Truncated text, or the original text if it fits
    """
    if renderer.get_text_width_height_descent(text, fontproperties, ismath=False)[0] <= max_width:
        return text

    truncated_text = ""
    current_line = ""

    for char in text:
        test_line = current_line + char
        width, _, _ = renderer.get_text_width_height_descent(test_line, fontproperties, ismath=False)

        if width <= max_width:
            current_line = test_line
        else:
            if current_line:
                truncated_text += current_line + "\n"
            current_line = char

    truncated_text += current_line

    return truncated_text


def order_batter_names(batter_dict):
    """
    Orders batter names according to the batting order (alphabetical otherwise).

    :param batter_dict: (dict) Dictionary of batter names, order spots, and defensive positioning info
    :return: (list, list) Ordered list of batter names and their corresponding spot in the order
    """
    # Separate the keys based on their sorting criteria
    numbered_keys = []
    dash_keys = []
    empty_keys = []

    for key, value in batter_dict.items():
        if value[0] == "-":
            dash_keys.append(key)
        elif value[0] == "":
            empty_keys.append(key)
        else:
            numbered_keys.append((key, int(value[0])))

    # Sort the numbered keys by their integer value
    numbered_keys.sort(key=lambda x: x[1])

    # Extract just the names in the correct order
    ordered_names = [key for key, _ in numbered_keys]

    # Find the appropriate place to insert dash_keys
    for key in dash_keys:
        for i in range(len(ordered_names)):
            if key.split()[-1].startswith('(') and ordered_names[i].split()[-1].startswith('('):
                ordered_names.insert(i + 1, key)
                break
        else:
            ordered_names.append(key)

    # Append the empty keys at the end
    ordered_names.extend(empty_keys)

    # Extract the values at index 0 in the same order
    ordered_values = [batter_dict[name][0] for name in ordered_names]

    return ordered_names, ordered_values


def generate_defpos(batter_dict, output_dir, pitcher_hand, num):
    """

    :param batter_dict: (dict) Dictionary of batter names, order spots, and defensive positioning info
    :param output_dir: (dir) Location for saving the output
    :param pitcher_hand: (str) Pitcher hand
    :param num: (int) Keeps track of function calls for later print formatting purposes
    :return: Generates and saves a copy of 12 doublesided defensive positioning cards (RHP one side, LHP the other)
    """
    # Collect names
    names, order = order_batter_names(batter_dict)
    lineup = get_last_name(list(names))
    numbers = [batter_dict[name][1] for name in names]

    # Initialize matrix and header rows
    if pitcher_hand == "Right":
        order_row = ["RHP"]
    else:
        order_row = ["LHP"]
    names_row = [""]
    number_row = [""]
    order_row.extend(order)
    number_row.extend(numbers)
    names_row.extend(lineup)
    defpos_matrix = [["1B"], ["2B"], ["3B"], ["SS"], ["LF"], ["CF"], ["RF"]]

    def find_key_by_last_name(batter_dict, last_name):
        """
        Finds the key in batter_dict that contains the given last name.

        :param batter_dict: Dictionary of batter names and their positions.
        :param last_name: The last name to search for.
        :return: The key that contains the last name, or None if not found.
        """
        for key in batter_dict.keys():
            if last_name in key.upper():
                return key
        return None

    # Add hitter rows to defpos_matrix
    for last_name in lineup:
        key = find_key_by_last_name(batter_dict, last_name)
        if key:
            append_column(defpos_matrix, batter_dict[key][2:], names_row.index(last_name.upper()))

    # Insert header rows into defpos_matrix
    defpos_matrix.insert(0, order_row)
    defpos_matrix.insert(1, number_row)
    defpos_matrix.insert(2, names_row)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis('off')
    table = Table(ax, bbox=[-0.12, -0.47, 1.24, 1.65])

    # Flatten the matrix and convert to numpy array for easy normalization
    flat_values = [cell for row in defpos_matrix[3:] for cell in row[1:] if isinstance(cell, (int, float))]
    location_array = np.array(flat_values)

    # Add properly formatted cells to the table
    for i, row in enumerate(defpos_matrix):
        for j, cell in enumerate(row):

            # Set default cell colors
            if i == 0 or j == 0 or i == 1 or i == 2:
                face_color = 'lightgray'
            else:
                face_color = 'white'

            # Color code shifts by direction, magnitude
            if i > 2 and j != 0 and isinstance(cell, (int, float)):
                cmap = plt.get_cmap('coolwarm')
                if cell == 0:
                    face_color = 'white'  # Set white for zero values
                else:
                    norm_cell_value = (cell - location_array.min()) / (location_array.max() - location_array.min())
                    face_color = cmap(norm_cell_value)

            # Set font weight and cell height for row 1
            font_weight = ["bold" if i == 0 else "normal" for i in range(len(defpos_matrix))]
            col_widths = [0.1 if j == 0 else 0.2 for j in range(len(defpos_matrix[0]))]
            cell_height = [0.05 if i == 0 or i == 1 else 0.1 for i in range(len(defpos_matrix))]

            cell_text = table.add_cell(i, j, width=col_widths[j], height=cell_height[i], text=cell, loc='center', facecolor=face_color)
            cell_text.set_text_props(weight=font_weight[i])

            # Truncate text to fit in the cell
            if isinstance(cell, str):
                renderer = fig.canvas.get_renderer()
                max_width = col_widths[j] * fig.dpi * 9 / 2
                fontproperties = FontProperties(size=14)
                truncated_text = truncate_text(cell, max_width, renderer, fontproperties)
                cell_text.get_text().set_text(truncated_text)

    # Format and save the output
    table.scale(3, 3)
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    ax.add_table(table)
    plt.subplots_adjust(left=0.1, bottom=0.3)
    plt.savefig(output_dir / f"defpos_{num}.pdf")
    plt.close()


def compile_pdf(output_dir, team_code):
    """
    Knits saved defense cards together into one PDF

    :param output_dir: (dir) File where saved defense cards can be found
    :param team_code: (str) Three letter team abbreviation, for titling
    :return: A compiled PDF of the defense cards, saved in the same location
    """
    # Initialize a PdfMerger object
    merger = PdfMerger()

    # Look for files matching the pattern Roster_{i}.pdf
    files_to_merge = [f for f in os.listdir(output_dir) if f.startswith('defpos') and f.endswith('.pdf')]

    # Sort files by the number in the filename
    files_to_merge.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Add each file to the merger
    for file in files_to_merge:
        file_path = os.path.join(output_dir, file)
        merger.append(file_path)

    # Write out the merged PDF
    if files_to_merge:
        merged_pdf_path = os.path.join(output_dir, f'DefPos_{team_code}.pdf')
        with open(merged_pdf_path, 'wb') as merged_pdf:
            merger.write(merged_pdf)

    # Remove the original individual PDFs
    for file in files_to_merge:
        os.remove(os.path.join(output_dir, file))
        print(f"Removed original PDF: {file}")


def main():
    """
    Inputs defensive positioning data, generates defense cards
    """
    # Initialize data file, batter info
    output_dir = "/Users/relliott/Desktop/AAAA"
    team_code = "YHW"

    # Vs. RHP
    batter_dict_r = {
        "Kirkland Banks": ["", "#2 (R)", 0, 0, 0, 0, 1, 1, 1],
        "Braedon Blackford": ["5", "#7 (L)", -1, -1, -1, -1, 1, 1, 1],
        "Edwin DeLaCruz": ["", "#14 (R)", 0, 0, 0, 0, -1, -1, -1],
        "Alejandro Figueredo": ["", "#44 (R)", 0, 0, 0, 0, -2, -2, -2],
        "David Glancy": ["6", "#10 (R)", 0, 0, 0, 0, 0, 0, 0],
        "Jose Gonzalez": ["4", "#23 (L)", 0, 0, 0, 0, 1, 1, 1],
        "Travis Holt": ["", "#9 (R)", 0, 0, 0, 0, 0, 0, 0],
        "Bobby Lada": ["3", "#11 (R)", 1, 1, 0, 1, 0, 0, 0],
        "Taylor Lomack": ["8", "#18 (R)", 0, 0, 0, 0, 0, 0, 0],
        "Braylin Marine": ["2", "#4 (R)", 0, 0, 0, 0, -1, -1, -1],
        "Angel Mendoza": ["9", "#35 (R)", 0, 0, 0, 0, 0, 0, 0],
        "Brayland Skinner": ["1", "#5 (L)", 0, 0, 0, 0, 0, 0, 0],
        "Tanner Smith": ["7", "#31 (L)", 0, 0, 0, 0, 1, 1, 1]
    }

    # Vs. LHP
    batter_dict_l = {
        "Kirkland Banks": ["", "#2 (R)", 0, 0, 0, 0, 2, 2, 2],
        "Braedon Blackford": ["5", "#7 (L)", 0, 0, 0, 0, 0, 0, 0],
        "Edwin DeLaCruz": ["", "#14 (R)", 0, 0, 0, 0, -1, -1, -1],
        "Alejandro Figueredo": ["", "#44 (R)", 0, 0, 0, 0, -2, -2, -2],
        "David Glancy": ["6", "#10 (R)", 0, 0, 0, 0, 2, 2, 2],
        "Jose Gonzalez": ["4", "#23 (L)", 0, 0, 0, 0, 2, 2, 2],
        "Travis Holt": ["", "#9 (R)", 0, 0, 0, 0, 0, 0, 0],
        "Bobby Lada": ["3", "#11 (R)", 2, 2, 1, 2, 1, 1, 1],
        "Taylor Lomack": ["8", "#18 (R)", 0, 0, 0, 0, 0, 0, 0],
        "Braylin Marine": ["2", "#4 (R)", 1, 1, 1, 1, -1, -1, -1],
        "Angel Mendoza": ["9", "#35 (R)", 2, 0, 0, 0, 1, 1, 1],
        "Brayland Skinner": ["1", "#5 (L)", 1, 2, 1, 2, 2, 2, 2],
        "Tanner Smith": ["7", "#31 (L)", 0, 1, 0, 1, 2, 2, 2]
    }

    # Generate cards in desired order
    for n in range(0, 24):
        if n in [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]:
            generate_defpos(batter_dict_r, output_dir, "Right", n)
        elif n in [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 24]:
            generate_defpos(batter_dict_l, output_dir, "Left", n)

    compile_pdf(output_dir, team_code)


if __name__ == "__main__":
    main()

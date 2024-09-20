"""
    Project Title: Ballers Baseball Pitcher Scouting Reports

    Author: Riley Elliott

    Date:   9/20/2024
"""

import os
import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from pathlib import Path
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from PIL import Image

font_properties = {'size': 13}


def fix_name(metrics_list, p_or_b):
    """
    Re-formats pitcher/batter name columns in an input csv file so that they appear in the form
    "First Last"

    :param metrics_list: (list) a list containing the information from one row of an input csv file
    :param p_or_b: (str) either pitcher or batter, to indicate which index to target
    :return: (str) a name in the form "First Last"
    """
    # Store characters to strip
    characters_to_strip = '" '
    new_name = ""

    # Replace the 5th index with reformatted pitcher name
    if p_or_b == "Pitcher":
        stripped_first = metrics_list[6].strip(characters_to_strip)
        stripped_last = metrics_list[5].strip(characters_to_strip)
        new_name = stripped_first + " " + stripped_last
        metrics_list[5] = new_name

    # Replace the 10th index with reformatted batter name
    elif p_or_b == "Batter":
        stripped_first = metrics_list[11].strip(characters_to_strip)
        stripped_last = metrics_list[10].strip(characters_to_strip)
        new_name = stripped_first + " " + stripped_last
        metrics_list[10] = new_name

    return new_name


def abbreviate_pitch(tagged_pitch):
    """
    Abbreviates tagged pitch names so that they can fit in the head-to-head matchups table

    :param tagged_pitch: (str) pitch type, as it appears in the input csv under TaggedPitchType
    :return: (str) an abbreviation of that pitch type
    """
    if tagged_pitch == "Sweeper":
        return "SWP"
    elif tagged_pitch == "FourSeamFastBall" or tagged_pitch == "4-seam":
        return "4S"
    elif tagged_pitch == "TwoSeamFastBall" or tagged_pitch == "2-seam":
        return "2S"
    elif tagged_pitch == "Sinker":
        return "SNK"
    elif tagged_pitch == "Cutter":
        return "CUT"
    elif tagged_pitch == "ChangeUp":
        return "CH"
    elif tagged_pitch == "Splitter":
        return "SPL"
    elif tagged_pitch == "Slider":
        return "SL"
    elif tagged_pitch == "Curveball":
        return "CB"
    elif tagged_pitch == "Total":
        return "Total"


def abbreviate_result(result):
    """
    Abbreviates pitch results so that they can fit in the head-to-head matchups table

    :param result: (str) pitch result, as it appears in the input csv under PitchCall,
    TaggedHitType, or PlayResult
    :return: (str) an abbreviation of that pitch result
    """
    if result == "BallCalled" or result == "BallinDirt" or result == "AutomaticBall":
        return "Ball"
    elif result == "BallIntentional":
        return "IBB"
    elif result == "StrikeCalled" or result == "AutomaticStrike":
        return "CStr"
    elif result == "StrikeSwinging":
        return "SwStr"
    elif result == "FoulBall" or result == "FoulBallNotFieldable" or result == "FoulBallFieldable":
        return "Foul"
    elif result == "HitByPitch":
        return "HBP"
    elif result == "HitByPitch":
        return "HBP"
    elif result == "GroundBall":
        return "GB"
    elif result == "FlyBall":
        return "FB"
    elif result == "LineDrive":
        return "LD"
    elif result == "Popup":
        return "PU"
    elif result == "Bunt":
        return "Bunt"
    elif result == "Out":
        return "Out"
    elif result == "Single":
        return "1B"
    elif result == "Double":
        return "2B"
    elif result == "Triple":
        return "3B"
    elif result == "HomeRun":
        return "HR"
    elif result == "FieldersChoice":
        return "FC"
    elif result == "Error":
        return "ROE"
    elif result == "Sacrifice":
        return "SAC"
    elif result == "Strikeout":
        return "K"
    elif result == "Walk":
        return "BB"
    else:
        return "**ERROR**"


def find_repertoire(data_dict, graph_purpose):
    """
    Parses a data dictionary to note which pitches a pitcher throws. Returns the indices (in the list of values in the
    dictionary) of those pitches.

    :param data_dict: (dict) a dictionary where the keys are pitch types that appear in consistent
    order across all intended graph types
    :param graph_purpose: (str) indicates which graph the data will go into
    :return: (list) a list of indices corresponding to pitch types the given pitcher throws
    """
    # Initialize used indices list
    used_indices = []

    # If graphing location:
    if graph_purpose == "Location":
        location = list(data_dict.values())

        # Pitch must be thrown at least once to qualify. Total is not included
        for i in range(len(location)):
            if i < 9 and len(location[i][0]) > 0:
                used_indices.append(i)

    # If graphing accuracy:
    elif graph_purpose == "Accuracy":
        accuracy = list(data_dict.values())

        # Pitch must be thrown at least once to qualify
        for i in range(len(accuracy)):
            if accuracy[i] != "N/A":
                used_indices.append(i)

    # If graphing batted balls:
    elif graph_purpose == "Batted balls":
        batted_balls = list(data_dict.values())

        # Pitch must be hit at least once to qualify
        for i in range(len(batted_balls)):
            try:
                # Try to convert the batted ball to a float
                float(batted_balls[i])
                # If the conversion succeeds, append the index to used_indices
                used_indices.append(i)
            except ValueError:
                # If the conversion fails, do nothing
                pass

    # If graphing slash line:
    elif graph_purpose == "Slash":
        slash = list(data_dict.values())

        # Pitch must be hit at least once to qualify. Total is not included
        for i in range(len(slash)):
            if slash[i][4] > 0:
                used_indices.append(i)

    return used_indices


def format_decimal(value):
    # Round the value to three decimal places
    rounded_value = round(value, 3)
    # Convert the value to a string formatted to three decimal places
    formatted_value = f"{rounded_value:.3f}"
    # Remove the leading zero if it starts with "0."
    if formatted_value.startswith("0."):
        formatted_value = formatted_value[1:]
    return formatted_value


def safe_division(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0


def parse_movement(filename, pitcher_name):
    """
    Parses a Trackman csv file and collects the pitch metrics of a given pitcher, in visual and table form

    :param filename: (str) The name of a Trackman csv file
    :param pitcher_name: (str) The name of a pitcher
    :return: (list) movement metrics, (matrix) all pitch metrics, (str) pitcher hand
    """
    # Initialize data dictionaries
    movement_coords = {"FourSeamFastBall": [[], []], "TwoSeamFastBall": [[], []],
                       "Sinker": [[], []], "Cutter": [[], []], "ChangeUp": [[], []], "Splitter": [[], []],
                       "Slider": [[], []], "Sweeper": [[], []], "Curveball": [[], []]}
    shape_dict = {
        "FourSeamFastBall": [0, 0, 0, 0, 0, 0],
        "TwoSeamFastBall": [0, 0, 0, 0, 0, 0],
        "Sinker": [0, 0, 0, 0, 0, 0],
        "Cutter": [0, 0, 0, 0, 0, 0],
        "ChangeUp": [0, 0, 0, 0, 0, 0],
        "Splitter": [0, 0, 0, 0, 0, 0],
        "Slider": [0, 0, 0, 0, 0, 0],
        "Sweeper": [0, 0, 0, 0, 0, 0],
        "Curveball": [0, 0, 0, 0, 0, 0]
    }
    percentile_dict = {"FourSeamFastBall": [], "TwoSeamFastBall": [], "Sinker": [], "Cutter": [],
                       "ChangeUp": [], "Splitter": [], "Slider": [], "Sweeper": [], "Curveball": []}
    pitcher_hand = None  # Initialize pitcher_hand
    assigned_pitcher_hand = False  # Flag to track assignment

    file = open(filename, "r")

    # Loop through the file
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Pitcher")

        # Filter for requirements
        if (metrics[5] == pitcher_name and metrics[21] != "" and metrics[30] != ""
                and metrics[33] != "" and metrics[40] != "" and metrics[40] != "" and metrics[41] != ""):

            # Note pitcher hand
            if not assigned_pitcher_hand:
                pitcher_hand = metrics[8]
                assigned_pitcher_hand = True

            # Record metrics
            movement_coords[metrics[21]][0].append(float(metrics[40]))
            movement_coords[metrics[21]][1].append(float(metrics[41]))
            shape_dict[metrics[21]][0] += float(metrics[30])
            percentile_dict[metrics[21]].append(float(metrics[30]))
            shape_dict[metrics[21]][1] += float(metrics[33])
            shape_dict[metrics[21]][2] += float(metrics[40])
            shape_dict[metrics[21]][3] += float(metrics[41])
            shape_dict[metrics[21]][4] += 1

    file.close()

    def format_number(num):
        """Format the number to display as an integer if it ends in .0."""
        if isinstance(num, float) and num.is_integer():
            return int(num)
        return num

    # Turn shape_dict into a table with pitch shape info
    shape_table = [["Pitch", "Velo (mph)", "Spin (rpm)", "IVB (in)", "HB (in)"]]
    for key in shape_dict:
        if shape_dict[key][4] > 0:
            shapes = [abbreviate_pitch(key)]
            for i in range(4):  # Loop over the first four elements
                if i == 1:  # "Spin" is the second element after "Pitch"
                    value = round(shape_dict[key][i] / shape_dict[key][4], 0)
                    value = format_number(value)
                else:
                    value = round(shape_dict[key][i] / shape_dict[key][4], 1)
                shapes.append(value)
            shape_table.append(shapes)

    # Function to append extra columns to matrix
    def append_column(matrix, new_column_values, column_index):
        if len(matrix) != len(new_column_values):
            raise ValueError("The number of rows in the new column must match the number of rows in the matrix.")

        if not (0 <= column_index <= len(matrix[0])):
            raise IndexError("The column index is out of bounds.")

        for i in range(len(matrix)):
            matrix[i].insert(column_index, new_column_values[i])

    # Append 90th percentile velo column
    velo90_column = ["Velo90"]
    for key in percentile_dict:
        if len(percentile_dict[key]) > 0:
            velo90_column.append(str(round(np.percentile(percentile_dict[key], 90), 1)))
    append_column(shape_table, velo90_column, 2)

    # Descriptions of pitch movement
    # descriptions = [
    #     "Descriptions",
    #     "Some sink,\nstraight",
    #     "Power sink,\ngreat run",
    #     "Good depth,\nfade",
    #     "Flat,\ngood cut"
    # ]
    # append_column(shape_table, descriptions, 6)

    return movement_coords, shape_table, pitcher_hand


def generate_movement(movement_coords, shape_table, pitcher_name, pitcher_hand, output_dir):
    """
    Visualizes pitch movement metrics for a given pitcher. Includes expanded metrics in table form

    :param movement_coords: (list) movement metrics
    :param shape_table: (matrix) all pitch metrics
    :param pitcher_name: (str) pitcher name
    :param pitcher_hand: (str) pitcher hand
    :param output_dir: (dir) destination for output
    :return: (pdf) visualization of pitch movement
    """
    # Note which pitches are used
    used_indices = find_repertoire(movement_coords, "Location")

    # Initialize generic and updated labels
    labels = ["FourSeamFastBall", "TwoSeamFastBall", "Sinker", "Cutter",
              "ChangeUp", "Splitter", "Slider", "Sweeper", "Curveball"]
    movement = list(movement_coords.values())
    colors = ["orange", "olive", "cyan", "purple",
              "green", "pink", (0.95, 0.8, 0), (1.0, 0.4, 0.4), (0, 0.3, 1)]
    markers = ["s", "^", "v", "*", "d", "p", "x", "o", "+"]
    new_labels = []
    new_coordinates = []
    new_colors = []
    new_markers = []

    # Only append pitch info to labels if pitch is sufficiently used
    for item in used_indices:
        new_labels.append(abbreviate_pitch(labels[item]))
        new_coordinates.append(movement[item])
        new_colors.append(colors[item])
        new_markers.append(markers[item])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot pitch movement by pitch type
    for i in range(len(new_coordinates)):
        x_coordinates = [value for value in new_coordinates[i][1]]
        y_coordinates = [value for value in new_coordinates[i][0]]
        ax.scatter(x_coordinates, y_coordinates, color=new_colors[i], marker=new_markers[i], label=new_labels[i])

    # Set axis limits and labels
    ax.set_xlim(25, -25)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal', adjustable='box')

    # Add grid with lines over every integer and darker lines on the axes
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1.5)
    ax.axvline(0, color='black', linewidth=1.5)
    ax.set_xticks(range(-25, 26, 5))
    ax.set_yticks(range(-25, 26, 5))

    # Add legend with collected handles and labels
    ax.legend(fontsize='9')
    plt.title(pitcher_name + " Movement Profile (Catcher View)\n\n", fontweight="bold", fontsize=13)

    # Add axis labels to describe movement types
    ax.text(0, 26.5, 'CARRY', ha='center', va='bottom', fontweight='normal')
    ax.text(0, -29, 'DEPTH', ha='center', va='top', fontweight='normal')
    if pitcher_hand == "Right":
        ax.text(32.5, 0, 'RUN', ha='left', va='center', fontweight='normal')
        ax.text(-30.5, 0, 'CUT', ha='right', va='center', fontweight='normal')
    elif pitcher_hand == "Left":
        ax.text(32.5, 0, 'CUT', ha='left', va='center', fontweight='normal')
        ax.text(-30.5, 0, 'RUN', ha='right', va='center', fontweight='normal')
    else:
        pass

    # Add shape table to visual
    table = plt.table(cellText=shape_table, colWidths=[0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], cellLoc='center', loc='bottom',
                      bbox=[-0.05, -0.4, 1.1, 0.25])

    # Add cells to the table
    for i, row in enumerate(shape_table):
        for j, cell in enumerate(row):

            # Set cell color
            light_red = (1.0, 0.6, 0.6)
            light_olive = (0.749, 0.718, 0.42)
            light_orange = (1.0, 0.8, 0.4)
            light_purple = (0.8, 0.6, 0.8)
            light_cyan = (0.8, 1.0, 1.0)

            if i == 0:
                face_color = 'lightgray'
            else:
                face_color = 'white'

            if j == 0 and i != 0:
                if "SWP" in cell:
                    face_color = light_red
                elif "4S" in cell:
                    face_color = light_orange
                elif "2S" in cell:
                    face_color = light_olive
                elif "SNK" in cell:
                    face_color = light_cyan
                elif "CUT" in cell:
                    face_color = light_purple
                elif "CH" in cell:
                    face_color = "lightgreen"
                elif "SPL" in cell:
                    face_color = "lightpink"
                elif "SL" in cell:
                    face_color = "lightyellow"
                elif "CB" in cell:
                    face_color = "lightblue"
                else:
                    face_color = "white"

            # Set text settings
            if i == 0:
                font_weight = 'bold'
            elif j == 0:
                font_weight = 'bold'
            else:
                font_weight = 'normal'
            cell_height = 0.05 if i == 0 else 0.05

            cell_text = table.add_cell(i, j, 0.2, cell_height, text=cell, loc='center', facecolor=face_color)
            cell_text.set_text_props(weight=font_weight)

    # Format and save the plot
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(3, 3)
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.1, bottom=0.3)
    first_last = pitcher_name.split()
    plt.savefig(output_dir / f"{first_last[-1]}_1.pdf")
    plt.close()


def generate_shape_table(shape_table, pitcher_name, output_dir):
    """
    Displays pitch metrics in table form only

    :param shape_table: (matrix) all pitch metrics
    :param pitcher_name: (str) pitcher name
    :param output_dir: (dir) destination for output
    :return: (pdf) pitch metric table
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')
    table = Table(ax, bbox=[0, 0, 1, 1])

    # Add cells to the table
    for i, row in enumerate(shape_table):
        for j, cell in enumerate(row):

            # Set cell color
            light_red = (1.0, 0.6, 0.6)
            light_olive = (0.749, 0.718, 0.42)
            light_orange = (1.0, 0.8, 0.4)
            light_purple = (0.8, 0.6, 0.8)
            light_cyan = (0.8, 1.0, 1.0)

            if i == 0:
                face_color = 'lightgray'
            else:
                face_color = 'white'

            if j == 0 and i != 0:
                if "SWP" in cell:
                    face_color = light_red
                elif "4S" in cell:
                    face_color = light_orange
                elif "2S" in cell:
                    face_color = light_olive
                elif "SNK" in cell:
                    face_color = light_cyan
                elif "CUT" in cell:
                    face_color = light_purple
                elif "CH" in cell:
                    face_color = "lightgreen"
                elif "SPL" in cell:
                    face_color = "lightpink"
                elif "SL" in cell:
                    face_color = "lightyellow"
                elif "CB" in cell:
                    face_color = "lightblue"
                else:
                    face_color = "white"

            # Set text settings
            if i == 0:
                font_weight = 'bold'
            elif j == 0:
                font_weight = 'bold'
            else:
                font_weight = 'normal'
            cell_height = 0.05 if i == 0 else 0.05

            cell_text = table.add_cell(i, j, 0.1, cell_height, text=cell, loc='center', facecolor=face_color)
            cell_text.set_text_props(weight=font_weight)

    # Format and save the plot
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=2.0)
    table.auto_set_font_size(False)
    ax.add_table(table)
    plt.title(pitcher_name + " Movement Profile", fontweight="bold", fontsize=13)
    first_last = pitcher_name.split()
    plt.savefig(output_dir / f"{first_last[-1]}_1.pdf")
    plt.close()


def parse_location(filename, pitcher_name, batter_side):
    """
    Parses a Trackman csv and collects pitch location coordinates by pitch type for a given pitcher.

    :param filename: (str) The title of a Trackman csv
    :param pitcher_name: (str) Pitcher name
    :return: (dict) Pitch location coordinates by pitch type
    """
    # Initialize data dictionary
    location_dict = {"FourSeamFastBall": [[], []], "TwoSeamFastBall": [[], []],
                     "Sinker": [[], []], "Cutter": [[], []], "ChangeUp": [[], []], "Splitter": [[], []],
                     "Slider": [[], []], "Sweeper": [[], []], "Curveball": [[], []]}

    # Loop through the file
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Pitcher")

        # Append locations to dictionary when pitcher and handedness are of interest
        if metrics[5] == pitcher_name and metrics[13] == batter_side and metrics[21] in location_dict.keys() and metrics[42] != "" and metrics[43] != "":
            location_dict[metrics[21]][0].append(float(metrics[43]))
            location_dict[metrics[21]][1].append(float(metrics[42]))

    file.close()

    return location_dict


def generate_location(location_dict, pitcher_name, batter_side, output_dir):
    """
    Generates and saves a pitch location heatmap for each pitch a pitcher throws, vs. RHB and LHB.

    :param location_dict:
    :param pitcher_name: (str) Pitcher name
    :param batter_side: (str) Batter side
    :param output_dir: (str) File location to save output
    :return: (pdf) Heatmaps of pitch locations for different pitches, vs. RHB and LHB
    """
    # Set plot number for output purposes
    if batter_side == "Right":
        plot_num = 20
    else:
        plot_num = 30

    # Create a location plot for each pitch and its corresponding coordinates
    for pitch, coordinates in location_dict.items():
        if coordinates[0] and coordinates[1]:  # Check if there is at least one value in both x and y coordinates
            x_coords = coordinates[0]
            y_coords = coordinates[1]

            # Create a new figure and axis for each pitch type
            fig, ax = plt.subplots(figsize=(5, 5))

            # Generate strike zone
            strike_zone = patches.Rectangle((-(1.37 / 2), 1.7), 1.37, 1.9,
                                            linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(strike_zone)
            buffer_zone = patches.Rectangle((-(1.37 / 2) - 0.166667, 1.7 - 0.166667), 1.37 + 0.333333, 1.9 + 0.333333,
                                           linewidth=1, edgecolor='black', facecolor='none', linestyle='solid')
            ax.add_patch(buffer_zone)

            # Generate zone quadrants
            rows, cols = 3, 3
            small_width = strike_zone.get_width() / cols
            small_height = strike_zone.get_height() / rows
            for i in range(rows):
                for j in range(cols):
                    x = strike_zone.get_x() + j * small_width
                    y = strike_zone.get_y() + i * small_height
                    small_rectangle = patches.Rectangle((x, y), small_width, small_height,
                                                        linewidth=1, edgecolor='black', facecolor='none')
                    ax.add_patch(small_rectangle)

            # Create a 2D histogram (density plot)
            heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=(70, 70))

            # Smooth the heatmap using Gaussian filter
            heatmap = gaussian_filter(heatmap, sigma=3.5)

            img = ax.imshow(heatmap.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                            cmap='coolwarm',
                            alpha=0.5)
            # cbar = fig.colorbar(img, ax=ax)
            # cbar.set_label('Frequency')
            ax.axis('off')

            # Set axis limits and title
            ax.set_xlim(1.3, -1.3)
            ax.set_ylim(1.2, 4.1)
            ax.set_aspect('equal', adjustable='box')
            plt.title(pitcher_name + " " + abbreviate_pitch(pitch) + " Location vs. " + batter_side + "\n(Catcher View)", fontweight="bold",
                      fontsize=13)

            # Save visual
            output_dir = Path(output_dir)  # Convert to Path object
            output_dir.mkdir(parents=True, exist_ok=True)
            first_last = pitcher_name.split()
            plt.savefig(output_dir / f"{first_last[-1]}_{plot_num}.pdf", bbox_inches='tight')
            plt.close()
            plot_num += 1


def parse_usage(filename, pitcher_name, batter_side):
    """
    Parses a Trackman csv to collect data on a pitcher's usage of different pitches in different counts.

    :param filename: (str) The title of a Trackman csv
    :param pitcher_name: (str) Pitcher name
    :param batter_side: (str) Batter side
    :return: (matrix) A matrix with pitch usage data
    """
    # Initialize data recorders
    usage_matrix = [["Count", "FourSeamFastBall", "TwoSeamFastBall", "Sinker", "Cutter", "ChangeUp",
                     "Splitter", "Slider", "Sweeper", "Curveball"], ["Overall", 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ["0-0", 0, 0, 0, 0, 0, 0, 0, 0, 0], ["Hitter's\nCount", 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ["2K", 0, 0, 0, 0, 0, 0, 0, 0, 0], ["3-2", 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    overall_counter = 0
    o_o_counter = 0
    hitters_counter = 0
    two_k_counter = 0
    three_two_counter = 0

    def get_pitch_column_index(usage_matrix, pitch_type):
        """
        Retrieve the column number corresponding to the pitch type.

        :param usage_matrix: List of lists containing usage data.
        :param pitch_type: String indicating the pitch type.
        :return: Integer column index if found, None otherwise.
        """
        if pitch_type in usage_matrix[0]:
            return usage_matrix[0].index(pitch_type)
        else:
            return None

    # Loop through file
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Pitcher")

        # Record pitch usage by count
        if metrics[5] == pitcher_name and metrics[13] == batter_side and get_pitch_column_index(usage_matrix, metrics[21]) is not None:
            if int(metrics[19]) == 0 and int(metrics[20]) == 0:  # 0-0
                overall_counter += 1
                o_o_counter += 1
                usage_matrix[1][get_pitch_column_index(usage_matrix, metrics[21])] += 1
                usage_matrix[2][get_pitch_column_index(usage_matrix, metrics[21])] += 1
            elif int(metrics[19]) > int(metrics[20]) and int(metrics[19]) > 1 and int(metrics[20]) < 2:  # Hitter's counts
                overall_counter += 1
                hitters_counter += 1
                usage_matrix[1][get_pitch_column_index(usage_matrix, metrics[21])] += 1
                usage_matrix[3][get_pitch_column_index(usage_matrix, metrics[21])] += 1
            elif int(metrics[20]) == 2 and int(metrics[19]) < 3:  # 2 strikes
                overall_counter += 1
                two_k_counter += 1
                usage_matrix[1][get_pitch_column_index(usage_matrix, metrics[21])] += 1
                usage_matrix[4][get_pitch_column_index(usage_matrix, metrics[21])] += 1
            elif int(metrics[19]) == 3 and int(metrics[20]) == 2:  # 3-2 counts
                overall_counter += 1
                three_two_counter += 1
                usage_matrix[1][get_pitch_column_index(usage_matrix, metrics[21])] += 1
                usage_matrix[5][get_pitch_column_index(usage_matrix, metrics[21])] += 1

    file.close()

    # Abbreviate pitch type names
    for i in range(1, len(usage_matrix[0])):  # Start from 1 to exclude "Count"
        usage_matrix[0][i] = abbreviate_pitch(usage_matrix[0][i])

    # Identify pitches that are not thrown
    columns_to_keep = []
    for col in range(1, len(usage_matrix[0])):  # Start from 1 to exclude "Count"
        if any(row[col] != 0 for row in usage_matrix[1:]):  # Check if any value is non-zero
            columns_to_keep.append(col)

    # Include the "Count" column (index 0)
    columns_to_keep.insert(0, 0)

    # Create a new matrix excluding pitches that are not thrown
    new_usage_matrix = [[row[col] for col in columns_to_keep] for row in usage_matrix]

    # Compile a list of sample size counters for easy access
    counters = [overall_counter, o_o_counter, hitters_counter, two_k_counter, three_two_counter]

    # Function to calculate usage percentages
    def calculate_usage_percentages(matrix, counters):
        for i in range(1, len(matrix)):
            if counters[i - 1] == 0:  # Avoid division by zero
                continue
            for j in range(1, len(matrix[i])):
                matrix[i][j] = round((matrix[i][j] / counters[i - 1]) * 100, 1)
        return matrix

    # Calculate usage percentages
    new_usage_matrix = calculate_usage_percentages(new_usage_matrix, counters)

    return new_usage_matrix


def generate_usage(usage_matrix, pitcher_name, batter_side, output_dir):
    """
    Generates and saves tables depicting pitch usage rates across different counts, vs. RHB and LHB.

    :param usage_matrix: (matrix) A matrix with pitch usage data
    :param pitcher_name: (str) Pitcher name
    :param batter_side: (str) Batter side
    :param output_dir: (str) File location to save output
    :return: (pdf) Tables depicting pitch usage rates across different counts
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('off')
    table = Table(ax, bbox=[0, 0, 1, 1])

    # Add properly formatted cells to the table
    for i, row in enumerate(usage_matrix):
        for j, cell in enumerate(row):

            # Set cell color
            light_red = (1.0, 0.6, 0.6)
            light_olive = (0.749, 0.718, 0.42)
            light_orange = (1.0, 0.8, 0.4)
            light_purple = (0.8, 0.6, 0.8)
            light_cyan = (0.8, 1.0, 1.0)

            if j == 0:
                face_color = 'lightgray'
            else:
                face_color = 'white'

            if i == 0 and j > 0:
                if "SWP" in cell:
                    face_color = light_red
                elif "4S" in cell:
                    face_color = light_orange
                elif "2S" in cell:
                    face_color = light_olive
                elif "SNK" in cell:
                    face_color = light_cyan
                elif "CUT" in cell:
                    face_color = light_purple
                elif "CH" in cell:
                    face_color = "lightgreen"
                elif "SPL" in cell:
                    face_color = "lightpink"
                elif "SL" in cell:
                    face_color = "lightyellow"
                elif "CB" in cell:
                    face_color = "lightblue"
                else:
                    face_color = "white"

            # Set text settings
            if i == 0:
                font_weight = 'bold'
            elif j == 0:
                font_weight = 'bold'
            else:
                font_weight = 'normal'

            cell_height = 0.03 if i == 0 else 0.04
            cell_text = table.add_cell(i, j, 0.1, cell_height, text=cell, loc='center', facecolor=face_color)
            cell_text.set_text_props(weight=font_weight)

    # Save the plot with appropriate formatting and title
    plt.tight_layout(pad=2.0)
    table.auto_set_font_size(False)
    ax.add_table(table)
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    first_last = pitcher_name.split()
    if batter_side == "Right":
        plt.title(pitcher_name + " Usage by Count vs. " + batter_side,
                  fontweight="bold", fontsize=font_properties['size'] + 1)
        plt.savefig(output_dir / f"{first_last[-1]}_4.pdf")
    else:
        plt.title(pitcher_name + " Usage by Count vs. " + batter_side,
                  fontweight="bold", fontsize=font_properties['size'] + 1)
        plt.savefig(output_dir / f"{first_last[-1]}_5.pdf")
    plt.close()


def parse_slash(filename, pitcher_name):
    """
    Parses a Trackman csv and calculate slash lines by pitch type for a given pitcher against RHB and LHB.

    :param filename: (str) Name of Trackman csv
    :param pitcher_name: (str) Pitcher name
    :return: (matrix) A matrix containing slash lines by pitch type and batter side
    """
    # Initialize data dictionaries
    right_dict = {"FourSeamFastBall": [0, 0, 0, 0, 0], "TwoSeamFastBall": [0, 0, 0, 0, 0],
                  "Sinker": [0, 0, 0, 0, 0], "Cutter": [0, 0, 0, 0, 0], "ChangeUp": [0, 0, 0, 0, 0],
                  "Splitter": [0, 0, 0, 0, 0], "Slider": [0, 0, 0, 0, 0], "Sweeper": [0, 0, 0, 0, 0], "Curveball": [0, 0, 0, 0, 0]}
    left_dict = {"FourSeamFastBall": [0, 0, 0, 0, 0], "TwoSeamFastBall": [0, 0, 0, 0, 0],
                 "Sinker": [0, 0, 0, 0, 0], "Cutter": [0, 0, 0, 0, 0], "ChangeUp": [0, 0, 0, 0, 0],
                 "Splitter": [0, 0, 0, 0, 0], "Slider": [0, 0, 0, 0, 0], "Sweeper": [0, 0, 0, 0, 0],
                 "Curveball": [0, 0, 0, 0, 0]}
    r_overall = [0, 0, 0, 0, 0]
    l_overall = [0, 0, 0, 0, 0]

    # Loop through the file
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Pitcher")

        # Record plate appearance data
        if metrics[5] == pitcher_name and metrics[21] in right_dict.keys():

            if metrics[13] == "Right":  # Vs. RHB

                if metrics[23] == "InPlay":
                    right_dict[metrics[21]][4] += 1
                    if metrics[26] != "Sacrifice":
                        right_dict[metrics[21]][3] += 1

                    if metrics[26] == "Single":
                        right_dict[metrics[21]][0] += 1
                        right_dict[metrics[21]][1] += 1
                    elif metrics[26] == "Double":
                        right_dict[metrics[21]][0] += 1
                        right_dict[metrics[21]][1] += 2
                    elif metrics[26] == "Triple":
                        right_dict[metrics[21]][0] += 1
                        right_dict[metrics[21]][1] += 3
                    elif metrics[26] == "HomeRun":
                        right_dict[metrics[21]][0] += 1
                        right_dict[metrics[21]][1] += 4

                if metrics[24] == "Strikeout":
                    right_dict[metrics[21]][4] += 1
                    right_dict[metrics[21]][3] += 1

                if metrics[24] == "Walk" or metrics[23] == "HitByPitch":
                    right_dict[metrics[21]][4] += 1
                    right_dict[metrics[21]][2] += 1

            else:  # Vs. LHB

                if metrics[23] == "InPlay":
                    left_dict[metrics[21]][4] += 1
                    if metrics[26] != "Sacrifice":
                        left_dict[metrics[21]][3] += 1

                    if metrics[26] == "Single":
                        left_dict[metrics[21]][0] += 1
                        left_dict[metrics[21]][1] += 1
                    elif metrics[26] == "Double":
                        left_dict[metrics[21]][0] += 1
                        left_dict[metrics[21]][1] += 2
                    elif metrics[26] == "Triple":
                        left_dict[metrics[21]][0] += 1
                        left_dict[metrics[21]][1] += 3
                    elif metrics[26] == "HomeRun":
                        left_dict[metrics[21]][0] += 1
                        left_dict[metrics[21]][1] += 4

                if metrics[24] == "Strikeout":
                    left_dict[metrics[21]][4] += 1
                    left_dict[metrics[21]][3] += 1

                if metrics[24] == "Walk" or metrics[23] == "HitByPitch":
                    left_dict[metrics[21]][4] += 1
                    left_dict[metrics[21]][2] += 1

    file.close()

    # Determine which pitches a pitcher throws
    used_indices = []
    used_right = find_repertoire(right_dict, "Slash")
    used_left = find_repertoire(left_dict, "Slash")
    for used in [used_right, used_left]:
        for item in used:
            if item not in used_indices:
                used_indices.append(item)

    # List data for reformatting
    pitches = list(right_dict.keys())
    right = list(right_dict.values())
    left = list(left_dict.values())

    # Calculate slash lines. For each pitch type, return {Slash, PA sample size} vs. R and L
    for slash in [right, left]:
        for i in range(len(slash)):
            item = slash[i]
            formatted_string = (format_decimal(safe_division(item[0], item[3])) + " / "
                                + format_decimal(safe_division((item[0] + item[2]), item[4])) + " / "
                                + format_decimal(safe_division((item[1]), item[3])))
            slash[i] = [formatted_string, str(item[4])]

    # Initialize slash matrix, input data
    slash_matrix = [["", "Slash vs. R", "PA", "Slash vs. L", "PA"]]
    for item in used_indices:
        row = [abbreviate_pitch(pitches[item]), right[item][0], right[item][1], left[item][0], left[item][1]]
        slash_matrix.append(row)

    # Calculate overall slash lines
    for dict in [right_dict, left_dict]:
        if dict == right_dict:
            for i in range(len(r_overall)):
                r_overall[i] = sum([dict[key][i] for key in dict])
        elif dict == left_dict:
            for i in range(len(l_overall)):
                l_overall[i] = sum([dict[key][i] for key in dict])

    # Calculate overall slash lines. Return {Slash, PA sample size} vs. R and L
    for overall in [r_overall, l_overall]:
        formatted_string = (format_decimal(safe_division(overall[0], overall[3])) + " / "
                            + format_decimal(safe_division((overall[0] + overall[2]), overall[4])) + " / "
                            + format_decimal(safe_division((overall[1]), overall[3])))
        overall[:] = [formatted_string, str(overall[4])]

    # Append last row to matrix
    row = ["Total", r_overall[0], r_overall[1], l_overall[0], l_overall[1]]
    slash_matrix.append(row)

    return slash_matrix


def generate_slash(slash_matrix, pitcher_name, output_dir):
    """
    Generates and saves slash lines by pitch type vs. RHB and LHB for a given pitcher.

    :param slash_matrix: (matrix) A matrix containing slash lines by pitch type and batter side
    :param pitcher_name: (str) Pitcher name
    :param output_dir: (str) File location for saving output
    :return: (pdf) Saves slash lines vs. RHB and LHB
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.axis('off')
    table = Table(ax, bbox=[0, 0, 1, 1])
    col_widths = [0.1, 0.1, 0.05, 0.1, 0.05]

    # Add properly formatted cells to the table
    for i, row in enumerate(slash_matrix):
        for j, cell in enumerate(row):

            # Set cell color
            light_red = (1.0, 0.6, 0.6)
            light_olive = (0.749, 0.718, 0.42)
            light_orange = (1.0, 0.8, 0.4)
            light_purple = (0.8, 0.6, 0.8)
            light_cyan = (0.8, 1.0, 1.0)

            if i == 0:
                face_color = 'lightgray'
            else:
                face_color = 'white'

            if j == 0 and i != 0:
                if "SWP" in cell:
                    face_color = light_red
                elif "4S" in cell:
                    face_color = light_orange
                elif "2S" in cell:
                    face_color = light_olive
                elif "SNK" in cell:
                    face_color = light_cyan
                elif "CUT" in cell:
                    face_color = light_purple
                elif "CH" in cell:
                    face_color = "lightgreen"
                elif "SPL" in cell:
                    face_color = "lightpink"
                elif "SL" in cell:
                    face_color = "lightyellow"
                elif "CB" in cell:
                    face_color = "lightblue"
                elif "Total" in cell:
                    face_color = "lightgray"
                else:
                    face_color = "white"

            # Set text settings
            if i == 0:
                font_weight = 'bold'
            elif j == 0:
                font_weight = 'bold'
            else:
                font_weight = 'normal'
            cell_height = 0.02 if i == 0 else 0.04

            cell_text = table.add_cell(i, j, col_widths[j % len(col_widths)], cell_height, text=cell, loc='center', facecolor=face_color)
            cell_text.set_text_props(weight=font_weight)

    # Save the plot with appropriate formatting and title
    plt.tight_layout(pad=2.0)
    table.auto_set_font_size(False)
    ax.add_table(table)
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    first_last = pitcher_name.split()
    plt.title(pitcher_name + " Slash Lines", fontweight="bold", fontsize=font_properties['size'] + 1)
    plt.savefig(output_dir / f"{first_last[-1]}_3.pdf")
    plt.close()


def parse_zone_control(filename, pitcher_name, batter_side):
    """
    Parses a Trackman csv file and collects pitch effectiveness metrics for a given pitcher.

    :param filename: (str) Name of a Trackman csv
    :param pitcher_name: (str) Pitcher name
    :param batter_side: (str) Batter side
    :return: (matrix) Table with pitch effectiveness metrics
    """
    # Initialize data dictionaries
    zone_dict = {"FourSeamFastBall": [0, 0], "TwoSeamFastBall": [0, 0], "Sinker": [0, 0],
                 "Cutter": [0, 0], "ChangeUp": [0, 0], "Splitter": [0, 0], "Slider": [0, 0], "Sweeper": [0, 0], "Curveball": [0, 0],
                 "Total": [0, 0]}
    whiff_dict = {"FourSeamFastBall": [0, 0], "TwoSeamFastBall": [0, 0], "Sinker": [0, 0],
                  "Cutter": [0, 0], "ChangeUp": [0, 0], "Splitter": [0, 0], "Slider": [0, 0], "Sweeper": [0, 0], "Curveball": [0, 0],
                  "Total": [0, 0]}
    chase_dict = {"FourSeamFastBall": [0, 0], "TwoSeamFastBall": [0, 0], "Sinker": [0, 0],
                  "Cutter": [0, 0], "ChangeUp": [0, 0], "Splitter": [0, 0], "Slider": [0, 0], "Sweeper": [0, 0], "Curveball": [0, 0],
                  "Total": [0, 0]}
    k_dict = {"FourSeamFastBall": [0, 0], "TwoSeamFastBall": [0, 0], "Sinker": [0, 0],
              "Cutter": [0, 0], "ChangeUp": [0, 0], "Splitter": [0, 0], "Slider": [0, 0], "Sweeper": [0, 0], "Curveball": [0, 0],
              "Total": [0, 0]}
    bb_dict = {"FourSeamFastBall": [0, 0], "TwoSeamFastBall": [0, 0], "Sinker": [0, 0],
               "Cutter": [0, 0], "ChangeUp": [0, 0], "Splitter": [0, 0], "Slider": [0, 0], "Sweeper": [0, 0], "Curveball": [0, 0],
               "Total": [0, 0]}

    # Loop through the file
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Pitcher")

        if metrics[5] == pitcher_name and metrics[21] in zone_dict.keys() and metrics[13] == batter_side:

            # Zone and chase rates
            if metrics[42] != "" and metrics[43] != "":
                if -(1.70333 / 2) <= float(metrics[43]) <= (1.70333 / 2) and 1.5333 <= float(metrics[42]) <= 3.7666:  # Pitches in zone
                    zone_dict[metrics[21]][0] += 1
                    zone_dict[metrics[21]][1] += 1
                    zone_dict["Total"][0] += 1
                    zone_dict["Total"][1] += 1
                else:  # Pitches out of zone
                    zone_dict[metrics[21]][1] += 1
                    zone_dict["Total"][1] += 1
                    if (metrics[23] == "StrikeSwinging" or metrics[23] == "FoulBall" or metrics[23] ==
                            "FoulBallNotFieldable" or metrics[23] == "FoulBallFieldable" or metrics[23] == "InPlay"):
                        chase_dict[metrics[21]][0] += 1
                        chase_dict[metrics[21]][1] += 1
                        chase_dict["Total"][0] += 1
                        chase_dict["Total"][1] += 1
                    else:
                        chase_dict[metrics[21]][1] += 1
                        chase_dict["Total"][1] += 1

            # Whiffs
            if metrics[23] == "StrikeSwinging":
                whiff_dict[metrics[21]][0] += 1
                whiff_dict[metrics[21]][1] += 1
                whiff_dict["Total"][0] += 1
                whiff_dict["Total"][1] += 1
            elif (metrics[23] == "FoulBall" or metrics[23] == "FoulBallNotFieldable"
                  or metrics[23] == "FoulBallFieldable" or metrics[23] == "InPlay"):
                whiff_dict[metrics[21]][1] += 1
                whiff_dict["Total"][1] += 1

            # K/BB
            if metrics[23] == "InPlay":
                k_dict[metrics[21]][1] += 1
                k_dict["Total"][1] += 1
                bb_dict[metrics[21]][1] += 1
                bb_dict["Total"][1] += 1
            elif metrics[24] == "Strikeout":
                k_dict[metrics[21]][0] += 1
                k_dict["Total"][0] += 1
                k_dict[metrics[21]][1] += 1
                k_dict["Total"][1] += 1
                bb_dict[metrics[21]][1] += 1
                bb_dict["Total"][1] += 1
            elif metrics[24] == "Walk":
                bb_dict[metrics[21]][0] += 1
                bb_dict["Total"][0] += 1
                bb_dict[metrics[21]][1] += 1
                bb_dict["Total"][1] += 1
                k_dict[metrics[21]][1] += 1
                k_dict["Total"][1] += 1

    file.close()

    # Process data for displaying/take averages
    dicts = [zone_dict, whiff_dict, chase_dict, k_dict, bb_dict]
    for pitch_dict in dicts:
        for key in pitch_dict.keys():
            if pitch_dict[key][1] > 0:
                pitch_dict[key] = round(100 * (pitch_dict[key][0] / pitch_dict[key][1]), 1)
            else:
                pitch_dict[key] = "N/A"

    # Note which pitches are used
    used_indices = find_repertoire(zone_dict, "Accuracy")

    # List data for reformatting
    pitches = list(zone_dict.keys())
    strikes = list(zone_dict.values())
    whiffs = list(whiff_dict.values())
    chases = list(chase_dict.values())
    ks = list(k_dict.values())
    bbs = list(bb_dict.values())

    # Convert to matrix
    zone_matrix = [["", "Zone%", "Whiff%", "Chase%", "K%", "BB%"]]
    for item in used_indices:
        row = [abbreviate_pitch(pitches[item]), strikes[item], whiffs[item], chases[item], ks[item], bbs[item]]
        zone_matrix.append(row)

    return zone_matrix


def generate_zone_control(zone_matrix, pitcher_name, batter_side, output_dir):
    """
    Generates and saves a table with pitch effectiveness metrics for a given pitcher.

    :param zone_matrix: (matrix) Matrix with pitch effectiveness metrics
    :param pitcher_name: (str) Pitcher name
    :param batter_side: (str) Batter side
    :param output_dir: (str) Location for saving output
    :return: (pdf) Table with pitch effectiveness metrics
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axis('off')
    table = Table(ax, bbox=[0, 0, 1, 1])

    # Add properly formatted cells to the table
    for i, row in enumerate(zone_matrix):
        for j, cell in enumerate(row):

            # Set cell color
            light_red = (1.0, 0.6, 0.6)
            light_olive = (0.749, 0.718, 0.42)
            light_orange = (1.0, 0.8, 0.4)
            light_purple = (0.8, 0.6, 0.8)
            light_cyan = (0.8, 1.0, 1.0)

            if i == 0:
                face_color = 'lightgray'
            else:
                face_color = 'white'

            if j == 0 and i != 0:
                if "SWP" in cell:
                    face_color = light_red
                elif "4S" in cell:
                    face_color = light_orange
                elif "2S" in cell:
                    face_color = light_olive
                elif "SNK" in cell:
                    face_color = light_cyan
                elif "CUT" in cell:
                    face_color = light_purple
                elif "CH" in cell:
                    face_color = "lightgreen"
                elif "SPL" in cell:
                    face_color = "lightpink"
                elif "SL" in cell:
                    face_color = "lightyellow"
                elif "CB" in cell:
                    face_color = "lightblue"
                elif "Total" in cell:
                    face_color = "lightgray"
                else:
                    face_color = "white"

            # Set text settings
            if i == 0:
                font_weight = 'bold'
            elif j == 0:
                font_weight = 'bold'
            else:
                font_weight = 'normal'
            cell_height = 0.02 if i == 0 else 0.04

            cell_text = table.add_cell(i, j, 0.1, cell_height, text=cell, loc='center', facecolor=face_color)
            cell_text.set_text_props(weight=font_weight)

    # Save the plot with appropriate formatting and title
    plt.tight_layout(pad=2.0)
    table.auto_set_font_size(False)
    ax.add_table(table)
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    first_last = pitcher_name.split()
    if batter_side == "Right":
        plt.title(pitcher_name + " Zone Control vs. " + batter_side, fontweight="bold", fontsize=font_properties['size'] + 1)
        plt.savefig(output_dir / f"{first_last[-1]}_40.pdf")
    else:
        plt.title(pitcher_name + " Zone Control vs. " + batter_side, fontweight="bold", fontsize=font_properties['size'] + 1)
        plt.savefig(output_dir / f"{first_last[-1]}_41.pdf")
    plt.close()


def parse_batted_balls(filename, pitcher_name, batter_side):
    """
    Parses a Trackman csv to calculate batted ball metrics by pitch type and batter side for a given pitcher.

    :param filename: (str) Name of Trackman csv
    :param pitcher_name: (str) Pitcher name
    :param batter_side: (str) Batter side
    :return: (matrix) Matrix with batted ball metrics
    """
    # Initialize data dictionaries
    ev_dict = {"FourSeamFastBall": [], "TwoSeamFastBall": [], "Sinker": [],
               "Cutter": [], "ChangeUp": [], "Splitter": [], "Slider": [], "Sweeper": [], "Curveball": [],
               "Total": 0}
    la_dict = {"FourSeamFastBall": [], "TwoSeamFastBall": [], "Sinker": [],
               "Cutter": [], "ChangeUp": [], "Splitter": [], "Slider": [], "Sweeper": [], "Curveball": [],
               "Total": 0}
    # hh_dict = {"FourSeamFastBall": 0, "TwoSeamFastBall": 0, "Sinker": 0,
    #            "Cutter": 0, "ChangeUp": 0, "Splitter": 0, "Sweeper": 0, "Slider": 0, "Curveball": 0,
    #            "Total": 0}
    # swsp_dict = {"FourSeamFastBall": 0, "TwoSeamFastBall": 0, "Sinker": 0,
    #              "Cutter": 0, "ChangeUp": 0, "Splitter": 0, "Sweeper": 0, "Slider": 0, "Curveball": 0,
    #              "Total": 0}
    gb_dict = {"FourSeamFastBall": 0, "TwoSeamFastBall": 0, "Sinker": 0,
               "Cutter": 0, "ChangeUp": 0, "Splitter": 0, "Sweeper": 0, "Slider": 0, "Curveball": 0,
               "Total": 0}
    ld_dict = {"FourSeamFastBall": 0, "TwoSeamFastBall": 0, "Sinker": 0,
               "Cutter": 0, "ChangeUp": 0, "Splitter": 0, "Sweeper": 0, "Slider": 0, "Curveball": 0,
               "Total": 0}
    fb_dict = {"FourSeamFastBall": 0, "TwoSeamFastBall": 0, "Sinker": 0,
               "Cutter": 0, "ChangeUp": 0, "Splitter": 0, "Sweeper": 0, "Slider": 0, "Curveball": 0,
               "Total": 0}
    total_bbe = {"FourSeamFastBall": 0, "TwoSeamFastBall": 0, "Sinker": 0,
                 "Cutter": 0, "ChangeUp": 0, "Splitter": 0, "Sweeper": 0, "Slider": 0, "Curveball": 0,
                 "Total": 0}

    file = open(filename, "r")

    # Loop through file
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Pitcher")

        # Collect data
        if (metrics[5] == pitcher_name and metrics[13] == batter_side and metrics[21] in ev_dict.keys()
                and metrics[23] == "InPlay" and metrics[48] != "" and metrics[49] != ""):
            ev_dict[metrics[21]].append(float(metrics[48]))
            ev_dict["Total"] += float(metrics[48])
            la_dict[metrics[21]].append(float(metrics[49]))
            la_dict["Total"] += float(metrics[49])
            total_bbe[metrics[21]] += 1
            total_bbe["Total"] += 1
            if metrics[25] == "GroundBall":
                gb_dict[metrics[21]] += 1
                gb_dict["Total"] += 1
            elif metrics[25] == "LineDrive":
                ld_dict[metrics[21]] += 1
                ld_dict["Total"] += 1
            elif metrics[25] == "FlyBall":
                fb_dict[metrics[21]] += 1
                fb_dict["Total"] += 1

    file.close()

    # Process data for displaying/take averages

    for key in ev_dict:
        hard_hit = 0
        total_ev = 0
        if key != "Total":
            for ev in ev_dict[key]:
                total_ev += ev
                if ev >= 95:
                    hard_hit += 1
            ev_dict[key] = total_ev
            # hh_dict[key] += hard_hit
            # hh_dict["Total"] += hard_hit

    for key in la_dict:
        sw_sp = 0
        total_la = 0
        if key != "Total":
            for la in la_dict[key]:
                total_la += la
                if 8 <= la <= 32:
                    sw_sp += 1
            la_dict[key] = total_la
            # swsp_dict[key] += sw_sp
            # swsp_dict["Total"] += sw_sp

    for dict in [ev_dict, la_dict]:
        for key in dict:
            if total_bbe[key] > 0:
                dict[key] = round(dict[key] / total_bbe[key], 1)
            else:
                dict[key] = "-"

    for dict in [gb_dict, ld_dict, fb_dict]:
        for key in dict:
            if total_bbe[key] > 0:
                dict[key] = round(100*float(dict[key] / total_bbe[key]), 1)
            else:
                dict[key] = "-"

    # List data for reformatting
    used_indices = find_repertoire(ev_dict, "Batted balls")
    pitches = list(ev_dict.keys())
    evs = list(ev_dict.values())
    las = list(la_dict.values())
    # hh = list(hh_dict.values())
    # swsp = list(swsp_dict.values())
    grounders = list(gb_dict.values())
    liners = list(ld_dict.values())
    flyballs = list(fb_dict.values())

    # Convert to matrix
    batted_ball_matrix = [["", "Avg. EV\n(mph)", "Avg. LA\n(°)", "GB%", "LD%", "FB%"]]
    for item in used_indices:
        row = [abbreviate_pitch(pitches[item]), evs[item], las[item], grounders[item], liners[item], flyballs[item]]
        batted_ball_matrix.append(row)

    return batted_ball_matrix


def generate_batted_balls(batted_ball_matrix, pitcher_name, batter_side, output_dir):
    """
    Generates and saves a table with batted ball metrics by pitch types/batter side for a given pitcher.

    :param batted_ball_matrix: (matrix) Matrix with batted ball metrics
    :param pitcher_name: (str) Pitcher name
    :param batter_side: (str) Batter side
    :param output_dir: (str) File location for saving output
    :return: (pdf) Saves batted ball metrics tables
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis('off')
    table = Table(ax, bbox=[0, 0, 1, 1])

    # Add properly formatted cells to the table
    for i, row in enumerate(batted_ball_matrix):
        for j, cell in enumerate(row):

            # Set cell color
            light_red = (1.0, 0.6, 0.6)
            light_olive = (0.749, 0.718, 0.42)
            light_orange = (1.0, 0.8, 0.4)
            light_purple = (0.8, 0.6, 0.8)
            light_cyan = (0.8, 1.0, 1.0)

            if i == 0:
                face_color = 'lightgray'

            elif j == 0 and i != 0:
                if "SWP" in cell:
                    face_color = light_red
                elif "4S" in cell:
                    face_color = light_orange
                elif "2S" in cell:
                    face_color = light_olive
                elif "SNK" in cell:
                    face_color = light_cyan
                elif "CUT" in cell:
                    face_color = light_purple
                elif "CH" in cell:
                    face_color = "lightgreen"
                elif "SPL" in cell:
                    face_color = "lightpink"
                elif "SL" in cell:
                    face_color = "lightyellow"
                elif "CB" in cell:
                    face_color = "lightblue"
                elif "Total" in cell:
                    face_color = "lightgray"
            else:
                face_color = "white"

            # Set text settings
            if i == 0:
                font_weight = 'bold'
            elif j == 0:
                font_weight = 'bold'
            else:
                font_weight = 'normal'
            cell_height = 0.02 if i == 0 else 0.03

            cell_text = table.add_cell(i, j, 0.1, cell_height, text=cell, loc='center', facecolor=face_color)
            cell_text.set_text_props(weight=font_weight)

    # Save the plot with appropriate formatting and title
    plt.tight_layout(pad=2.0)
    table.auto_set_font_size(False)
    ax.add_table(table)
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    first_last = pitcher_name.split()
    if batter_side == "Right":
        plt.title(pitcher_name + " Batted Ball Profile vs. " + batter_side, fontweight="bold", fontsize=font_properties['size'] + 1)
        plt.savefig(output_dir / f"{first_last[-1]}_50.pdf")
    else:
        plt.title(pitcher_name + " Batted Ball Profile vs. " + batter_side, fontweight="bold", fontsize=font_properties['size'] + 1)
        plt.savefig(output_dir / f"{first_last[-1]}_51.pdf")
    plt.close()


def parse_release(filename, pitcher_name):
    """
    Parses a Trackman csv and collects data on a given pitcher's release point by pitch type.

    :param filename: (str) Name of Trackman csv
    :param pitcher_name: (str) Pitcher name
    :return: A list of release point coordinates and a matrix with release point averages
    """
    # Initialize data dictionaries
    release_coords = {"FourSeamFastBall": [[], []], "TwoSeamFastBall": [[], []],
                      "Sinker": [[], []], "Cutter": [[], []], "ChangeUp": [[], []], "Splitter": [[], []],
                      "Slider": [[], []], "Sweeper": [[], []], "Curveball": [[], []]}
    metric_dict = {"FourSeamFastBall": [0, 0, 0], "TwoSeamFastBall": [0, 0, 0],
                   "Sinker": [0, 0, 0], "Cutter": [0, 0, 0], "ChangeUp": [0, 0, 0],
                   "Splitter": [0, 0, 0], "Slider": [0, 0, 0], "Sweeper": [0, 0, 0], "Curveball": [0, 0, 0]}

    # Loop through the file
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Pitcher")

        # Record data
        if (metrics[5] == pitcher_name and metrics[21] != "Undefined" and metrics[21] != "" and metrics[36] != ""
                and metrics[37] != "" and metrics[36] != "" and metrics[36] != "" and metrics[37] != ""):
            release_coords[metrics[21]][0].append(float(metrics[36]))
            release_coords[metrics[21]][1].append(float(metrics[37]))
            metric_dict[metrics[21]][0] += float(metrics[36])
            metric_dict[metrics[21]][1] += float(metrics[37])
            metric_dict[metrics[21]][2] += 1

    file.close()

    # Convert to matrix
    metric_table = [["Pitch", "RHeight (ft)", "RSide (ft)"]]
    for key in metric_dict:
        if metric_dict[key][2] > 0:
            metrics = [abbreviate_pitch(key)]
            for i in range(2):  # Loop over the first two elements
                value = round(metric_dict[key][i] / metric_dict[key][2], 1)
                metrics.append(value)
            metric_table.append(metrics)

    return release_coords, metric_table


def generate_release(release_coords, shape_table, pitcher_name, output_dir):
    """
    Generates and saves a plot of a pitcher's release point by pitch type.

    :param release_coords: (list) A list of release point coordinates
    :param shape_table: (matrix) A matrix with release point averages
    :param pitcher_name: (str) Pitcher name
    :param output_dir: (str) File location for saving output
    :return: (pdf) A plot of a pitcher's release point by pitch type
    """
    # Note which pitches are used
    used_indices = find_repertoire(release_coords, "Location")

    # Initialize generic and updated labels
    labels = ["FourSeamFastBall", "TwoSeamFastBall", "Sinker", "Cutter",
              "ChangeUp", "Splitter", "Slider", "Sweeper", "Curveball"]
    release = list(release_coords.values())
    colors = ["orange", "olive", "cyan", "purple",
              "green", "pink", (0.95, 0.8, 0), (1.0, 0.4, 0.4), (0, 0.3, 1)]
    markers = ["s", "^", "v", "*", "d", "p", "x", "o", "+"]
    new_labels = []
    new_coordinates = []
    new_colors = []
    new_markers = []

    # Only append pitch info to labels if pitch is sufficiently used
    for item in used_indices:
        new_labels.append(abbreviate_pitch(labels[item]))
        new_coordinates.append(release[item])
        new_colors.append(colors[item])
        new_markers.append(markers[item])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot release point by pitch type
    for i in range(len(new_coordinates)):
        x_coordinates = [value for value in new_coordinates[i][1]]
        y_coordinates = [value for value in new_coordinates[i][0]]
        ax.scatter(x_coordinates, y_coordinates, color=new_colors[i], marker=new_markers[i], label=new_labels[i])

    # Set axis limits and labels
    ax.set_xlim(4, -4)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal', adjustable='box')

    # Add grid with lines over every integer and darker lines on the axes
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1.5)
    ax.axvline(0, color='black', linewidth=1.5)
    # ax.set_xticks(range(-5, 6, 1))
    # ax.set_yticks(range(0, 8, 1))

    # Add legend with collected handles and labels
    ax.legend(fontsize='9')
    ax.set_xlabel("Release Side (ft, 0=middle of rubber)")
    ax.set_ylabel("Release Height (ft)")

    # Add release metrics table to the visual
    table = plt.table(cellText=shape_table, colWidths=[0.5, 0.5, 0.5], cellLoc='center', loc='bottom',
                      bbox=[0, -0.45, 1, 0.3])

    # Add cells to table, format
    for i, row in enumerate(shape_table):
        for j, cell in enumerate(row):

            # Set cell color
            light_red = (1.0, 0.6, 0.6)
            light_olive = (0.749, 0.718, 0.42)
            light_orange = (1.0, 0.8, 0.4)
            light_purple = (0.8, 0.6, 0.8)
            light_cyan = (0.8, 1.0, 1.0)

            if i == 0:
                face_color = 'lightgray'
            else:
                face_color = 'white'

            if j == 0 and i != 0:
                if "SWP" in cell:
                    face_color = light_red
                elif "4S" in cell:
                    face_color = light_orange
                elif "2S" in cell:
                    face_color = light_olive
                elif "SNK" in cell:
                    face_color = light_cyan
                elif "CUT" in cell:
                    face_color = light_purple
                elif "CH" in cell:
                    face_color = "lightgreen"
                elif "SPL" in cell:
                    face_color = "lightpink"
                elif "SL" in cell:
                    face_color = "lightyellow"
                elif "CB" in cell:
                    face_color = "lightblue"
                else:
                    face_color = "white"

            # Set text settings
            if i == 0:
                font_weight = 'bold'
            elif j == 0:
                font_weight = 'bold'
            else:
                font_weight = 'normal'
            cell_height = 0.05 if i == 0 else 0.05
            cell_width = 0.6

            cell_text = table.add_cell(i, j, width=cell_width, height=cell_height, text=cell, loc='center', facecolor=face_color)
            cell_text.set_text_props(weight=font_weight)

    # Format
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(3, 3)
    plt.subplots_adjust(left=0.1, bottom=0.3)

    # Save the visual
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    first_last = pitcher_name.split()
    plt.title(pitcher_name + " Release Point (Catcher View)", fontweight="bold", fontsize=font_properties['size'])
    plt.savefig(output_dir / f"{first_last[-1]}_2.pdf")
    plt.close()


def create_cover(pitcher_name, handedness, team):
    """
    Generates and saves a pitcher report cover.

    :param pitcher_name: (str) Pitcher name
    :param handedness: (str) Pitcher hand
    :param team: (str) Team name
    :return: (pdf) Cover image for hitter report
    """
    # Set output path
    first_last = pitcher_name.split()
    output_path = f'/Users/relliott/Desktop/AAAA/{first_last[-1]}_0.pdf'
    logo_path = "/Users/relliott/Desktop/Ballers Analytics/logo.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Set up the canvas
    c = canvas.Canvas(output_path, pagesize=landscape(letter))
    width, height = landscape(letter)

    # Draw the logo
    logo_height = 3 * inch  # Set the height of the logo
    logo_width = logo_height  # Assuming the logo is square
    logo_x = (width - logo_width) / 2
    logo_y = (height - logo_height) / 2 + 50  # Center vertically with an upward adjustment
    c.drawImage(logo_path, logo_x, logo_y, width=logo_width, height=logo_height, mask='auto')

    # Draw the text
    c.setFont("Helvetica-Bold", 22)
    text_y_start = logo_y - 50  # Start drawing text below the logo
    c.drawCentredString(width / 2, text_y_start, "Scouting Report:")
    c.drawCentredString(width / 2, text_y_start - 30, f"{handedness} {pitcher_name},")
    c.drawCentredString(width / 2, text_y_start - 60, f"{team}")

    # Save the PDF
    c.save()


def combine_pdfs(pitcher_names, directory):
    """
    Compiles saved PDFs under each pitcher name into one file each.

    :param pitcher_names: (list) List of pitcher names
    :param directory: (str) Location for saving pitcher reports
    :return: (pdf) Pitcher reports
    """
    # Iterate over each pitcher's name
    for pitcher_name in pitcher_names:
        # Initialize a PdfMerger object
        merger = PdfMerger()

        # Look for files matching the pattern {pitcher_name}_{i}.pdf
        first_last = pitcher_name.split()
        files_to_merge = [f for f in os.listdir(directory) if f.startswith(first_last[-1]) and f.endswith('.pdf')]

        # Sort files by the number in the filename
        files_to_merge.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        # Add each file to the merger
        for file in files_to_merge:
            file_path = os.path.join(directory, file)
            merger.append(file_path)

        # Write out the merged PDF
        if files_to_merge:
            output_path = os.path.join(directory, f'{first_last[-1]}.pdf')
            with open(output_path, 'wb') as output_pdf:
                merger.write(output_pdf)

            # Delete the individual PDFs
            for file in files_to_merge:
                file_path = os.path.join(directory, file)
                os.remove(file_path)

        # Close the merger
        merger.close()


def main():
    """
    Generates pitcher scouting reports.
    """
    # Input parameters
    filename = "csvs/yhwpreseries.csv"
    pitcher_names = [
        "Kris Anglin",
        "Ethan Bates",
        "Ty Buckner",
        "Ben Ferrer",
        "Reed Garland",
        "Jonah Jenkins",
        "Brendan Knoll",
        "Andrew LaCour",
        "Connor Langrell",
        "Brandon McPherson",
        "Brandon Mitchell",
        "Cameron Repetti",
        "Jack Zalasky"
    ]
    team = "Yolo High Wheelers"
    team_code = "YOL_HIG"

    # Non-input parameters
    batter_side = ["Right", "Left"]
    output_dir = "/Users/relliott/Desktop/AAAA"
    pitcher_dict = {}

    # Collect all unique pitcher names, corresponding handedness
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Pitcher")

        if metrics[9] == team_code and metrics[5] not in pitcher_dict.keys():
            if metrics[8] == "Right":
                pitcher_dict[metrics[5]] = "RHP"
            else:
                pitcher_dict[metrics[5]] = "LHP"

    file.close()

    # Generate a report for each pitcher
    for pitcher_name in pitcher_names:

        # Generate cover image
        create_cover(pitcher_name, pitcher_dict[pitcher_name], team)

        # Generate movement plot
        movement_coords, shape_table, pitcher_hand = parse_movement(filename, pitcher_name)
        generate_movement(movement_coords, shape_table, pitcher_name, pitcher_hand, output_dir)

        # # Generate shape table
        # movement_coords, shape_table, pitcher_hand = parse_movement(filename, pitcher_name)
        # generate_shape_table(shape_table, pitcher_name, output_dir)

        for side in batter_side:

            # Generate usage table
            usage_matrix = parse_usage(filename, pitcher_name, side)
            generate_usage(usage_matrix, pitcher_name, side, output_dir)

        for side in batter_side:

            # Generate location plots
            location_dict = parse_location(filename, pitcher_name, side)
            generate_location(location_dict, pitcher_name, side, output_dir)

        for side in batter_side:

            # Generate zone control
            zone_matrix = parse_zone_control(filename, pitcher_name, side)
            generate_zone_control(zone_matrix, pitcher_name, side, output_dir)

        for side in batter_side:

            # Generate batted ball profile
            batted_ball_matrix = parse_batted_balls(filename, pitcher_name, side)
            generate_batted_balls(batted_ball_matrix, pitcher_name, side, output_dir)

        # Generate slash line table
        slash_matrix = parse_slash(filename, pitcher_name)
        generate_slash(slash_matrix, pitcher_name, output_dir)

        # Generate release point plot
        release_coords, shape_table = parse_release(filename, pitcher_name)
        generate_release(release_coords, shape_table, pitcher_name, output_dir)

    # Combine pdfs
    combine_pdfs(pitcher_names, output_dir)


if __name__ == '__main__':
    main()

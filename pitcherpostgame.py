"""
    Project Title: Ballers Baseball Pitcher Postgame Reports

    Author: Riley Elliott

    Date:   9/19/2024
"""

import os
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.table import Table
from pathlib import Path
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from PIL import Image

font_properties = {'size': 12}


def fix_name(metrics_list, p_or_b):
    """
    Re-formats pitcher/batter name columns in an input csv file so that they appear in the form
    "First Last".

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
    Abbreviates tagged pitch names.

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
    Abbreviates pitch results.

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
    Parses a data dictionary to note which pitches a pitcher throws, according to the requirements
    of the visual the data would be used for. Returns the indices (in the list of values in the
    dictionary) of those pitches.

    :param data_dict: (dict) a dictionary where the keys are pitch types that appear in consistent
    order across all intended visual types
    :param graph_purpose: (str) indicates which visual the data will go into
    :return: (list) a list of indices corresponding to pitch types the given pitcher throws
    """
    # Initialize used indices list
    used_indices = []

    # If plotting location:
    if graph_purpose == "Location":
        location = list(data_dict.values())

        # Pitch must be thrown at least once to qualify. Total is not included
        for i in range(len(location)):
            if i < 9 and len(location[i][0]) > 0:
                used_indices.append(i)

    # If plotting accuracy:
    elif graph_purpose == "Accuracy":
        accuracy = list(data_dict.values())

        # Pitch must be thrown at least once to qualify
        for i in range(len(accuracy)):
            if accuracy[i] != "N/A":
                used_indices.append(i)

    return used_indices


def process_number(num):
    """
    Processes a number representing quantity of innings pitched, returns it in conventional format.

    :param num: (float) A number representing quantity of innings pitched
    :return: (float) innings pitched, in the conventional format
    """
    num_str = str(num)

    if num_str.endswith('333'):
        # Change the digits after the decimal to 1
        integer_part = int(num)
        return float(f"{integer_part}.1")
    elif num_str.endswith('667') or num_str.endswith('666'):
        # Change the digits after the decimal to 2
        integer_part = int(num)
        return float(f"{integer_part}.2")
    else:
        return num


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
                       "Sweeper": [[], []], "Slider": [[], []], "Curveball": [[], []]}
    shape_dict = {
        "FourSeamFastBall": [0, 0, 0, 0, 0, 0],
        "TwoSeamFastBall": [0, 0, 0, 0, 0, 0],
        "Sinker": [0, 0, 0, 0, 0, 0],
        "Cutter": [0, 0, 0, 0, 0, 0],
        "ChangeUp": [0, 0, 0, 0, 0, 0],
        "Splitter": [0, 0, 0, 0, 0, 0],
        "Sweeper": [0, 0, 0, 0, 0, 0],
        "Slider": [0, 0, 0, 0, 0, 0],
        "Curveball": [0, 0, 0, 0, 0, 0]
    }
    pitch_count = 0
    out_count = 0

    # Loop through the file
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Pitcher")

        # Record movement data, pitch count, and out count
        if metrics[5] == pitcher_name:
            pitch_count += 1

            if float(metrics[27]) != 0:
                out_count += float(metrics[27])
            if metrics[24] == "Strikeout":
                out_count += 1

            if metrics[21] != "" and metrics[30] != "" and metrics[33] != "" and metrics[40] != "" and metrics[40] != "" and metrics[41] != "":
                movement_coords[metrics[21]][0].append(float(metrics[40]))
                movement_coords[metrics[21]][1].append(float(metrics[41]))
                shape_dict[metrics[21]][0] += float(metrics[30])
                if float(metrics[30]) >= shape_dict[metrics[21]][1]:
                    shape_dict[metrics[21]][1] = float(metrics[30])
                shape_dict[metrics[21]][2] += float(metrics[33])
                shape_dict[metrics[21]][3] += float(metrics[40])
                shape_dict[metrics[21]][4] += float(metrics[41])
                shape_dict[metrics[21]][5] += 1

    file.close()

    # Print workload data to terminal
    print(pitcher_name + ": " + str(process_number(round(out_count/3, 3))) + " IP (" + str(pitch_count) + " P)")

    def format_number(num):
        """Format the number to display as an integer if it ends in .0."""
        if isinstance(num, float) and num.is_integer():
            return int(num)
        return num

    # Turn shape_dict into a table with pitch shape info
    shape_table = [["Pitch", "Velo (mph)", "MaxVelo", "Spin (rpm)", "IVB (in)", "HB (in)"]]
    for key in shape_dict:  # For each pitch type
        if shape_dict[key][5] > 0:
            shapes = [abbreviate_pitch(key)]
            for i in range(5):  # Loop over the first four elements
                if i == 1:  # "MaxVelo"
                    value = round(shape_dict[key][i], 1)
                elif i == 2:  # "Spin"
                    value = round(shape_dict[key][i] / shape_dict[key][5], 0)
                    value = format_number(value)
                else:
                    value = round(shape_dict[key][i] / shape_dict[key][5], 1)
                shapes.append(value)
            shape_table.append(shapes)

    return movement_coords, shape_table, out_count, pitch_count


def generate_shape_table(shape_table, pitcher_name, output_dir):
    """
    Displays pitch metrics in table form only.

    :param shape_table: (matrix) all pitch metrics
    :param pitcher_name: (str) pitcher name
    :param output_dir: (dir) destination for output
    :return: (pdf) pitch metric table
    """
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')
    table = Table(ax, bbox=[0, 0, 1, 1])

    # Add properly formatted cells to the table
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

    # Save the plot with appropriate formatting and title
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=2.0)
    table.auto_set_font_size(False)
    ax.add_table(table)
    plt.title("Movement Profile", fontweight="bold", fontsize=13)

    # Save image as PDF
    first_last = pitcher_name.split()
    plt.savefig(output_dir / f"{first_last[-1]}_1.pdf")
    plt.close()


def parse_location(filename, pitcher_name):
    """
    Parses a Trackman csv and collects pitch location coordinates by pitch type for a given pitcher.

    :param filename: (str) The title of a Trackman csv
    :param pitcher_name: (str) Pitcher name
    :return: (dict) Pitch location coordinates by pitch type
    """
    # Initialize the location dictionary and list of pitches
    location_dict = {"FourSeamFastBall": [[], []], "TwoSeamFastBall": [[], []],
                     "Sinker": [[], []], "Cutter": [[], []], "ChangeUp": [[], []], "Splitter": [[], []],
                     "Sweeper": [[], []], "Slider": [[], []], "Curveball": [[], []], "Batter": "No"}
    whiff_coordinates = []
    hit_coordinates = []

    # Loop through the file
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Pitcher")

        # Record pitch location data. Note whiffs and hits
        if ((metrics[5] == pitcher_name and metrics[21] in location_dict.keys()
             and metrics[42] != "" and metrics[43] != "")):
            pitcher_name = metrics[5]
            if metrics[23] == "StrikeSwinging":
                location_dict[metrics[21]][0].append(float(metrics[43]))
                location_dict[metrics[21]][1].append(float(metrics[42]))
                whiff_coordinates.append([float(metrics[43]), float(metrics[42])])
            elif metrics[23] == "InPlay" and metrics[26] in ["Single", "Double", "Triple", "HomeRun"]:
                location_dict[metrics[21]][0].append(float(metrics[43]))
                location_dict[metrics[21]][1].append(float(metrics[42]))
                hit_coordinates.append([float(metrics[43]), float(metrics[42])])
            else:
                location_dict[metrics[21]][0].append(float(metrics[43]))
                location_dict[metrics[21]][1].append(float(metrics[42]))

    file.close()

    return location_dict, whiff_coordinates, hit_coordinates


def generate_location(location_dict, whiff_coordinates, hit_coordinates, pitcher_name, output_dir):
    """
    Generates and saves location plots for each pitch a given pitcher throws.

    :param location_dict: (dict) Pitch location coordinates by pitch type
    :param whiff_coordinates: (list) Coordinates of pitches resulting in whiffs
    :param hit_coordinates: (list) Coordinates of pitches resulting in hits
    :param pitcher_name: (str) Pitcher name
    :param output_dir: (dir) Location for saving the location plots
    :return: (pdf) One location plot for each pitch a given pitcher throw
    """
    # Note which pitches the pitcher throws
    used_indices = find_repertoire(location_dict, "Location")

    # Initialize generic and updated labels
    labels = ["4-seam", "2-seam", "Sinker", "Cutter",
              "ChangeUp", "Splitter", "Sweeper", "Slider", "Curveball"]
    locations = list(location_dict.values())
    colors = ["orange", "olive", "cyan", "purple",
              "green", "pink", (1.0, 0.6, 0.6), (0.95, 0.8, 0), (0, 0.3, 1)]
    markers = ["s", "^", "v", "*", "d", "p", "o", "x", "+"]
    new_labels = []
    new_coordinates = []
    new_colors = []
    new_markers = []

    # Only append pitch info to labels if pitch is sufficiently used
    for item in used_indices:
        new_labels.append(abbreviate_pitch(labels[item]))
        new_coordinates.append(locations[item])
        new_colors.append(colors[item])
        new_markers.append(markers[item])

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Generate strike zone and buffer
    strike_zone = patches.Rectangle((-(1.37/2), 1.7), 1.37, 1.9,
                                    linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(strike_zone)
    buffer_zone = patches.Rectangle((-(1.37/2) - 0.166667, 1.7 - 0.166667), 1.37 + 0.333333, 1.9 + 0.333333,
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

    # Plot pitch coordinates by pitch type with proper labels
    for i in range(len(new_coordinates)):
        x_coordinates = [value for value in new_coordinates[i][0]]
        y_coordinates = [value for value in new_coordinates[i][1]]
        ax.scatter(x_coordinates, y_coordinates, color=new_colors[i], marker=new_markers[i], label=new_labels[i], s=120)

    # Circle pitches that resulted in whiffs (red circle)
    for whiff_coord in whiff_coordinates:
        whiff_x, whiff_y = whiff_coord
        circle_radius = 0.15
        circle = Circle((whiff_x, whiff_y), circle_radius, color='red', fill=False)
        ax.add_patch(circle)

    # Circle pitches that resulted in hits (black circle)
    for hit_coordinate in hit_coordinates:
        contact_x, contact_y = hit_coordinate
        circle_radius = 0.15
        circle = Circle((contact_x, contact_y), circle_radius, color='black', fill=False)
        ax.add_patch(circle)

    # Set axis limits and labels
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')

    # Add legend with collected handles and labels
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    ax.legend(fontsize='9')
    ax.set_xlabel("Width (feet)")
    ax.set_ylabel("Height (feet)")
    plt.suptitle("Location (Pitcher View)", fontweight="bold", fontsize=11)
    plt.title("Whiffs = Red, Hits = Black", fontsize=9)

    # Save table as PDF
    first_last = pitcher_name.split()
    plt.savefig(output_dir / f"{first_last[-1]}_2.pdf")
    plt.close()


def parse_batted_balls(filename, pitcher_name):
    """
    Parses a Trackman csv file and collects batted ball and corresponding pitch metrics for a given pitcher.

    :param filename: (str) Name of a Trackman csv
    :param pitcher_name: (str) Pitcher name
    :return: (matrix) Table with batted ball data
    """
    # Initialize matrix
    batted_ball_matrix = [["Pitch", "Velo (mph)", "Spin (rpm)", "IVB (in)", "HB (in)", "EV (mph)", "LA (°)",
                           "Spray (°)", "Result"]]

    # Loop through file
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Pitcher")

        # Record batted ball data
        if metrics[5] == pitcher_name and metrics[23] == "InPlay" and metrics[25] != "Bunt":
            metrics_list = [metrics[30], metrics[33], metrics[40], metrics[41], metrics[48], metrics[49],
                            metrics[50]]
            metrics_list = [0 if metric == "" else metric for metric in metrics_list]
            batted_ball_matrix.append([abbreviate_pitch(metrics[21]), round(float(metrics_list[0]), 1),
                                       round(float(metrics_list[1]), 0), round(float(metrics_list[2]), 1),
                                       round(float(metrics_list[3]), 1), round(float(metrics_list[4]), 1),
                                       round(float(metrics_list[5]), 0), round(float(metrics_list[6]), 0),
                                       str(abbreviate_result(metrics[25])) + " " + str(abbreviate_result(metrics[26]))])

    file.close()

    def format_number(num):
        """Format the number to display as an integer if it ends in .0."""
        if isinstance(num, float) and num.is_integer():
            return int(num)
        return num

    def process_matrix(matrix):
        # Column indices for "EV", "LA", and "Spray"
        ev_index = matrix[0].index("EV (mph)")
        la_index = matrix[0].index("LA (°)")
        spray_index = matrix[0].index("Spray (°)")

        # Replace "EV", "LA", and "Spray" with "-" only if all three are 0
        for i in range(1, len(matrix)):  # Start from 1 to skip the header
            # Format all numbers first
            for j in range(len(matrix[i])):
                matrix[i][j] = format_number(matrix[i][j])

            if (matrix[i][ev_index] == 0 and
                    matrix[i][la_index] == 0 and
                    matrix[i][spray_index] == 0):
                matrix[i][ev_index] = "-"
                matrix[i][la_index] = "-"
                matrix[i][spray_index] = "-"

            # Replace other 0 values unconditionally
            for j in range(len(matrix[i])):
                if matrix[i][j] == 0:
                    if j not in [ev_index, la_index, spray_index]:
                        matrix[i][j] = "-"

    # Apply the function to process the matrix
    process_matrix(batted_ball_matrix)

    return batted_ball_matrix


def generate_batted_balls(batted_ball_matrix, pitcher_name):
    """
    Generates a table containing batted ball and corresponding pitch metrics for a given pitcher.

    :param batted_ball_matrix: (matrix) Table with batted ball info
    :param pitcher_name: (str) Pitcher name
    :return: Table containing batted ball metrics for a given pitcher
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 5))
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
            font_weight = 'bold' if i == 0 else 'normal'
            cell_height = 0.06 if i == 0 else 0.04
            cell_text = table.add_cell(i, j, 0.1, cell_height, text=cell, loc='center', facecolor=face_color)
            cell_text.set_text_props(weight=font_weight)

    # Display the plot with appropriate formatting and title
    plt.tight_layout(pad=2.0)
    table.auto_set_font_size(False)
    ax.add_table(table)
    plt.title(pitcher_name + " Batted Ball Events", fontweight="bold", fontsize=font_properties['size'] + 1)
    plt.show()


def parse_zone_control(filename, pitcher_name):
    """
    Parses a Trackman csv file and collects pitch effectiveness metrics for a given pitcher.

    :param filename: (str) Name of a Trackman csv
    :param pitcher_name: (str) Pitcher name
    :return: (matrix) Table with pitch effectiveness metrics
    """
    # Initialize data dictionaries
    zone_dict = {"FourSeamFastBall": [0, 0], "TwoSeamFastBall": [0, 0], "Sinker": [0, 0],
                 "Cutter": [0, 0], "ChangeUp": [0, 0], "Splitter": [0, 0], "Sweeper": [0, 0], "Slider": [0, 0], "Curveball": [0, 0],
                 "Total": [0, 0]}
    conversion_dict = {"FourSeamFastBall": [0, 0], "TwoSeamFastBall": [0, 0], "Sinker": [0, 0],
                       "Cutter": [0, 0], "ChangeUp": [0, 0], "Splitter": [0, 0], "Sweeper": [0, 0], "Slider": [0, 0], "Curveball": [0, 0],
                       "Total": [0, 0]}
    whiff_dict = {"FourSeamFastBall": [0, 0], "TwoSeamFastBall": [0, 0], "Sinker": [0, 0],
                  "Cutter": [0, 0], "ChangeUp": [0, 0], "Splitter": [0, 0], "Sweeper": [0, 0], "Slider": [0, 0], "Curveball": [0, 0],
                  "Total": [0, 0]}
    chase_dict = {"FourSeamFastBall": [0, 0], "TwoSeamFastBall": [0, 0], "Sinker": [0, 0],
                  "Cutter": [0, 0], "ChangeUp": [0, 0], "Splitter": [0, 0], "Sweeper": [0, 0], "Slider": [0, 0], "Curveball": [0, 0],
                  "Total": [0, 0]}

    # Loop through the file
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Pitcher")

        # Record pitch effectiveness data
        if metrics[5] == pitcher_name and metrics[21] in zone_dict.keys():

            # Conversion rates
            if int(metrics[19]) == int(metrics[20]):  # When count is even
                if (metrics[23] == "StrikeCalled" or metrics[23] == "StrikeSwinging" or metrics[23] == "FoulBall" or
                        metrics[23] == "FoulBallNotFieldable" or metrics[23] == "FoulBallFieldable" or metrics[26] == "Out"):
                    conversion_dict[metrics[21]][0] += 1
                    conversion_dict[metrics[21]][1] += 1
                    conversion_dict["Total"][0] += 1
                    conversion_dict["Total"][1] += 1
                elif (metrics[23] == "BallCalled" or metrics[23] == "BallinDirt" or metrics[23] == "HitByPitch"
                      or metrics[26] in ["Single, Double", "Triple", "HomeRun"]):
                    conversion_dict[metrics[21]][1] += 1
                    conversion_dict["Total"][1] += 1

            # Zone rates
            if metrics[42] != "" and metrics[43] != "":
                if -(1.70333 / 2) <= float(metrics[43]) <= (1.70333 / 2) and 1.5333 <= float(metrics[42]) <= 3.7666:
                    zone_dict[metrics[21]][0] += 1
                    zone_dict[metrics[21]][1] += 1
                    zone_dict["Total"][0] += 1
                    zone_dict["Total"][1] += 1
                else:
                    zone_dict[metrics[21]][1] += 1
                    zone_dict["Total"][1] += 1

                    # Chase rates
                    if (metrics[23] == "StrikeSwinging" or metrics[23] == "FoulBall" or metrics[23] ==
                            "FoulBallNotFieldable" or metrics[23] == "FoulBallFieldable" or metrics[23] == "InPlay"):
                        chase_dict[metrics[21]][0] += 1
                        chase_dict[metrics[21]][1] += 1
                        chase_dict["Total"][0] += 1
                        chase_dict["Total"][1] += 1
                    else:
                        chase_dict[metrics[21]][1] += 1
                        chase_dict["Total"][1] += 1

            # Whiff rates
            if metrics[23] == "StrikeSwinging":
                whiff_dict[metrics[21]][0] += 1
                whiff_dict[metrics[21]][1] += 1
                whiff_dict["Total"][0] += 1
                whiff_dict["Total"][1] += 1
            elif (metrics[23] == "FoulBall" or metrics[23] == "FoulBallNotFieldable"
                  or metrics[23] == "FoulBallFieldable" or metrics[23] == "InPlay"):
                whiff_dict[metrics[21]][1] += 1
                whiff_dict["Total"][1] += 1

    file.close()

    # Check for sample size
    dicts = [zone_dict, conversion_dict, whiff_dict, chase_dict]
    for pitch_dict in dicts:
        for key in pitch_dict.keys():
            if pitch_dict[key][1] > 0:
                pitch_dict[key] = round(100 * (pitch_dict[key][0] / pitch_dict[key][1]), 1)
            else:
                pitch_dict[key] = "N/A"

    # Note which pitches are used
    used_indices = find_repertoire(zone_dict, "Accuracy")
    pitches = list(zone_dict.keys())
    strikes = list(zone_dict.values())
    conversions = list(conversion_dict.values())
    whiffs = list(whiff_dict.values())
    chases = list(chase_dict.values())

    # Create data matrix
    zone_matrix = [["", "Zone%", "Even Count\nConversion%", "Whiff%", "Chase%"]]
    for item in used_indices:
        row = [abbreviate_pitch(pitches[item]), strikes[item], conversions[item], whiffs[item], chases[item]]
        zone_matrix.append(row)

    return zone_matrix


def generate_zone_control(zone_matrix, pitcher_name, output_dir):
    """
    Generates and saves a table with pitch effectiveness metrics for a given pitcher.

    :param zone_matrix: (matrix) Matrix with pitch effectiveness metrics
    :param pitcher_name: (str) Pitcher name
    :param output_dir: (str) Location for saving output
    :return: (pdf) Table with pitch effectiveness metrics
    """
    # Create figure, axis
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
                    face_color = 'lightgray'
                else:
                    face_color = "white"

            # Set font weight and cell height for row 1
            if i == 0:
                font_weight = 'bold'
            elif j == 0:
                font_weight = 'bold'
            else:
                font_weight = 'normal'
            cell_height = 0.02 if i == 0 else 0.04

            # Set text settings
            cell_text = table.add_cell(i, j, 0.1, cell_height, text=cell, loc='center', facecolor=face_color)
            cell_text.set_text_props(weight=font_weight)

    # Display the plot with appropriate formatting and title
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=2.0)
    table.auto_set_font_size(False)
    ax.add_table(table)
    plt.title("Zone Control", fontweight="bold", fontsize=font_properties['size'] + 1)

    # Save image as PDF
    first_last = pitcher_name.split()
    plt.savefig(output_dir / f"{first_last[-1]}_3.pdf")
    plt.close()


def create_cover(date, pitcher_name, out_count, pitch_count, handedness, team):
    """
    Generates and saves a postgame report cover. Contains information on player, game date, and opponent.

    :param date: (str) Game date
    :param pitcher_name: (str) Pitcher name
    :param out_count: (int) Number of outs the pitcher recorded
    :param pitch_count: (int) Pitcher pitch count
    :param handedness: (str) Pitcher hand
    :param team: (str) Opponent team name
    :return: (pdf) Cover image for postgame report
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
    logo_y = (height - logo_height) / 2 + 75  # Center vertically with an upward adjustment
    c.drawImage(logo_path, logo_x, logo_y, width=logo_width, height=logo_height, mask='auto')

    # Draw the text
    c.setFont("Helvetica-Bold", 22)
    text_y_start = logo_y - 50  # Start drawing text below the logo
    c.drawCentredString(width / 2, text_y_start, "Postgame Report:")
    c.drawCentredString(width / 2, text_y_start - 30, f"{handedness} {pitcher_name},")
    c.drawCentredString(width / 2, text_y_start - 60, f"{date} vs. {team}")
    c.drawCentredString(width / 2, text_y_start - 90, f"{process_number(round(out_count/3, 3))} IP ({pitch_count} P)")

    # Save the PDF
    c.save()


def compile_pdf(pitcher_name, output_dir):
    """
    Compiles saved PDFs under one name into one file, collates pages onto one sheet.

    :param pitcher_name: (str) Pitcher name
    :param output_dir: (str) Location for saving final postgame report
    :return: (pdf) Final postgame report
    """
    # Set file names
    first_last = pitcher_name.split()
    if first_last[-1] != "Pierre":
        output_file = f'/Users/relliott/Desktop/AAAA/{first_last[-1]}.pdf'
    else:
        output_file = f'/Users/relliott/Desktop/AAAA/St. Pierre.pdf'

    # Initialize a PdfMerger object
    merger = PdfMerger()

    # Look for files matching the pattern Roster_{i}.pdf
    files_to_merge = [f for f in os.listdir(output_dir) if f.startswith(first_last[-1]) and f.endswith('.pdf')]

    # Sort files by the number in the filename
    files_to_merge.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Add each file to the merger
    for file in files_to_merge:
        file_path = os.path.join(output_dir, file)
        merger.append(file_path)

    # Write out the merged PDF
    if files_to_merge:
        merged_pdf_path = os.path.join(output_dir, f'{first_last[-1]}.pdf')
        with open(merged_pdf_path, 'wb') as merged_pdf:
            merger.write(merged_pdf)

    # Close the merger
    merger.close()

    # Convert PDF pages to images
    images = convert_from_path(merged_pdf_path)

    # Create a canvas for the new PDF
    can = canvas.Canvas(output_file, pagesize=landscape(letter))
    page_width, page_height = landscape(letter)

    # Calculate the dimensions for each quadrant
    quadrant_width = page_width / 2
    quadrant_height = page_height / 2

    # Add each image (representing a PDF page) to the new PDF, placing them in quadrants
    for i, img in enumerate(images):
        if i % 4 == 0 and i > 0:
            can.showPage()

        x_offset = (i % 2) * quadrant_width
        y_offset = page_height - ((i // 2) % 2 + 1) * quadrant_height

        img_path = f"/tmp/page_{i}.png"
        img.save(img_path, 'PNG')

        # Load the image
        image_width, image_height = img.size

        # Calculate the new dimensions to maintain aspect ratio
        aspect_ratio = image_width / image_height
        if aspect_ratio > 1:  # Landscape
            new_width = min(quadrant_width, image_width)
            new_height = new_width / aspect_ratio
        else:  # Portrait
            new_height = min(quadrant_height, image_height)
            new_width = new_height * aspect_ratio

        # Center the image within the quadrant
        centered_x_offset = x_offset + (quadrant_width - new_width) / 2
        centered_y_offset = y_offset + (quadrant_height - new_height) / 2

        # Draw the image
        can.drawImage(img_path, centered_x_offset, centered_y_offset, new_width, new_height)

        # Remove the temporary image file
        os.remove(img_path)

    can.save()
    print(f"Saved final PDF: {output_file}")

    # Remove the original individual PDFs
    for file in files_to_merge:
        os.remove(os.path.join(output_dir, file))

    # Additionally, remove Pierre.pdf if it exists
    for spare_pdf in ['Pierre.pdf']:
        spare_pdf_path = os.path.join(output_dir, spare_pdf)
        if os.path.exists(spare_pdf_path):
            os.remove(spare_pdf_path)


def main():
    """
    Parses game data, generates postgame report for each Oakland pitcher.
    """
    # Initialize parameters
    filename = "/Users/relliott/Desktop/Ballers Analytics/BallersCode/venv/csvs/0913yhw.csv"
    team = "Yolo"
    output_dir = "/Users/relliott/Desktop/AAAA"
    pitcher_dict = {}

    # Extract date of game for labeling
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        if metrics[1] != "":
            date = metrics[1]
            break

    # Collect all unique pitcher names, corresponding pitcher hand
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Pitcher")

        if metrics[9] == "OAK_BAL" and metrics[5] not in pitcher_dict.keys():
            if metrics[8] == "Right":
                pitcher_dict[metrics[5]] = "RHP"
            else:
                pitcher_dict[metrics[5]] = "LHP"

    file.close()

    # For each unique pitcher in the game csv:
    for pitcher_name in pitcher_dict.keys():

        # Generate pitch shape table
        movement_coords, shape_table, out_count, pitch_count = parse_movement(filename, pitcher_name)
        generate_shape_table(shape_table, pitcher_name, output_dir)

        # Generate cover image
        create_cover(date, pitcher_name, out_count, pitch_count, pitcher_dict[pitcher_name], team)

        # Generate pitch locations
        location_dict, whiff_coordinates, hit_coordinates = parse_location(filename, pitcher_name)
        generate_location(location_dict, whiff_coordinates, hit_coordinates, pitcher_name, output_dir)

        # # Generate batted ball metrics
        # batted_ball_matrix = parse_batted_balls(filename, pitcher_name)
        # generate_batted_balls(batted_ball_matrix, pitcher_name)

        # Generate zone control
        zone_matrix = parse_zone_control(filename, pitcher_name)
        generate_zone_control(zone_matrix, pitcher_name, output_dir)

        compile_pdf(pitcher_name, output_dir)


if __name__ == "__main__":
    main()

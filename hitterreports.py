"""
    Project Title: Ballers Baseball Hitter Scouting Reports

    Author: Riley Elliott

    Date:   5/14/2024
"""

import os
import math
import matplotlib.pyplot as plt
from matplotlib.table import Table
import matplotlib.colors as mcolors
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

font_properties = {'size': 11}


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
    if tagged_pitch == "Fastball":
        return "FB"
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


def parse_chase(filename, batter_name, pitcher_hand):
    """
    Parses a Trackman csv to collect chase locations and chase rates by pitch type for a given batter.

    :param filename: (str) Name of Trackman csv
    :param batter_name: (str) Batter name
    :param pitcher_hand: (str) Pitcher hand
    :return: (dict) Dictionaries containing chase locations and chase rates by pitch type
    """
    # Allow for filtering by pitcher hand
    if pitcher_hand == "Right":
        pitcher_hand = ["Right"]
    elif pitcher_hand == "Left":
        pitcher_hand = ["Left"]
    elif pitcher_hand == "Both":
        pitcher_hand = ["Right", "Left"]

    # Organize pitch types to maximize sample size, simplify output
    def categorize_pitch(pitch_type):
        if pitch_type in ["FourSeamFastBall", "Fastball", "Cutter"]:
            return "4S"
        elif pitch_type in ["TwoSeamFastBall", "Sinker"]:
            return "SNK"
        elif pitch_type in ["ChangeUp", "Splitter", "Knuckleball"]:
            return "CH"
        elif pitch_type in ["Slider", "Sweeper"]:
            return "SL"
        elif pitch_type in ["Curveball"]:
            return "CB"

    # Initialize dictionaries for chase locations and chase rates by pitch type
    chase_locations = {
        "4S": [[], []],
        "SNK": [[], []],
        "CH": [[], []],
        "SL": [[], []],
        "CB": [[], []],
    }

    chase_rates = {
        "4S": {"Down": [0, 0], "Up": [0, 0], "R-Gloveside": [0, 0], "R-Armside": [0, 0]},
        "SNK": {"Down": [0, 0], "Up": [0, 0], "R-Gloveside": [0, 0], "R-Armside": [0, 0]},
        "CH": {"Down": [0, 0], "Up": [0, 0], "R-Gloveside": [0, 0], "R-Armside": [0, 0]},
        "SL": {"Down": [0, 0], "Up": [0, 0], "R-Gloveside": [0, 0], "R-Armside": [0, 0]},
        "CB": {"Down": [0, 0], "Up": [0, 0], "R-Gloveside": [0, 0], "R-Armside": [0, 0]},
    }

    # Loop through the file
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Batter")

        if metrics[21] not in ["Fastball", "FourSeamFastBall", "TwoSeamFastBall", "Sinker", "Cutter", "ChangeUp", "Splitter", "Slider", "Sweeper", "Curveball", "Knuckleball"]:
            continue

        # Record pitch location data. Note which pitches are swung at
        if metrics[10] == batter_name and metrics[8] in pitcher_hand and metrics[42] != "" and metrics[43] != "":
            if (metrics[23] in ["StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]
                    and (-(1.5 / 2) > float(metrics[43]) or float(metrics[43]) > (1.5 / 2)
                         or 1.7 > float(metrics[42]) or float(metrics[42]) > 3.6)):
                chase_locations[categorize_pitch(metrics[21])][0].append(float(metrics[43]))
                chase_locations[categorize_pitch(metrics[21])][1].append(float(metrics[42]))

            # Update chase rates by pitch type and location
            if float(metrics[42]) < 1.7:  # Below zone
                chase_rates[categorize_pitch(metrics[21])]["Down"][1] += 1
                if metrics[23] in ["StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]:
                    chase_rates[categorize_pitch(metrics[21])]["Down"][0] += 1
            elif float(metrics[42]) > 3.6:  # Above zone
                chase_rates[categorize_pitch(metrics[21])]["Up"][1] += 1
                if metrics[23] in ["StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]:
                    chase_rates[categorize_pitch(metrics[21])]["Up"][0] += 1
            if float(metrics[43]) < (-1.5 / 2):  # LHB side
                chase_rates[categorize_pitch(metrics[21])]["R-Gloveside"][1] += 1
                if metrics[23] in ["StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]:
                    chase_rates[categorize_pitch(metrics[21])]["R-Gloveside"][0] += 1
            elif float(metrics[43]) > (1.5 / 2):  # RHB side
                chase_rates[categorize_pitch(metrics[21])]["R-Armside"][1] += 1
                if metrics[23] in ["StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]:
                    chase_rates[categorize_pitch(metrics[21])]["R-Armside"][0] += 1

    file.close()

    # Convert chase rates to percentages
    for pitch in chase_rates:
        for key in chase_rates[pitch]:
            if chase_rates[pitch][key][1] != 0:
                chase_rates[pitch][key] = str(
                    round(100 * safe_division(chase_rates[pitch][key][0], chase_rates[pitch][key][1]), 1)) + "%"
            else:
                chase_rates[pitch][key] = ""

    return chase_locations, chase_rates


def generate_chase(chase_locations, chase_rates, batter_name, pitcher_hand, plot_num, output_dir):
    """
    Generates heatmaps for chase locations and rates by pitch type for a given batter.

    :param chase_locations: (dict) Dictionary containing chase locations by pitch type
    :param chase_rates: (dict) Dictionary containing chase rates by pitch type
    :param batter_name: (str) Batter name
    :param pitcher_hand: (str) Pitcher name
    :param plot_num: (int) Number of plot for printing
    :param output_dir: (dir) File location for saving the output
    :return: (pdf) Saves a chase heatmap visual for each pitch type
    """
    # List pitch types
    pitches = list(chase_locations.keys())

    # Loop through each pitch type
    for pitch_type in pitches:
        if not chase_locations[pitch_type][0]:  # Skip if no data for this pitch type
            continue

        # Create a figure and axis
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

        # Add swing locations as coordinates
        x_coordinates = chase_locations[pitch_type][0]
        y_coordinates = chase_locations[pitch_type][1]

        # Create a 2D histogram, smooth using Gaussian filter
        heatmap, xedges, yedges = np.histogram2d(x_coordinates, y_coordinates, bins=(50, 50))
        heatmap = gaussian_filter(heatmap, sigma=2)
        img = ax.imshow(heatmap.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='coolwarm',
                        alpha=0.5)
        # cbar = fig.colorbar(img, ax=ax)
        # cbar.set_label('Frequency')

        # Set axis limits and labels
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(1.0, 4.3)
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')

        # Add chase rate numbers to the plot
        ax.text(0, 1.32, chase_rates[pitch_type]["Down"], fontsize=10, ha='center', fontweight='bold')  # Below strike zone
        ax.text(0, 3.89, chase_rates[pitch_type]["Up"], fontsize=10, ha='center', fontweight='bold')  # Above strike zone
        ax.text(-1.08, 2.68, chase_rates[pitch_type]["R-Gloveside"], fontsize=10, ha='center', fontweight='bold')  # Left of strike zone
        ax.text(1.08, 2.68, chase_rates[pitch_type]["R-Armside"], fontsize=10, ha='center', fontweight='bold')  # Right of strike zone

        # Save the plot
        plt.title(f"{batter_name} {pitch_type} Chase Locations\nvs. {pitcher_hand} (Pitcher View)", fontweight="bold", fontsize=13)
        first_last = batter_name.split()
        output_dir = Path(output_dir)  # Convert to Path object
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{first_last[-1]}_{plot_num}.pdf", bbox_inches='tight')
        plt.close()
        plot_num += 1  # Increment the plot number after each pitch type


def parse_whiff_by_location(filename, batter_name, pitcher_hand):
    """
    Parses a Trackman csv to calculate whiff rates by zone quadrant against each pitch type for a given batter.

    :param filename: (str) Name of Trackman csv
    :param batter_name: (str) Batter name
    :param pitcher_hand: (str) Pitcher hand
    :return: (dict) A dictionary containing whiff rates by pitch and by zone quadrant
    """
    # Allow filtering by pitcher hand
    if pitcher_hand == "Right":
        pitcher_hand = ["Right"]
    elif pitcher_hand == "Left":
        pitcher_hand = ["Left"]
    elif pitcher_hand == "Both":
        pitcher_hand = ["Right", "Left"]

    # Organize pitch types to maximize sample size, simplify output
    def categorize_pitch(pitch_type):
        if pitch_type in ["FourSeamFastBall", "Fastball", "Cutter"]:
            return "4S"
        elif pitch_type in ["TwoSeamFastBall", "Sinker"]:
            return "SNK"
        elif pitch_type in ["ChangeUp", "Splitter", "Knuckleball"]:
            return "CH"
        elif pitch_type in ["Slider", "Sweeper"]:
            return "SL"
        elif pitch_type in ["Curveball"]:
            return "CB"

    # Initialize data dictionary
    whiff_locations = {
        "4S": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "SNK": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "CH": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "SL": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "CB": [[0, 0], [0, 0], [0, 0], [0, 0]],
    }

    # Loop through file
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Batter")

        # Calculate whiff rates
        if metrics[10] == batter_name and metrics[8] in pitcher_hand and metrics[21] != "" and metrics[42] != "" and metrics[43] != "":
            if 0 >= float(metrics[43]) >= -0.851665 and 2.649965 < float(metrics[42]) <= 3.76662999999:  # In to LHB, up
                if metrics[23] == "StrikeSwinging":
                    whiff_locations[categorize_pitch(metrics[21])][0][0] += 1
                    whiff_locations[categorize_pitch(metrics[21])][0][1] += 1
                elif metrics[23] in ["InPlay", "FoulBall", "FoulBallFieldable", "FoulBallNotFieldable"]:
                    whiff_locations[categorize_pitch(metrics[21])][0][1] += 1
            elif 0.851665 >= float(metrics[43]) > 0 and 2.649965 < float(metrics[42]) <= 3.76662999999:  # Out to RHB, up
                if metrics[23] == "StrikeSwinging":
                    whiff_locations[categorize_pitch(metrics[21])][1][0] += 1
                    whiff_locations[categorize_pitch(metrics[21])][1][1] += 1
                elif metrics[23] in ["InPlay", "FoulBall", "FoulBallFieldable", "FoulBallNotFieldable"]:
                    whiff_locations[categorize_pitch(metrics[21])][1][1] += 1
            elif 0 >= float(metrics[43]) >= -0.851665 and 1.5333 < float(metrics[42]) <= 2.649965:  # In to LHB, down
                if metrics[23] == "StrikeSwinging":
                    whiff_locations[categorize_pitch(metrics[21])][2][0] += 1
                    whiff_locations[categorize_pitch(metrics[21])][2][1] += 1
                elif metrics[23] in ["InPlay", "FoulBall", "FoulBallFieldable", "FoulBallNotFieldable"]:
                    whiff_locations[categorize_pitch(metrics[21])][2][1] += 1
            elif 0.851665 >= float(metrics[43]) > 0 and 1.5333 < float(metrics[42]) <= 2.649965:  # In to RHB, down
                if metrics[23] == "StrikeSwinging":
                    whiff_locations[categorize_pitch(metrics[21])][3][0] += 1
                    whiff_locations[categorize_pitch(metrics[21])][3][1] += 1
                elif metrics[23] in ["InPlay", "FoulBall", "FoulBallFieldable", "FoulBallNotFieldable"]:
                    whiff_locations[categorize_pitch(metrics[21])][3][1] += 1

    file.close()

    # Convert chase rates to percentages
    for pitch in whiff_locations:
        for i in range(len(whiff_locations[pitch])):
            if whiff_locations[pitch][i][1] >= 5:
                whiff_locations[pitch][i] = float(round(100 * (whiff_locations[pitch][i][0] / whiff_locations[pitch][i][1]), 1))
            else:
                whiff_locations[pitch][i] = "-"

    return whiff_locations


def generate_whiff_by_location(whiff_locations, batter_name, pitcher_hand, plot_num, output_dir):
    """
    Generates color coded chart of whiff locations by zone quadrant against each pitch type for a given batter.

    :param whiff_locations: (dict) A dictionary containing whiff rates by pitch and by zone quadrant
    :param batter_name: (str) Batter name
    :param pitcher_hand: (str) Pitcher hand
    :param plot_num: (int) Number of plot for printing
    :param output_dir: (str) File location for saving output
    :return: (pdf) Saves a whiff rate visual for each pitch type
    """
    # List pitch types
    pitches = list(whiff_locations.keys())

    # Loop through each pitch type
    for pitch in pitches:
        if not any(item != "-" for item in whiff_locations[pitch]):
            continue

        # Convert data to matrix
        whiff_matrix = [[whiff_locations[pitch][0], whiff_locations[pitch][1]], [whiff_locations[pitch][2], whiff_locations[pitch][3]]]

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.axis('off')
        table = Table(ax, bbox=[0, 0, 1, 1])

        # Convert location_list to a numpy array for numerical operations
        location_array = np.array([cell for row in whiff_matrix for cell in row if isinstance(cell, float)])

        # Define the min, mid, and max values for your scale
        min_value = 0
        mid_value = 25
        max_value = 50

        # Create a colormap and a TwoSlopeNorm normalization
        cmap = plt.get_cmap('coolwarm')
        norm = mcolors.TwoSlopeNorm(vmin=min_value, vcenter=mid_value, vmax=max_value)

        # Add cells to the table, format
        for i, row in enumerate(whiff_matrix):
            for j, cell in enumerate(row):
                cell_height = 0.05

                # Ensure cell contains numerical values
                if isinstance(cell, (float, int)):
                    cell_value = float(cell)

                    # Normalize the cell value based on the custom normalization
                    norm_cell_value = 1 - norm(cell_value)

                    # Get the color from the colormap
                    cell_color = cmap(norm_cell_value)

                    # Add the cell with the specified color
                    table.add_cell(i, j, 0.1, cell_height, text=cell, loc='center', facecolor=cell_color)
                else:
                    table.add_cell(i, j, 0.1, cell_height, text=cell, loc='center')

        # Set formatting and title
        plt.tight_layout(pad=2.0)
        table.auto_set_font_size(False)
        ax.add_table(table)
        plt.title(f"{batter_name} {pitch} Whiff Rates vs. {pitcher_hand} (P. View)", fontweight="bold",
                  fontsize=12)

        # Save the output
        output_dir = Path(output_dir)  # Convert to Path object
        output_dir.mkdir(parents=True, exist_ok=True)
        first_last = batter_name.split()
        plt.savefig(output_dir / f"{first_last[-1]}_{plot_num}.pdf")
        plt.close()
        plot_num += 1  # Increment the plot number after each pitch type


def parse_slg_by_location(filename, batter_name, pitcher_hand):
    """
    Parses a Trackman csv to calculate slugging percentages by zone quadrant against each pitch type for a given batter.

    :param filename: (str) Name of Trackman file
    :param batter_name: (str) Batter name
    :param pitcher_hand: (str) Pitcher hand
    :return: (dict) A dictionary with slugging percentage by pitch type and zone quadrant
    """
    # Allow for filtering by pitcher hand
    if pitcher_hand == "Right":
        pitcher_hand = ["Right"]
    elif pitcher_hand == "Left":
        pitcher_hand = ["Left"]
    elif pitcher_hand == "Both":
        pitcher_hand = ["Right", "Left"]

    # Organize pitch types to maximize sample size, simplify output
    def categorize_pitch(pitch_type):
        if pitch_type in ["FourSeamFastBall", "Fastball", "Cutter"]:
            return "4S"
        elif pitch_type in ["TwoSeamFastBall", "Sinker"]:
            return "SNK"
        elif pitch_type in ["ChangeUp", "Splitter", "Knuckleball"]:
            return "CH"
        elif pitch_type in ["Slider", "Sweeper"]:
            return "SL"
        elif pitch_type in ["Curveball"]:
            return "CB"

    # Total base counter
    def count_total_bases(outcome):
        if outcome == "Single":
            return 1
        elif outcome == "Double":
            return 2
        elif outcome == "Triple":
            return 3
        elif outcome == "HomeRun":
            return 4

    # Initialize data dictionary
    slg_locations = {
        "4S": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "SNK": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "CH": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "SL": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "CB": [[0, 0], [0, 0], [0, 0], [0, 0]],
    }

    # Loop through file
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Batter")

        # Calculate slugging percentages
        if metrics[10] == batter_name and metrics[8] in pitcher_hand and metrics[21] != "" and metrics[42] != "" and \
                metrics[43] != "":
            if 0 >= float(metrics[43]) >= -0.851665 and 2.649965 < float(metrics[42]) <= 3.76662999999:  # In to LHB, up
                if metrics[23] == "InPlay" and metrics[26] in ["Single", "Double", "Triple", "HomeRun"]:
                    slg_locations[categorize_pitch(metrics[21])][0][0] += count_total_bases(metrics[26])
                    slg_locations[categorize_pitch(metrics[21])][0][1] += 1
                elif (metrics[23] == "InPlay" and metrics[26] not in ["Single", "Double", "Triple", "HomeRun"]) or metrics[24] == "Strikeout":
                    slg_locations[categorize_pitch(metrics[21])][0][1] += 1
            elif 0.851665 >= float(metrics[43]) > 0 and 2.649965 < float(metrics[42]) <= 3.76662999999:  # In to RHB, up
                if metrics[23] == "InPlay" and metrics[26] in ["Single", "Double", "Triple", "HomeRun"]:
                    slg_locations[categorize_pitch(metrics[21])][1][0] += count_total_bases(metrics[26])
                    slg_locations[categorize_pitch(metrics[21])][1][1] += 1
                elif (metrics[23] == "InPlay" and metrics[26] not in ["Single", "Double", "Triple", "HomeRun"]) or \
                        metrics[24] == "Strikeout":
                    slg_locations[categorize_pitch(metrics[21])][1][1] += 1
            elif 0 >= float(metrics[43]) >= -0.851665 and 1.5333 < float(metrics[42]) <= 2.649965:  # In to LHB, down
                if metrics[23] == "InPlay" and metrics[26] in ["Single", "Double", "Triple", "HomeRun"]:
                    slg_locations[categorize_pitch(metrics[21])][2][0] += count_total_bases(metrics[26])
                    slg_locations[categorize_pitch(metrics[21])][2][1] += 1
                elif (metrics[23] == "InPlay" and metrics[26] not in ["Single", "Double", "Triple", "HomeRun"]) or \
                        metrics[24] == "Strikeout":
                    slg_locations[categorize_pitch(metrics[21])][2][1] += 1
            elif 0.851665 >= float(metrics[43]) > 0 and 1.5333 < float(metrics[42]) <= 2.649965:  # In to LHB, down
                if metrics[23] == "InPlay" and metrics[26] in ["Single", "Double", "Triple", "HomeRun"]:
                    slg_locations[categorize_pitch(metrics[21])][3][0] += count_total_bases(metrics[26])
                    slg_locations[categorize_pitch(metrics[21])][3][1] += 1
                elif (metrics[23] == "InPlay" and metrics[26] not in ["Single", "Double", "Triple", "HomeRun"]) or \
                        metrics[24] == "Strikeout":
                    slg_locations[categorize_pitch(metrics[21])][3][1] += 1

    file.close()

    # Convert chase rates to percentages
    for pitch in slg_locations:
        for i in range(len(slg_locations[pitch])):
            if slg_locations[pitch][i][1] >= 5:
                slg_locations[pitch][i] = float(slg_locations[pitch][i][0] / slg_locations[pitch][i][1])
            else:
                slg_locations[pitch][i] = "-"

    return slg_locations


def generate_slg_by_location(slg_locations, batter_name, pitcher_hand, plot_num, output_dir):
    """
    Generates color coded chart of slugging percentages by zone quadrant against each pitch type for a given batter.

    :param slg_locations: (dict) A dictionary with slugging percentage by pitch type and zone quadrant
    :param batter_name: (str) Batter name
    :param pitcher_hand: (str) Pitcher hand
    :param plot_num: (int) Plot number for printing
    :param output_dir: (str) File location for saving output
    :return: (pdf) Saves slugging percentage by pitch type and zone quadrant
    """
    # List pitch types
    pitches = list(slg_locations.keys())

    # Loop through each pitch type
    for pitch in pitches:
        if not any(item != "-" for item in slg_locations[pitch]):
            continue

        # Convert data to matrix
        slg_matrix = [[slg_locations[pitch][0], slg_locations[pitch][1]],
                        [slg_locations[pitch][2], slg_locations[pitch][3]]]

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.axis('off')
        table = Table(ax, bbox=[0, 0, 1, 1])

        # Define the min, mid, and max values for your scale
        min_value = 0
        mid_value = 0.45
        max_value = 1

        # Create a colormap and a TwoSlopeNorm normalization
        cmap = plt.get_cmap('coolwarm')
        norm = mcolors.TwoSlopeNorm(vmin=min_value, vcenter=mid_value, vmax=max_value)

        # Add cells to table, format
        for i, row in enumerate(slg_matrix):
            for j, cell in enumerate(row):
                cell_height = 0.05

                # Ensure cell contains numerical values
                if isinstance(cell, (float, int)):
                    cell_value = float(cell)

                    # Normalize the cell value based on the custom normalization
                    norm_cell_value = norm(cell_value)

                    # Get the color from the colormap
                    cell_color = cmap(norm_cell_value)

                    # Convert slg to conventional string
                    cell_text = format_decimal(cell_value)  # Example: format as a string with 2 decimal places

                    # Add the cell with the specified color
                    table.add_cell(i, j, 0.1, cell_height, text=cell_text, loc='center', facecolor=cell_color)
                else:
                    table.add_cell(i, j, 0.1, cell_height, text=cell, loc='center')

        # Set formatting and title
        plt.tight_layout(pad=2.0)
        table.auto_set_font_size(False)
        ax.add_table(table)
        plt.title(f"{batter_name} {pitch} SLG% vs. {pitcher_hand} (P. View)", fontweight="bold",
                  fontsize=12)

        # Save the output
        output_dir = Path(output_dir)  # Convert to Path object
        output_dir.mkdir(parents=True, exist_ok=True)
        first_last = batter_name.split()
        plt.savefig(output_dir / f"{first_last[-1]}_{plot_num}.pdf")
        plt.close()
        plot_num += 1  # Increment the plot number after each pitch type


def parse_spraychart(filename, batter_name, pitcher_hand):
    """
    Parses a Trackman csv and collects batted ball location data for a given batter.

    :param filename: (str) Name of Trackman csv
    :param batter_name: (str) Batter name
    :param pitcher_hand: (str) Pitcher hand
    :return:
    """
    # Initialize batted ball coordinates
    fb_coordinates = [[], []]
    gb_coordinates = [[], []]

    # Loop through file
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        fix_name(metrics, "Batter")

        # Record distance and bearing on fly balls, popups, and line drives
        if (metrics[10] == batter_name and metrics[8] == pitcher_hand and metrics[23] == "InPlay"
                and metrics[25] != "GroundBall" and metrics[55] != "" and metrics[57] != ""):
            if float(metrics[55]) >= 0:
                fb_coordinates[0].append(float(metrics[55]))
                fb_coordinates[1].append(90 - float(metrics[57]))

        # Record bearing of ground balls. Set distance to the infield arc for interpretability
        elif (metrics[10] == batter_name and metrics[8] == pitcher_hand and metrics[23] == "InPlay"
              and metrics[25] == "GroundBall" and metrics[55] != "" and metrics[57] != ""):
            if float(metrics[55]) >= 0:
                # Convert angle to radians
                angle = math.radians(90 - float(metrics[57]))

                # Calculate distance from the origin to the infield arc at the given angle
                infield_arc_width = 1.42 * 90  # Width of the arc
                infield_arc_height = 1.9 * 90  # Height of the arc
                distance_to_arc = (
                        infield_arc_width * infield_arc_height /
                        math.sqrt(
                            (infield_arc_height * math.cos(angle)) ** 2 + (infield_arc_width * math.sin(angle)) ** 2
                        )
                )

                # Set gb_coordinates[0] to the calculated distance
                gb_coordinates[0].append(distance_to_arc)
                gb_coordinates[1].append(90 - float(metrics[57]))

    file.close()

    return fb_coordinates, gb_coordinates


def generate_spraychart(fb_coordinates, gb_coordinates, batter_name, pitcher_hand, output_dir):
    """
    Generates and saves spraycharts vs. RHP and LHP for a given hitter.

    :param fb_coordinates: (list) Distance/bearing of batted balls in the air
    :param gb_coordinates: (list) Distance/bearing of batted balls on the ground
    :param batter_name: (str) Batter name
    :param pitcher_hand: (str) Pitcher hand
    :param output_dir: (str) File location for saving outputs
    :return: (pdf) Saves spray charts vs. RHP and LHP
    """
    # Convert polar coordinates to Cartesian coordinates
    def polar_to_cartesian(r, theta):
        x = r * np.cos(np.radians(theta))
        y = r * np.sin(np.radians(theta))
        return x, y

    # Construct baseball field
    def plot_baseball_field(ax, radius=330, infield_radius=90):
        # Plot the infield square
        infield_square = np.array([
            [0, 0],
            [0, 90],
            [90, 90],
            [90, 0],
            [0, 0]
        ])
        rotation_matrix = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                                    [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
        rotated_infield_square = infield_square @ rotation_matrix.T
        ax.plot(rotated_infield_square[:, 0], rotated_infield_square[:, 1], 'k-')

        # Plot the bases
        bases = [(0, 0), (0, 90), (90, 90), (90, 0)]
        rotated_bases = [np.dot(rotation_matrix, base) for base in bases]
        for i in range(len(rotated_bases)):
            if i == 0:
                vertices = [
                    (rotated_bases[i][0] - 7.5, rotated_bases[i][1] + 7.5),  # top left
                    (rotated_bases[i][0] + 7.5, rotated_bases[i][1] + 7.5),  # top right
                    (rotated_bases[i][0] + 7.5, rotated_bases[i][1] - 2.5),  # bottom right
                    (rotated_bases[i][0], rotated_bases[i][1] - 7.5),  # bottom center
                    (rotated_bases[i][0] - 7.5, rotated_bases[i][1] - 2.5),  # bottom left
                    (rotated_bases[i][0] - 7.5, rotated_bases[i][1] + 7.5)  # closing the shape
                ]
                home_plate = patches.Polygon(vertices, closed=True, edgecolor='black', facecolor='white')
                ax.add_patch(home_plate)
            else:
                ax.add_patch(
                    plt.Rectangle((rotated_bases[i][0] - 7.5, rotated_bases[i][1] - 7.5), 15, 15, edgecolor='black',
                                  facecolor='white'))

        # Plot the pitcher's mound
        rotated_pitchers_mound = np.dot(rotation_matrix, [45, 45])
        ax.add_patch(
            plt.Circle((rotated_pitchers_mound[0], rotated_pitchers_mound[1]), 9, edgecolor='black', facecolor='none'))

        # Plot the infield arc
        infield_arc = patches.Arc((0, 0), 2.9 * infield_radius, 3.8 * infield_radius, theta1=45, theta2=135,
                                  edgecolor='black')
        ax.add_patch(infield_arc)

        # Plot the outfield arc
        outfield_arc = patches.Arc((0, 0), 1.85 * radius, 2.2 * radius, theta1=45, theta2=135, edgecolor='black')
        ax.add_patch(outfield_arc)

        # Plot the foul lines
        ax.plot([0, -radius / np.sqrt(2)], [0, radius / np.sqrt(2)], 'k-')
        ax.plot([0, radius / np.sqrt(2)], [0, radius / np.sqrt(2)], 'k-')

    # Set the figure size (width, height in inches)
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the baseball field
    plot_baseball_field(ax, radius=340, infield_radius=90)

    # Convert polar coordinates to Cartesian coordinates
    for coordinates in [fb_coordinates, gb_coordinates]:
        r_values = []
        theta_values = []
        x_coordinates = [value for value in coordinates[0]]
        y_coordinates = [value for value in coordinates[1]]
        for r, theta in zip(x_coordinates, y_coordinates):
            x, y = polar_to_cartesian(r, theta)
            r_values.append(x)
            theta_values.append(y)

        # Plot fly balls, popups, and line drives with normalized heatmap
        if coordinates == fb_coordinates:

            # Create the 2D histogram (density plot)
            if r_values and theta_values:  # Only plot if there are coordinates
                heatmap, xedges, yedges = np.histogram2d(r_values, theta_values, bins=(80, 80))
                heatmap = gaussian_filter(heatmap, sigma=8)

                img = ax.imshow(heatmap.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                    cmap='coolwarm', alpha=0.5)
                # cbar = fig.colorbar(img, ax=ax)
                # cbar.set_label('Frequency')

                ax.scatter(r_values, theta_values, color='black', s=5)  # Scatter plot of the actual coordinates

            # Add lines from the origin to each point
            for x, y in zip(r_values, theta_values):
                ax.plot([0, x], [0, y], color='black', linewidth=0.15,  alpha=0.5)

        # Plot ground balls
        else:

            # Add dotted lines from the origin to the infield arc
            for x, y in zip(r_values, theta_values):
                ax.plot([0, x], [0, y], color='black', linewidth=0.6, linestyle=':', alpha=1)

    # Set the axis limits to scale the field
    ax.set_xlim(-250, 250)  # Adjust as needed to fit the entire field and the batted balls
    ax.set_ylim(-50, 400)  # Adjust as needed to fit the entire field and the batted balls

    # Title and save plot
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.title(batter_name + " Spray Chart vs. " + pitcher_hand + "\n", fontweight="bold")
    first_last = batter_name.split()
    if pitcher_hand == "Right":
        plt.savefig(output_dir / f"{first_last[-1]}_2.pdf", bbox_inches='tight', pad_inches=0.6)
    else:
        plt.savefig(output_dir / f"{first_last[-1]}_3.pdf", bbox_inches='tight', pad_inches=0.6)
    plt.close()


def parse_slash(filename, batter_name):
    """
    Parses a Trackman csv and calculate slash lines by pitch type and pitcher hand for a given batter.

    :param filename: (str) Name of Trackman csv
    :param batter_name: (str) Batter name
    :return: (matrix) A matrix containing slash lines by pitch type and pitcher hand
    """
    # Organize pitch types to maximize sample size, simplify output
    pitch_categories = {
        "4S": ["Fastball", "FourSeamFastBall", "Cutter"],
        "SNK": ["TwoSeamFastBall", "Sinker"],
        "CH": ["ChangeUp", "Splitter", "Knuckleball"],
        "SL": ["Slider", "Sweeper"],
        "CB": ["Curveball"],
        "Overall": ["Fastball", "FourSeamFastBall", "TwoSeamFastBall", "Sinker", "Cutter", "ChangeUp", "Splitter", "Slider", "Sweeper", "Curveball", "Knuckleball"]
    }

    # Initialize data dictionaries for RHP and LHP
    def initialize_dict():
        return {category: [0, 0, 0, 0, 0] for category in pitch_categories}

    right_dict = initialize_dict()
    left_dict = initialize_dict()

    # Function that records at bat data
    def update_pitch_dict(metrics, pitch_dict):
        """
        Update the pitch dictionary with metrics for either right or left-handed pitchers
        :param metrics: List of metrics from the CSV row
        :param pitch_dict: right_dict or left_dict
        """
        if metrics[23] == "InPlay":
            pitch_dict[4] += 1
            if metrics[26] != "Sacrifice":
                pitch_dict[3] += 1

            if metrics[26] == "Single":
                pitch_dict[0] += 1
                pitch_dict[1] += 1
            elif metrics[26] == "Double":
                pitch_dict[0] += 1
                pitch_dict[1] += 2
            elif metrics[26] == "Triple":
                pitch_dict[0] += 1
                pitch_dict[1] += 3
            elif metrics[26] == "HomeRun":
                pitch_dict[0] += 1
                pitch_dict[1] += 4

        if metrics[24] == "Strikeout":
            pitch_dict[4] += 1
            pitch_dict[3] += 1

        if metrics[24] == "Walk" or metrics[23] == "HitByPitch":
            pitch_dict[4] += 1
            pitch_dict[2] += 1

    # Loop through the file
    with open(filename, "r") as file:
        file.readline()
        for line in file:
            metrics = line.split(",")
            fix_name(metrics, "Batter")

            # Call recording function, organize by pitch types
            if metrics[10] == batter_name:
                for category, pitches in pitch_categories.items():
                    if metrics[21] in pitches:
                        if metrics[8] == "Right":
                            update_pitch_dict(metrics, right_dict[category])
                        else:
                            update_pitch_dict(metrics, left_dict[category])

    # Calculate slash lines
    for slash_dict in [right_dict, left_dict]:
        for category in pitch_categories.keys():
            item = slash_dict[category]
            formatted_string = (format_decimal(safe_division(item[0], item[3])) + " / "
                                + format_decimal(safe_division((item[0] + item[2]), item[4])) + " / "
                                + format_decimal(safe_division((item[1]), item[3])))
            slash_dict[category] = [formatted_string, str(item[4])]

    # Convert data to matrix
    slash_matrix = [["", "Slash vs. R", "PA", "Slash vs. L", "PA"]]
    for category in pitch_categories.keys():
        row = [category, right_dict[category][0], right_dict[category][1], left_dict[category][0], left_dict[category][1]]
        slash_matrix.append(row)

    return slash_matrix


def generate_slash(slash_matrix, batter_name, output_dir):
    """
    Generates and saves slash lines vs. RHP and LHP for a given hitter.

    :param slash_matrix: (matrix) A matrix containing slash lines by pitch type and pitcher hand
    :param batter_name: (str) Batter name
    :param output_dir: (str) File location for saving output
    :return: (pdf) Saves slash lines vs. RHP and LHP
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')
    table = Table(ax, bbox=[0, 0, 1, 1])
    col_widths = [0.1, 0.1, 0.05, 0.1, 0.05]

    # Add cells to table, format
    for i, row in enumerate(slash_matrix):
        for j, cell in enumerate(row):

            # Set cell color
            light_orange = (1.0, 0.8, 0.4)
            light_cyan = (0.8, 1.0, 1.0)

            if i == 0:
                face_color = 'lightgray'
            else:
                face_color = "white"

            if j == 0 and i != 0:
                if "4S" in cell:
                    face_color = light_orange
                elif "SNK" in cell:
                    face_color = light_cyan
                elif "CH" in cell:
                    face_color = "lightgreen"
                elif "SL" in cell:
                    face_color = "lightyellow"
                elif "CB" in cell:
                    face_color = "lightblue"
                elif "Overall" in cell:
                    face_color = 'lightgray'
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

    # Format
    plt.tight_layout(pad=2.0)
    table.auto_set_font_size(False)
    ax.add_table(table)

    # Save the plot
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    first_last = batter_name.split()
    plt.title(batter_name + " Slash Lines", fontweight="bold", fontsize=font_properties['size'] + 1)
    plt.savefig(output_dir / f"{first_last[-1]}_1.pdf")
    plt.close()


def parse_contact(filename, batter_name, pitcher_hand):
    """
    Parses a Trackman csv to calculate contact metrics by pitch type and pitcher hand for a given batter.

    :param filename: (str) Name of Trackman csv
    :param batter_name: (str) Batter name
    :param pitcher_hand: (str) Pitcher hand
    :return: (matrix) Matrix with contact metrics
    """
    # Allow for filtering by pitcher hand
    if pitcher_hand == "Right":
        pitcher_hand = ["Right"]
    elif pitcher_hand == "Left":
        pitcher_hand = ["Left"]
    elif pitcher_hand == "Both":
        pitcher_hand = ["Right", "Left"]

    # Organize pitch types to maximize sample size, simplify output
    pitch_categories = {
        "4S": ["Fastball", "FourSeamFastBall", "Cutter"],
        "SNK": ["TwoSeamFastBall", "Sinker"],
        "CH": ["ChangeUp", "Splitter"],
        "SL": ["Slider", "Sweeper"],
        "CB": ["Curveball", "Knuckleball"],
        "Overall": ["Fastball", "FourSeamFastBall", "TwoSeamFastBall", "Sinker", "Cutter", "ChangeUp", "Splitter",
                    "Slider", "Sweeper", "Curveball", "Knuckleball"]
    }

    # Initialize dictionaries by pitch type
    def initialize_dict():
        return {
            "Z-Swing": 0, "O-Swing": 0, "Whiff%": 0, "K%": 0, "BB%": 0, "z_count": 0, "o_count": 0, "pa_count": 0
        }

    fourseam_dict = initialize_dict()
    sinker_dict = initialize_dict()
    changeup_dict = initialize_dict()
    slider_dict = initialize_dict()
    curveball_dict = initialize_dict()
    overall_dict = initialize_dict()

    # Link data dictionaries to corresponding pitch abbreviation
    pitch_dicts = {
        "4S": fourseam_dict,
        "SNK": sinker_dict,
        "CH": changeup_dict,
        "SL": slider_dict,
        "CB": curveball_dict,
        "Overall": overall_dict
    }

    # Function that records contact data
    def update_dict(metrics, contact_dict):
        if metrics[42] != "" and metrics[43] != "":
            if -(1.5 / 2) <= float(metrics[43]) <= (1.5 / 2) and 1.7 <= float(metrics[42]) <= 3.6:  # All pitches in zone
                contact_dict["z_count"] += 1
                if metrics[23] in ["FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]:
                    contact_dict["Z-Swing"] += 1
                elif metrics[23] == "StrikeSwinging":
                    contact_dict["Z-Swing"] += 1
                    contact_dict["Whiff%"] += 1
            else:  # All pitches out of zone
                contact_dict["o_count"] += 1
                if metrics[23] in ["FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]:
                    contact_dict["O-Swing"] += 1
                elif metrics[23] == "StrikeSwinging":
                    contact_dict["O-Swing"] += 1
                    contact_dict["Whiff%"] += 1

        if metrics[23] in ["InPlay", "HitByPitch"]:
            contact_dict["pa_count"] += 1
        elif metrics[24] == "Strikeout":
            contact_dict["K%"] += 1
            contact_dict["pa_count"] += 1
        elif metrics[24] == "Walk":
            contact_dict["BB%"] += 1
            contact_dict["pa_count"] += 1

    # Loop through file
    with open(filename, "r") as file:
        file.readline()  # Skip the header
        for line in file:
            metrics = line.split(",")
            fix_name(metrics, "Batter")

            if metrics[10] == batter_name and metrics[8] in pitcher_hand:
                pitch_type = metrics[21]

                # Call recording function, organize by pitch types
                for category, types in pitch_categories.items():
                    if pitch_type in types:
                        update_dict(metrics, pitch_dicts[category])

    # Function to process raw data
    def calculate_metrics(contact_dict):
        if contact_dict["pa_count"] > 0:
            k = round(100 * (contact_dict["K%"] / contact_dict["pa_count"]), 1)
            bb = round(100 * (contact_dict["BB%"] / contact_dict["pa_count"]), 1)
        else:
            k = "-"
            bb = "-"
        if (contact_dict["Z-Swing"] + contact_dict["O-Swing"]) > 0:
            whiff = round(100 * (contact_dict["Whiff%"] / (contact_dict["Z-Swing"] + contact_dict["O-Swing"])), 1)
        else:
            whiff = "-"
        if contact_dict["z_count"] > 0:
            z_swing = round(100 * (contact_dict["Z-Swing"] / contact_dict["z_count"]), 1)
        else:
            z_swing = "-"
        if contact_dict["o_count"] > 0:
            o_swing = round(100 * (contact_dict["O-Swing"] / contact_dict["o_count"]), 1)
        else:
            o_swing = "-"
        return [z_swing, o_swing, whiff, k, bb]

    # Create the contact matrix with processed data
    contact_matrix = [
        ["", "Zone\nSwing%", "Chase%", "Whiff%", "K%", "BB%"]
    ]
    contact_matrix.append(["4S"] + calculate_metrics(fourseam_dict))
    contact_matrix.append(["SNK"] + calculate_metrics(sinker_dict))
    contact_matrix.append(["CH"] + calculate_metrics(changeup_dict))
    contact_matrix.append(["SL"] + calculate_metrics(slider_dict))
    contact_matrix.append(["CB"] + calculate_metrics(curveball_dict))
    contact_matrix.append(["Overall"] + calculate_metrics(overall_dict))

    return contact_matrix


def generate_contact(contact_matrix, batter_name, pitcher_hand, output_dir):
    """
    Generates and saves a table with contact metrics by pitch types/pitcher hand for a given batter.

    :param contact_matrix: (matrix) Matrix with contact metrics
    :param batter_name: (str) Batter name
    :param pitcher_hand: (str) Pitcher hand
    :param output_dir: (str) File location for saving output
    :return: (pdf) Saves contact metrics visual
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis('off')
    table = Table(ax, bbox=[0, 0, 1, 1])

    # Add properly formatted cells to the table
    for i, row in enumerate(contact_matrix):
        for j, cell in enumerate(row):

            # Set cell color
            light_orange = (1.0, 0.8, 0.4)
            light_cyan = (0.8, 1.0, 1.0)

            if i == 0:
                face_color = 'lightgray'
            else:
                face_color = "white"

            if j == 0 and i != 0:
                if "4S" in cell:
                    face_color = light_orange
                elif "SNK" in cell:
                    face_color = light_cyan
                elif "CH" in cell:
                    face_color = "lightgreen"
                elif "SL" in cell:
                    face_color = "lightyellow"
                elif "CB" in cell:
                    face_color = "lightblue"
                elif "Overall" in cell:
                    face_color = 'lightgray'
                else:
                    face_color = "white"

            # Set text settings
            font_weight = 'bold' if i == 0 or j == 0 else 'normal'
            cell_height = 0.025 if i == 0 else 0.04
            cell_text = table.add_cell(i, j, 0.1, cell_height, text=cell, loc='center', facecolor=face_color)
            cell_text.set_text_props(weight=font_weight)

    # Save the plot
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=2.0)
    table.auto_set_font_size(False)
    ax.add_table(table)
    plt.title(batter_name + " Contact and Plate Discipline vs. " + pitcher_hand,
              fontweight="bold", fontsize=font_properties['size'] + 1)
    first_last = batter_name.split()
    if pitcher_hand == "Right":
        plt.savefig(output_dir / f"{first_last[-1]}_100.pdf")
    else:
        plt.savefig(output_dir / f"{first_last[-1]}_101.pdf")
    plt.close()


def parse_batted_balls(filename, batter_name, pitcher_hand):
    """
    Parses a Trackman csv to calculate batted ball metrics by pitch type and pitcher hand for a given batter.

    :param filename: (str) Name of Trackman csv
    :param batter_name: (str) Batter name
    :param pitcher_hand: (str) Pitcher hand
    :return: (matrix) Matrix with batted ball metrics
    """
    # Organize pitch types to maximize sample size, simplify output
    pitch_categories = {
        "4S": ["Fastball", "FourSeamFastBall", "Cutter"],
        "SNK": ["TwoSeamFastBall", "Sinker"],
        "CH": ["ChangeUp", "Splitter"],
        "SL": ["Slider", "Sweeper"],
        "CB": ["Curveball", "Knuckleball"],
        "Overall": ["Fastball", "FourSeamFastBall", "TwoSeamFastBall", "Sinker", "Cutter", "ChangeUp", "Splitter",
                    "Slider", "Sweeper", "Curveball", "Knuckleball"]
    }

    # Initialize data dictionaries by pitch type
    def initialize_dict():
        return {
            "EV": [], "LA": [], "GB": 0, "LD": 0, "FB": 0,
            "bbe_counter": 0, "total_la": 0, "total_ev": 0
        }

    fourseam_dict = initialize_dict()
    sinker_dict = initialize_dict()
    changeup_dict = initialize_dict()
    slider_dict = initialize_dict()
    curveball_dict = initialize_dict()
    overall_dict = initialize_dict()

    # Link data dictionaries to corresponding pitch abbreviation
    pitch_dicts = {
        "4S": fourseam_dict,
        "SNK": sinker_dict,
        "CH": changeup_dict,
        "SL": slider_dict,
        "CB": curveball_dict,
        "Overall": overall_dict
    }

    # Function that records batted ball data
    def update_dict(metrics, batted_ball_dict):
        batted_ball_dict["bbe_counter"] += 1
        ev = float(metrics[48])
        la = float(metrics[49])
        batted_ball_dict["EV"].append(ev)
        batted_ball_dict["LA"].append(la)

        if metrics[25] == "GroundBall":
            batted_ball_dict["GB"] += 1
        elif metrics[25] == "LineDrive":
            batted_ball_dict["LD"] += 1
        elif metrics[25] == "FlyBall":
            batted_ball_dict["FB"] += 1

        batted_ball_dict["total_ev"] += ev
        # if ev >= 95:
        #     batted_ball_dict["hard_hit"] += 1

        batted_ball_dict["total_la"] += la
        # if 8 <= la <= 32:
        #     batted_ball_dict["sweet_spot"] += 1

    # Loop through file
    with open(filename, "r") as file:
        file.readline()  # Skip the header
        for line in file:
            metrics = line.split(",")
            fix_name(metrics, "Batter")

            if (metrics[10] == batter_name and metrics[8] == pitcher_hand
                    and metrics[23] == "InPlay" and metrics[48] != "" and metrics[49] != ""):
                pitch_type = metrics[21]

                # Call recording function, organize by pitch types
                for category, types in pitch_categories.items():
                    if pitch_type in types:
                        update_dict(metrics, pitch_dicts[category])

    # Function to process raw data
    def calculate_averages(batted_ball_dict):
        if batted_ball_dict["bbe_counter"] == 0:
            return ["-"] * 5

        return [
            round(batted_ball_dict["total_ev"] / batted_ball_dict["bbe_counter"], 1),
            round(batted_ball_dict["total_la"] / batted_ball_dict["bbe_counter"], 1),
            round(100 * batted_ball_dict["GB"] / batted_ball_dict["bbe_counter"], 1),
            round(100 * batted_ball_dict["LD"] / batted_ball_dict["bbe_counter"], 1),
            round(100 * batted_ball_dict["FB"] / batted_ball_dict["bbe_counter"], 1)
        ]

    # Convert data to matrix
    batted_ball_matrix = [
        ["", "Avg. EV\n(mph)", "Avg. LA\n(°)", "GB%", "LD%", "FB%"]
    ]

    for category in ["4S", "SNK", "CH", "SL", "CB", "Overall"]:
        batted_ball_matrix.append([category] + calculate_averages(pitch_dicts[category]))

    return batted_ball_matrix


def generate_batted_balls(batted_ball_matrix, batter_name, pitcher_hand, output_dir):
    """
    Generates and saves a table with batted ball metrics by pitch types/pitcher hand for a given batter.

    :param batted_ball_matrix: (matrix) Matrix with batted ball metrics
    :param batter_name: (str) Batter name
    :param pitcher_hand: (str) Pitcher hand
    :param output_dir: (str) File location for saving output
    :return: (pdf) Saves batted ball metrics visual
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')
    table = Table(ax, bbox=[0, 0, 1, 1])

    # Add properly formatted cells to the table
    for i, row in enumerate(batted_ball_matrix):
        for j, cell in enumerate(row):

            # Set cell color
            light_orange = (1.0, 0.8, 0.4)
            light_cyan = (0.8, 1.0, 1.0)

            if i == 0:
                face_color = 'lightgray'
            else:
                face_color = "white"

            if j == 0 and i != 0:
                if "4S" in cell:
                    face_color = light_orange
                elif "SNK" in cell:
                    face_color = light_cyan
                elif "CH" in cell:
                    face_color = "lightgreen"
                elif "SL" in cell:
                    face_color = "lightyellow"
                elif "CB" in cell:
                    face_color = "lightblue"
                elif "Overall" in cell:
                    face_color = 'lightgray'
                else:
                    face_color = "white"

            font_weight = 'bold' if i == 0 or j == 0 else 'normal'
            cell_height = 0.02 if i == 0 else 0.03

            cell_text = table.add_cell(i, j, 0.1, cell_height, text=cell, loc='center', facecolor=face_color)
            cell_text.set_text_props(weight=font_weight)

    # Display the plot with appropriate formatting and title
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=2.0)
    table.auto_set_font_size(False)
    ax.add_table(table)
    plt.title(batter_name + " Batted Ball Profile vs. " + pitcher_hand, fontweight="bold", fontsize=font_properties['size'] + 1)
    first_last = batter_name.split()
    if pitcher_hand == "Right":
        plt.savefig(output_dir / f"{first_last[-1]}_102.pdf")
    else:
        plt.savefig(output_dir / f"{first_last[-1]}_103.pdf")
    plt.close()


def create_cover(batter_name, handedness, team):
    """
    Generates and saves a hitter report cover.

    :param batter_name: (str) Batter name
    :param handedness: (str) Batter side
    :param team: (str) Team name
    :return: (pdf) Cover image for hitter report
    """
    # Set output path
    first_last = batter_name.split()
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
    c.drawCentredString(width / 2, text_y_start - 30, f"{handedness} {batter_name},")
    c.drawCentredString(width / 2, text_y_start - 60, f"{team}")

    # Save the PDF
    c.save()


def combine_pdfs(batter_names, directory):
    """
    Compiles saved PDFs under each batter name into one file each.

    :param batter_names: (list) List of batter names
    :param directory: (str) Location for saving hitter reports
    :return: (pdf) Hitter reports
    """
    # Iterate over all batters
    for batter_name in batter_names:
        # Initialize a PdfMerger object
        merger = PdfMerger()

        # Look for files matching the pattern {pitcher_name}_{i}.pdf
        first_last = batter_name.split()
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
                print(f"Saved final PDF: {first_last[-1]}.pdf")

            # Delete the individual PDFs
            for file in files_to_merge:
                file_path = os.path.join(directory, file)
                os.remove(file_path)

        # Close the merger
        merger.close()


def compile_pdf(batter_name, output_dir):
    """
    Compiles saved PDFs under one name into one file, collates pages (4 image to a sheet).

    :param batter_name: (str) Batter name
    :param output_dir: (str) Location for saving hitter report
    :return: (pdf) Hitter report
    """
    # Set file name
    first_last = batter_name.split()
    output_file = f'/Users/relliott/Desktop/AAAA/{first_last[-1]}.pdf'

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


def main():
    """
    Generates hitter scouting reports.
    """
    # Input parameters
    filename = "csvs/yhwpreseries.csv"
    batter_names = [
        "Kirkland Banks",
        "Braedon Blackford",
        "Edwin DeLaCruz",
        "Alejandro Figueredo",
        "David Glancy",
        "Jose Gonzalez",
        "Travis Holt",
        "Bobby Lada",
        "Taylor Lomack",
        "Braylin Marine",
        "Angel Mendoza",
        "Brayland Skinner",
        "Tanner Smith"
    ]
    team = "Yolo High Wheelers"
    team_code = "YOL_HIG"

    # Non-input parameters
    pitcher_hand = ["Right", "Left"]
    output_dir = "/Users/relliott/Desktop/AAAA"
    hitter_dict = {}

    # Collect batter side information
    with (open(filename, "r") as file):
        file.readline()  # Skip the header line
        for line in file:
            metrics = line.split(",")
            fix_name(metrics, "Batter")

            if metrics[14] == team_code and metrics[10] not in hitter_dict.keys():

                if metrics[13] == "Right":
                    hitter_dict[metrics[10]] = "RHH"
                else:
                    hitter_dict[metrics[10]] = "LHH"

            if metrics[14] == team_code and metrics[10] in hitter_dict.keys():
                if (metrics[13] == "Right" and hitter_dict[metrics[10]] != "RHH") or (metrics[13] == "Left" and hitter_dict[metrics[10]] != "LHH"):
                    hitter_dict[metrics[10]] = "SHH"

    file.close()

    # Generate a report for each hitter
    for batter_name in batter_names:

        # 4 IMAGES TO A SHEET. Run the following together

        # Generate cover image
        create_cover(batter_name, hitter_dict[batter_name], team)

        slash_matrix = parse_slash(filename, batter_name)
        generate_slash(slash_matrix, batter_name, output_dir)

        for hand in pitcher_hand:

            # Plot spray chart
            fb_coordinates, gb_coordinates = parse_spraychart(filename, batter_name, hand)
            generate_spraychart(fb_coordinates, gb_coordinates, batter_name, hand, output_dir)

        for hand in pitcher_hand:

            # Display contact and plate discipline info
            contact = parse_contact(filename, batter_name, hand)
            generate_contact(contact, batter_name, hand, output_dir)

        for hand in pitcher_hand:

            # Plot batted ball profile
            batted_ball_matrix = parse_batted_balls(filename, batter_name, hand)
            generate_batted_balls(batted_ball_matrix, batter_name, hand, output_dir)

            # Compile pdfs - requires terminal, automatically collates (4 images to a sheet)
            compile_pdf(batter_name, output_dir)

        # 9 IMAGES TO A SHEET.  Run the following together

        # plot_num = 8
        #
        # for hand in pitcher_hand:
        #
        #     # Plot chase location heatmaps
        #     chase_locations, chase_rates = parse_chase(filename, batter_name, hand)
        #     generate_chase(chase_locations, chase_rates, batter_name, hand, plot_num, output_dir)
        #     plot_num += len(chase_locations)  # Increment plot_num based on the number of pitch types
        #
        # for hand in pitcher_hand:
        #
        #     # Plot whiffs by location
        #     whiff_locations = parse_whiff_by_location(filename, batter_name, hand)
        #     generate_whiff_by_location(whiff_locations, batter_name, hand, plot_num, output_dir)
        #     plot_num += len(whiff_locations)
        #
        # for hand in pitcher_hand:
        #
        #     # Plot slugging percentage by location
        #     slg_locations = parse_slg_by_location(filename, batter_name, hand)
        #     generate_slg_by_location(slg_locations, batter_name, hand, plot_num, output_dir)
        #     plot_num += len(slg_locations)

    # # Combine pdfs - no terminal, does not collate
    # combine_pdfs(batter_names, output_dir)


if __name__ == '__main__':
    main()

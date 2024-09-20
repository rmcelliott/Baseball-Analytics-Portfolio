"""
    Project Title: Ballers Baseball Hitter Postgame Reports

    Author: Riley Elliott

    Date:   9/19/2024
"""

import os
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.table import Table
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch

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
    elif tagged_pitch == "Splitter":
        return "SPL"
    elif tagged_pitch == "Undefined" or tagged_pitch == "":
        return "?"
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
        return "ERROR"


def parse_plate_appearances(filename, batter_name, line_count):
    """
    Parses a Trackman csv and creates a visualization of a given plate appearance for a given batter.

    :param filename: (str) Name of a Trackman file
    :param batter_name: (str) Batter name
    :param line_count: (int) Line of the Trackman file to start on. Helps locate the correct plate appearance
    :return: Information about a plate appearance, including game context, pitch types/locations/results, etc.
    """
    # Initialize the location dictionary and list of pitches
    pitch_sequence = []
    outcome = ""
    pitcher_name = ""
    inning = ""
    pitch_table = [['Pitch', 'Count', 'Type', 'Velo', 'Result']]
    review_coordinates = []

    # Flag to start recording data
    record_data = False

    # Skip through the file until the desired line is found
    file = open(filename, "r")
    if line_count == 0:
        file.readline()
    else:
        for i in range(line_count):
            file.readline()

    # Loop through the file
    for line in file:
        line_count += 1
        metrics = line.split(",")
        fix_name(metrics, "Batter")
        fix_name(metrics, "Pitcher")

        # Start recording when batter name is found
        if metrics[10] == batter_name:
            record_data = True

        # Stop recording if the batter name is no longer found
        if record_data and metrics[10] != batter_name:
            break

        # Record data
        if metrics[10] == batter_name and metrics[42] != "" and metrics[43] != "":
            inning = str(metrics[16])
            pitcher_name = metrics[5]

            # Note coordinates of reviewed pitches
            if metrics[29] != "":
                review_coordinates.append([float(metrics[43]), float(metrics[42])])

            # Note what happened on each pitch. Keep track of pitch sequence
            if metrics[23] == "BallCalled" or metrics[23] == "BallinDirt":
                result = "Ball"
                pitch_sequence.append(("Take", metrics[43], metrics[42]))
            elif metrics[23] == "StrikeCalled":
                result = "Take"
                pitch_sequence.append(("Take", metrics[43], metrics[42]))
            elif metrics[23] == "HitByPitch":
                result = "HBP"
                outcome = "HBP"
                pitch_sequence.append(("Take", metrics[43], metrics[42]))
            elif metrics[23] == "StrikeSwinging":
                result = "Whiff"
                pitch_sequence.append(("Swing", metrics[43], metrics[42]))
            elif metrics[23] == "FoulBall" or metrics[23] == "FoulBallFieldable" or metrics[23] == "FoulBallNotFieldable":
                result = "Foul"
                pitch_sequence.append(("Swing", metrics[43], metrics[42]))
            elif metrics[23] == "InPlay":
                result = "IP"
                pitch_sequence.append(("Swing", metrics[43], metrics[42]))
                if metrics[48] != "" or metrics[49] != "" or metrics[55] != "":
                    if metrics[26] != "HomeRun":
                        outcome = (
                                str(abbreviate_result(metrics[25]))
                                + " "
                                + str(abbreviate_result(metrics[26]))
                                + ": "
                                + str(round(float(metrics[48]), 1))
                                + " mph, "
                                + str(round(float(metrics[49]), 1))
                                + "°"
                        )
                    else:
                        outcome = (
                                str(abbreviate_result(metrics[25]))
                                + " "
                                + str(abbreviate_result(metrics[26]))
                                + ": "
                                + str(round(float(metrics[48]), 1))
                                + " mph, "
                                + str(round(float(metrics[49]), 1))
                                + "°, "
                                + str(int(round(float(metrics[55]), 0)))
                                + " ft"
                        )
                else:
                    outcome = str(abbreviate_result(metrics[25])) + " " + str(abbreviate_result(metrics[26]))
            if metrics[24] != "Undefined":
                outcome = str(abbreviate_result(metrics[24]))

            # Append information to pitch table
            if metrics[30] != "":
                pitch_table.append([str(metrics[4]), str(metrics[19]) + "-" + str(metrics[20]), str(abbreviate_pitch(metrics[21])), str(round(float(metrics[30]), 1)), result])
            else:
                pitch_table.append([str(metrics[4]), str(metrics[19]) + "-" + str(metrics[20]), str(abbreviate_pitch(metrics[21])), "-", result])

        # Record data, but in the case that pitch location data is not available
        elif metrics[10] == batter_name and metrics[42] == "" and metrics[43] == "":
            inning = str(metrics[16])
            pitcher_name = metrics[5]
            if metrics[23] == "BallCalled" or metrics[23] == "BallinDirt" or metrics[23] == "AutomaticBall" or metrics[23] == "BallIntentional":
                result = "Ball"
                pitch_sequence.append(("Take", "-", "-"))
            elif metrics[23] == "StrikeCalled":
                result = "Take"
                pitch_sequence.append(("Take", "-", "-"))
            elif metrics[23] == "HitByPitch":
                result = "HBP"
                outcome = "HBP"
                pitch_sequence.append(("Take", "-", "-"))
            elif metrics[23] == "StrikeSwinging":
                result = "Whiff"
                pitch_sequence.append(("Swing", "-", "-"))
            elif metrics[23] == "FoulBall" or metrics[23] == "FoulBallFieldable" or metrics[
                23] == "FoulBallNotFieldable":
                result = "Foul"
                pitch_sequence.append(("Swing", "-", "-"))
            elif metrics[23] == "InPlay":
                result = "IP"
                pitch_sequence.append(("Swing", "-", "-"))
                if metrics[48] != "" or metrics[49] != "" or metrics[55] != "":
                    if metrics[26] != "HomeRun":
                        outcome = (
                                str(abbreviate_result(metrics[25]))
                                + " "
                                + str(abbreviate_result(metrics[26]))
                                + ": "
                                + str(round(float(metrics[48]), 1))
                                + " mph, "
                                + str(round(float(metrics[49]), 1))
                                + "°"
                        )
                    else:
                        outcome = (
                                str(abbreviate_result(metrics[25]))
                                + " "
                                + str(abbreviate_result(metrics[26]))
                                + ": "
                                + str(round(float(metrics[48]), 1))
                                + " mph, "
                                + str(round(float(metrics[49]), 1))
                                + "°, "
                                + str(int(round(float(metrics[55]), 0)))
                                + " ft"
                        )
                else:
                    outcome = str(abbreviate_result(metrics[25])) + " " + str(abbreviate_result(metrics[26]))
            if metrics[24] != "Undefined":
                outcome = str(abbreviate_result(metrics[24]))

            # Append information to pitch table
            if metrics[30] != "":
                pitch_table.append(
                    [str(metrics[4]), str(metrics[19]) + "-" + str(metrics[20]), str(abbreviate_pitch(metrics[21])),
                     str(round(float(metrics[30]), 1)), result])
            else:
                pitch_table.append(
                    [str(metrics[4]), str(metrics[19]) + "-" + str(metrics[20]), str(abbreviate_pitch(metrics[21])),
                     "-", result])

    file.close()

    return pitch_table, pitch_sequence, outcome, review_coordinates, pitcher_name, inning, line_count


def generate_plate_appearances(pitch_table, pitch_sequence, outcome, review_coordinates, pitcher_name, inning, batter_name, pa_count, output_dir):
    """
    Generates and saves a visualization of a given plate appearance by a given batter.

    :param pitch_table: (matrix) Pitch-by-pitch description of plate appearance
    :param pitch_sequence: (list) Location coordinates of each pitch in proper sequence. Swing vs. take noted
    :param outcome: (str) Outcome of the plate appearance
    :param review_coordinates: (list) List of the coordinates of pitches that were reviewed
    :param pitcher_name: (str) Pitcher name
    :param inning: (int) Inning
    :param batter_name: (str) Batter name
    :param pa_count: (int) Number of plate appearance in the game for the given batter
    :param output_dir: (str) File location for saving visual
    :return: (pdf) Visualization of a given plate appearance by a given batter
    """
    # Initialize generic and updated labels
    labels = ["Take", "Swing"]
    colors = [(0, 0.3, 1), "red"]
    markers = ["o", "o"]

    # Create a figure and axis
    fig = plt.figure(figsize=(8, 5))
    gs = GridSpec(1, 2, width_ratios=[2, 1])
    ax = fig.add_subplot(gs[0])

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

    # Plot pitch coordinates and collect handles for legend
    handles = []
    legend_labels = []
    for pitch_index, (take_or_swing, x, y) in enumerate(pitch_sequence, start=1):
        if x != "-" and y != "-":
            x, y = float(x), float(y)  # Convert coordinates to floats
            color = colors[labels.index(take_or_swing)]
            marker = markers[labels.index(take_or_swing)]
            scatter = ax.scatter(x, y, color=color, marker=marker, s=140)
            ax.annotate(str(pitch_index), (x, y), textcoords="offset points", xytext=(0, 5), ha='center')
            if take_or_swing not in legend_labels:
                handles.append(scatter)
                legend_labels.append(take_or_swing)

    # Circle reviewed pitches
    for review_coord in review_coordinates:
        review_x, review_y = review_coord
        circle_radius = 0.2
        circle = Circle((review_x, review_y), circle_radius, color='black', fill=False)
        ax.add_patch(circle)

    # Set axis limits and labels
    ax.set_xlim(2.5, -2.5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')

    # Add legend with collected handles and labels
    ax.legend(handles, legend_labels, fontsize='9')
    ax.set_xlabel("Width (feet)")
    ax.set_ylabel("Height (feet)")
    ax.text(0.5, 1.1, f"Inning {inning} vs. {pitcher_name}",
            fontweight="bold", fontsize=13, ha='center', transform=ax.transAxes)
    ax.set_title("(Catcher View, Reviews Circled)", fontsize=11)

    # Create pitch table
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    fig.text(0.8, 0.8, outcome, ha='center', va='bottom', fontsize=12, fontweight='bold',
             transform=fig.transFigure)
    table = ax_table.table(cellText=pitch_table, cellLoc='center', loc='center')

    for i, row in enumerate(pitch_table):
        for j, cell in enumerate(row):

            # Set cell color
            light_red = (1.0, 0.8, 0.8)
            light_olive = (0.749, 0.718, 0.42)
            light_orange = (1.0, 0.8, 0.4)
            light_purple = (0.8, 0.6, 0.8)
            light_cyan = (0.8, 1.0, 1.0)

            if i == 0:
                face_color = 'lightgray'
            else:
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

            # Set font weight and cell height for row 1
            if i == 0:
                font_weight = 'bold'
            elif j == 0:
                font_weight = 'bold'
            else:
                font_weight = 'normal'
            cell_height = 0.05
            cell_width = 0.3

            # Set text settings
            cell_text = table.add_cell(i, j, width=cell_width, height=cell_height, text=cell, loc='center', facecolor=face_color)
            cell_text.set_text_props(weight=font_weight)

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Save visual
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(wspace=0.4)
    first_last = batter_name.split()
    plt.savefig(output_dir / f"{first_last[-1]}_{pa_count}.pdf")
    plt.close()


def create_cover(date, batter_name, handedness, team):
    """
    Generates and saves a postgame report cover. Contains information on player, game date, and opponent.

    :param date: (str) Game date
    :param batter_name: (str) Batter name
    :param handedness: (str) Batter side
    :param team: (str) Opposing team name
    :return: (pdf) Cover image for postgame report
    """
    # Set file name and output path
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
    c.drawCentredString(width / 2, text_y_start, "Postgame Report:")
    c.drawCentredString(width / 2, text_y_start - 30, f"{handedness} {batter_name},")
    c.drawCentredString(width / 2, text_y_start - 60, f"{date} vs. {team}")

    # Save the PDF
    c.save()


def compile_pdf(batter_name, output_dir):
    """
    Compiles saved PDFs under one name into one file, collates pages onto one sheet.

    :param batter_name: (str) Batter name
    :param output_dir: (str) Location for saving final postgame report
    :return: (pdf) Final postgame report
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
    print(f"Processed {len(images)} plate appearances.")

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
    Parses game data, generates postgame report for each Oakland batter.
    """
    # Initialize data file, pitcher names
    filename = "/Users/relliott/Desktop/Ballers Analytics/BallersCode/venv/csvs/0913yhw.csv"
    team = "Yolo"
    output_dir = "/Users/relliott/Desktop/AAAA"
    hitter_dict = {}

    # Extract date of game for labeling
    file = open(filename, "r")
    file.readline()
    for line in file:
        metrics = line.split(",")
        if metrics[1] != "":
            date = metrics[1]
            break

    # Collect all unique batter names
    with open(filename, "r") as file:
        file.readline()  # Skip the header line
        for line in file:
            metrics = line.split(",")
            fix_name(metrics, "Batter")

            if metrics[14] == "OAK_BAL" and metrics[10] not in hitter_dict.keys():
                if metrics[13] == "Right":
                    hitter_dict[metrics[10]] = "RHH"
                else:
                    hitter_dict[metrics[10]] = "LHH"

    # For each unique batter in the game csv:
    for batter_name in hitter_dict.keys():

        # Create cover image
        create_cover(date, batter_name, hitter_dict[batter_name], team)

        # Initialize PA and csv line count trackers
        pa_count = 0
        line_count = 0

        # Process each plate appearance for the current batter
        while True:
            result = parse_plate_appearances(filename, batter_name, line_count)
            pitch_table, pitch_sequence, outcome, review_coordinates, pitcher_name, inning, line_count = result

            # Break if no more plate appearances are found
            if not pitch_sequence:
                break

            # Generate visuals for each plate appearance
            pa_count += 1
            generate_plate_appearances(pitch_table, pitch_sequence, outcome, review_coordinates, pitcher_name, inning, batter_name, pa_count, output_dir)

        compile_pdf(batter_name, output_dir)


if __name__ == "__main__":
    main()

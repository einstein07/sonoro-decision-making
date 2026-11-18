import asyncio
import csv
import math
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import matplotlib.pyplot as plt
import qtm_rt  # pip install qtm-rt

# ============================================================
# SETTINGS
# ============================================================

QTM_SERVER = "10.111.21.127"   # QTM host; RT port defaults to 22223

experiment_name = sys.argv[1]

date_prefix = datetime.now().strftime("%d_%m")

CSV_FILE = f"{date_prefix}_{experiment_name}.csv"

# 6DOF body names as defined in QTM (Project Options -> 6DOF Tracking)
ROBOT = "sonoro"
SPEAKER_L = "left"
SPEAKER_R = "right"


# ============================================================
# DATA FOR PLOT
# ============================================================

robot_x, robot_y = [], []
speakerL_x, speakerL_y = [], []
speakerR_x, speakerR_y = [], []


def is_occluded(pos):
    # QTM reports NaN coordinates for an untracked body
    return any(math.isnan(v) for v in (pos.x, pos.y, pos.z))


async def record():

    # --------------------------------------------------------
    # CONNECT TO QTM
    # --------------------------------------------------------

    print("Connecting to QTM...")

    connection = None
    while connection is None:
        connection = await qtm_rt.connect(QTM_SERVER)
        if connection is None:
            print("Connection failed, retrying...")
            await asyncio.sleep(1)

    print("Connected!")

    # Body order in each packet follows the order in the parameter XML
    xml = await connection.get_parameters(parameters=["6d"])
    root = ET.fromstring(xml)
    body_names = [b.find("Name").text for b in root.findall(".//The_6D/Body")]
    body_index = {name: i for i, name in enumerate(body_names)}

    for name in (ROBOT, SPEAKER_L, SPEAKER_R):
        if name not in body_index:
            print(f"ERROR: 6DOF body '{name}' not defined in QTM. "
                  f"Available: {body_names}")
            connection.disconnect()
            return

    # --------------------------------------------------------
    # OPEN CSV
    # --------------------------------------------------------

    csvfile = open(CSV_FILE, "w", newline="")
    writer = csv.writer(csvfile)

    # Same headers as your old collector
    writer.writerow([
        "timestamp",
        "robot_x", "robot_y",
        "speakerL_x", "speakerL_y",
        "speakerR_x", "speakerR_y",
    ])
    csvfile.flush()

    print()
    print("Recording to:")
    print(os.path.abspath(CSV_FILE))
    print()
    print("Press Ctrl+C to stop.")
    print()

    # --------------------------------------------------------
    # PER-FRAME CALLBACK
    # --------------------------------------------------------

    def on_packet(packet):

        frame = packet.framenumber
        timestamp = datetime.now().isoformat(timespec="milliseconds")

        _, bodies = packet.get_6d()

        def body_xy(name):
            pos, _rot = bodies[body_index[name]]
            if is_occluded(pos):
                print(f"WARNING: {name} is occluded")
            return pos.x, pos.y

        x_robot, y_robot = body_xy(ROBOT)
        x_left, y_left = body_xy(SPEAKER_L)
        x_right, y_right = body_xy(SPEAKER_R)

        # SAVE FOR PLOT
        robot_x.append(x_robot); robot_y.append(y_robot)
        speakerL_x.append(x_left); speakerL_y.append(y_left)
        speakerR_x.append(x_right); speakerR_y.append(y_right)

        # WRITE CSV
        writer.writerow([
            timestamp,
            x_robot, y_robot,
            x_left, y_left,
            x_right, y_right,
        ])
        csvfile.flush()

        # CONSOLE
        print(
            f"Frame {frame} | "
            f"{timestamp} | "
            f"Robot=({x_robot:.2f}, {y_robot:.2f}) | "
            f"L=({x_left:.2f}, {y_left:.2f}) | "
            f"R=({x_right:.2f}, {y_right:.2f})"
        )

    # --------------------------------------------------------
    # STREAM
    # --------------------------------------------------------

    await connection.stream_frames(components=["6d"], on_packet=on_packet)

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await connection.stream_frames_stop()
        connection.disconnect()
        csvfile.close()


# ============================================================
# MAIN
# ============================================================

try:
    asyncio.run(record())
except KeyboardInterrupt:
    print()
    print("Recording stopped.")


# ============================================================
# PLOT
# ============================================================

print()
print("Creating plot...")

plt.figure(figsize=(8, 6))

if robot_x:
    plt.plot(robot_x, robot_y, label="Robot")

if speakerL_x:
    plt.plot(speakerL_x, speakerL_y, label="Target 1")

if speakerR_x:
    plt.plot(speakerR_x, speakerR_y, label="Target 2")

plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("Qualisys Trajectories")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()


# ============================================================
# DONE
# ============================================================

print()
print("CSV file saved:")
print(os.path.abspath(CSV_FILE))
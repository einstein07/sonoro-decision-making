import asyncio
import math
import xml.etree.ElementTree as ET

import qtm_rt  # pip install qtm-rt

QTM_SERVER = "10.111.21.127"   # QTM host; RT port defaults to 22223


def is_occluded(pos):
    return any(math.isnan(v) for v in (pos.x, pos.y, pos.z))


async def main():
    print("Connecting...")
    connection = await qtm_rt.connect(QTM_SERVER)
    state = await connection.get_state()
    print("QTM state:", state)
    if connection is None:
        print("Failed to connect")
        return
    print("Connected!")

    # Names of 6DOF bodies and labelled markers come from the parameter XML
    xml = await connection.get_parameters(parameters=["6d", "3d"])
    root = ET.fromstring(xml)
    body_names = [b.find("Name").text for b in root.findall(".//The_6D/Body")]
    marker_names = [m.find("Name").text for m in root.findall(".//The_3D/Label")]

    print("\nWaiting for frame...")
    packet = await connection.get_current_frame(
        components=["6d", "3d", "3dnolabels"]
    )
    print("Frame:", packet.framenumber)

    # --- 6DOF bodies (Vicon subjects/segments) ---
    print("\n6DOF bodies available through RT stream:")
    try:
        _, bodies = packet.get_6d()
        print("Number of bodies:", len(bodies))
        for name, (pos, rot) in zip(body_names, bodies):
            print("  Body:", name)
            print("    Translation:", (pos.x, pos.y, pos.z))   # mm, global frame
            print("    Rotation (3x3, row-major):", rot.matrix)
            print("    Occluded:", is_occluded(pos))
    except Exception as e:
        print("  6DOF ERROR:", repr(e))

    # --- labelled markers ---
    print("\nLabelled markers:")
    try:
        _, markers = packet.get_3d_markers()
        for name, m in zip(marker_names, markers):
            print("  ", name, (m.x, m.y, m.z), "occluded" if is_occluded(m) else "")
    except Exception as e:
        print("  Marker ERROR:", repr(e))

    # --- unlabelled markers ---
    print("\nUnlabelled markers:")
    try:
        _, unlabelled = packet.get_3d_markers_no_label()
        print("  Count:", len(unlabelled))
        for m in unlabelled:
            print("   ", (m.x, m.y, m.z), "id:", m.id)
    except Exception as e:
        print("  Unlabelled ERROR:", repr(e))

    connection.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
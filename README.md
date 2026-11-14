# Sonoro sound–robot experiment protocol

This document describes preparation, planning, execution, recording, and shutdown of the Sonoro experiment. It follows Phase 0 (preparation), Phase 1 (planning and positioning), and Phase 2 (execution).

Use the experiment sheet for each session. Keep experiment names and run IDs consistent across the sheet, the Vicon collector, and the Sonoro script.

---

## Phase 0 — Equipment and system preparation

### 0.1 Equipment checklist

Prepare:

- Sonoro (the Thymio robot)
- Left speaker and right speaker
- A laptop to control the speakers
- A PC to control Sonoro (this may be the same PC as the Vicon PC)
- Adapters needed to connect the speakers to the Mac/laptop
- Required cables, including the AUX cables

Sonoro and the speakers are stored in the box. The speaker-control laptop must be close enough to the speakers for the short AUX cables to reach. It is a separate laptop from the Sonoro-control PC.

The Sonoro-control PC does not have to be physically separate from the Vicon PC. Sonoro can be controlled locally or remotely, for example through SSH or VS Code Remote SSH.

### 0.2 Start Vicon and check tracking subjects

1. Start the Vicon PC.
2. Start the Vicon server so that the camera system is running.
3. Check the Vicon view and confirm that all three required subjects are recognised:
   - Sonoro
   - Left speaker
   - Right speaker
4. Run the Vicon test script:

   ```bash
   python viconScripts/test.py
   ```

5. Confirm that the three subjects are listed/available.

The supplied test script checks the subjects available through the Vicon DataStream. It connects to `localhost:801`, enables segment, marker, and unlabelled-marker data, and prints the available subjects and their segments. Run it where the Vicon DataStream server is reachable at that address (normally on the Vicon PC).

Do not proceed until all three subjects are visible and available. If a subject is missing, check its visibility/recognition in Vicon before continuing.

---

## Phase 1 — Experiment planning and sound–robot positioning

Complete the planning for an experiment before starting its runs.

### 1.1 Decide the arena arrangement

Decide and set up:

- Whether the arrangement is before or after the bifurcation
- The planned starting position of Sonoro
- The planned positions of the left and right speakers
- The distance from Sonoro to each speaker
- The distance between the two speakers

Use Vicon to record/measure positions and distances. These measurements do not need to be copied into the experiment sheet.

Place Sonoro and both speakers in their planned starting positions.

**Speaker labels matter.** Check the physical left/right labels on the speakers carefully. The laptop's speaker-control script drives the outputs according to those labels. Do not swap the connections or rely only on where a speaker happens to appear visually.

### 1.2 Decide the run plan and sound levels

For each experiment condition, decide:

- How many runs will be performed
- The left-speaker sound level
- The right-speaker sound level

The sound controller uses `base_left` and `base_right`, each in the range **0.0 to 1.0**. Enter the chosen values in the script for the planned condition. Record the corresponding left and right values in the experiment sheet.

### 1.3 Fill in the experiment sheet

Use the supplied `experimentSheet.pdf`. Its fields are:

- **Date**
- **Comments**
- **Experiment Name**
- **Right Sound**
- **Left Sound**
- **Run IDs**

For each condition, use experiment names such as `A1`, `A2`, `A3`, `B1`, `B2`, etc. Record the sound values in the matching Right Sound and Left Sound columns.

Record the run IDs for that condition in the Run IDs area. Use a consistent convention such as `A1_01`, `A1_02`, `A1_03`. The shorter run numbers `01`, `02`, `03` are also acceptable in the sheet if the experiment name makes their association unambiguous. The commands in Phase 2 use the full form `A1_01`.

Use the Comments field for relevant session notes.

---

## Phase 2 — Run the experiment

### 2.1 Start the sound controller

Use the laptop positioned near the speakers.

1. Connect the speakers to the laptop using the correct adapters and AUX connections.
2. Check the speaker labels and connections: left output to the labelled left speaker, right output to the labelled right speaker.
3. Open `soundScript/sound_generator_switch.py`.
4. Set the sound levels for the current experiment by editing:

   ```python
   base_left = 1.0
   base_right = 1.0
   ```

   Replace these example values with the planned values (each between `0.0` and `1.0`) recorded in the experiment sheet.

5. Run the sound controller from the directory containing `soundScript/`:

   ```bash
   python soundScript/sound_generator_switch.py
   ```

The supplied script is configured for a total playback duration of **6000 seconds (100 minutes)**. It produces a short chirp on the left output, then on the right output, alternating at one-second intervals. The sound controller remains running throughout all runs of the current experiment, including pauses while Sonoro is reset between runs.

Do not restart the sound controller between individual runs of the same experiment. Stop and restart it when moving to a different experiment condition that requires different sound levels.

### 2.2 Power on Sonoro and prepare its environment

1. Connect the power to Sonoro.
2. Wait for the Thymio robot to switch on.
3. Open a terminal on the Sonoro-control PC, or connect through SSH/VS Code Remote SSH.
4. Activate the Conda environment:

   ```bash
   conda activate robat
   ```

5. Change to the experiment directory:

   ```bash
   cd /home/thymio/robat_py/robat_swarm/passive_robat/Scripts/ring_with_ring
   ```

The entry point is `ring_circle_target.py`. Its output CSV filename uses the date in `DD_MM` format and the run ID passed on the command line. For example, run ID `A1_01` on 13 September produces:

```text
ringStats/13_09_A1_01.csv
```

The script creates `ringStats/` if it does not exist. It rejects the run if the output CSV already exists, to avoid appending to an earlier run. Use a new run ID or deliberately archive the old file before starting.

### 2.3 Start each individual run

Use the same full run ID in both the Vicon and Sonoro commands.

Example for the first run of experiment A1:

**A. Start Vicon data collection**

On the Vicon PC, from the directory containing `viconScripts/`, run:

```bash
python viconScripts/vicon.py A1_01
```

**B. Start the Sonoro experiment**

On the Sonoro-control PC, in the activated `robat` environment and the experiment directory, run:

```bash
python /home/thymio/robat_py/robat_swarm/passive_robat/Scripts/ring_with_ring/ring_circle_target.py A1_01
```

Here `A1_01` is the run ID (`experimentName_runID`). Use the matching ID for both scripts. For subsequent runs, change the ID in both commands, for example `A1_02`, `A1_03`, and so on.

### 2.4 End a run and record the outcome

When Sonoro reaches the proximity of one of the speakers:

1. Stop the Sonoro experiment script first by pressing `Ctrl+C` in its terminal.
2. Then stop the Vicon data-collection script by pressing `Ctrl+C` in its terminal.
3. Physically reset Sonoro to its planned starting position before the next run.
4. Record the outcome in the experiment sheet:
   - `L` if the outcome is the left speaker
   - `R` if the outcome is the right speaker
5. Check the next run ID and repeat Section 2.3.

Keep the sound controller running while completing runs for the same experiment condition. It is configured for 100 minutes, so check that it is still running before continuing.

### 2.5 Move to the next experiment condition

When all planned runs for the current condition are complete:

1. Stop the sound controller with `Ctrl+C` on the speaker-control laptop.
2. Update `base_left` and `base_right` in `soundScript/sound_generator_switch.py` for the next condition.
3. Start the sound controller again.
4. Continue with the next experiment name (for example, from `A1` to `A2`) and use new run IDs.

Do not change sound levels in the middle of a condition unless that change is part of the planned protocol and is recorded in the experiment sheet.

---

## End-of-session shutdown

At the end of the complete experiment session, stop the systems in this order:

1. **Sonoro:** stop its experiment script first (`Ctrl+C`), then power off/disconnect Sonoro as appropriate.
2. **Vicon:** stop the Vicon data-collection script (`Ctrl+C`), then stop the Vicon system/server as appropriate.
3. **Sound:** stop the sound controller on the speaker laptop (`Ctrl+C`), then switch off/disconnect the speakers as appropriate.

The required shutdown order is **Sonoro → Vicon → sound**.

---

## Quick checklist

### Before the experiment

- [ ] Sonoro and both speakers are available.
- [ ] Speaker-control laptop is close enough for the AUX cables.
- [ ] Required adapters and cables are available.
- [ ] Vicon PC and server are running.
- [ ] Sonoro, left speaker, and right speaker are visible in Vicon.
- [ ] `python viconScripts/test.py` confirms the three subjects.
- [ ] Arena arrangement and speaker labels are checked.
- [ ] Run count and left/right sound levels are decided.
- [ ] Experiment sheet has date, comments, experiment names, and sound levels.

### For each experiment condition

- [ ] Set `base_left` and `base_right`.
- [ ] Start the sound controller.
- [ ] Power on Sonoro and activate `robat`.
- [ ] For each run, start Vicon first and Sonoro second, using the same run ID.
- [ ] When Sonoro reaches a speaker, stop Sonoro first and Vicon second.
- [ ] Physically reset Sonoro.
- [ ] Record `L` or `R` in the experiment sheet.
- [ ] Use a fresh run ID for the next run.
- [ ] After the condition, stop and restart the sound controller with the next condition's levels.

### At the end

- [ ] Stop Sonoro first.
- [ ] Stop Vicon second.
- [ ] Stop sound playback last.
- [ ] Confirm the run data and experiment sheet have been saved.

---

## Notes and scope

- The supplied `sound_generator_switch.py` sets `target_duration = 6000`, which is 100 minutes, and `gap_duration = 1.0`. Its comments mention different timing/duration values; this protocol follows the executable parameter values.
- The supplied `test.py` connects to `localhost:801`. It expects the Vicon DataStream server to be reachable locally at that address. If it is run from a different computer, the connection configuration may need to be changed; no remote-server address was provided.
- The exact Vicon server startup procedure, the speaker laptop's audio-device selection, and the detailed physical power-off procedure were not specified, so this guide does not invent those steps.

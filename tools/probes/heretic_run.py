#!/usr/bin/env python3
"""Drive Heretic's interactive TUI non-interactively via a pexpect PTY.

Heretic has no headless/save-path CLI option: after optimization it opens a prompt_toolkit
menu (select trial -> "What do you want to do" -> "Path to the folder"), which EOFErrors under
a null stdin (background job). pexpect gives it a real PTY so the TUI works, and we send the
keystrokes: Enter picks the top (best) trial, Enter picks "Save the model to a local folder"
(the first action), then we type the output path. Once "Model saved to" prints, the fp16 model
is on disk and we terminate. --export-strategy MERGE is passed so the export-strategy sub-prompt
is skipped (main.py obtain_export_strategy returns early when it's set).

    python heretic_run.py <out_dir> <n_trials>
"""
import sys
import pexpect

OUT = sys.argv[1]
NTRIALS = sys.argv[2] if len(sys.argv) > 2 else "60"

cmd = (f"/home/augus/sglang-venv/bin/heretic "
       f"--model /home/augus/models/fp16/tc "
       f"--quantization BNB_4BIT "
       f"--n-trials {NTRIALS} --n-startup-trials 8 "
       f"--export-strategy MERGE "
       f"--study-checkpoint-dir /home/augus/heretic-tc-ckpt "
       f"--no-plot-residuals")

child = pexpect.spawn(cmd, timeout=None, encoding="utf-8", dimensions=(50, 220))
child.logfile_read = sys.stdout

# Long optimization phase streams here; block until the trial-selection menu.
child.expect("Which trial do you want to use")
child.send("\r")                       # top = best (lowest refusals) default

child.expect("What do you want to do with the decensored model")
child.send("\r")                       # first action = "Save the model to a local folder"

child.expect("Path to the folder")
child.sendline(OUT)

child.expect("Model saved to", timeout=3600)
print("\n=== HERETIC_SAVE_OK ===", flush=True)
child.sendcontrol("c")                 # model is on disk; leave the menu
try:
    child.expect(pexpect.EOF, timeout=60)
except Exception:
    child.terminate(force=True)

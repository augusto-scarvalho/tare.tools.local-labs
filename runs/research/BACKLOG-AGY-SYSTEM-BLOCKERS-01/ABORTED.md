# Execution aborted fail-closed

The WSL command transport interpreted extended-regex metacharacters as shell syntax during the RETRO-01 search. Execution stopped before any blocker receipt or claim. A successor must change only command transport to direct argv execution (`wsl -e`).

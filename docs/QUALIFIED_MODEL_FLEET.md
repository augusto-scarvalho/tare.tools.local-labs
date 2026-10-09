# Qualified local model fleet

Moved to [tare.tools.node](https://github.com/augusto-scarvalho/tare.tools.node) on 2026-10-09. The node ships the
**catalog** of the models tare.tools qualified (`tare_node/config/qualified_model_fleet.json`); a model qualified here
enters the catalog by a pull request there. Which models a node serves is its owner's **fleet**
(`~/.config/tare-node/fleet.json`, chosen with `tare fleet` or `/fleet` in the TUI): catalog models first, and the
owner's own. See that repository's `docs/QUALIFIED_MODEL_FLEET.md`.

[`config/qualified_model_fleet.json`](../config/qualified_model_fleet.json) stays here as the registry the research
scripts and their evidence cite, as of the move. `tools/agents/modelctl.py` reads the node's catalog and needs
tare.tools.node installed:

```powershell
python -m pip install "git+ssh://git@github.com/augusto-scarvalho/tare.tools.node"
python tools/agents/modelctl.py status
```

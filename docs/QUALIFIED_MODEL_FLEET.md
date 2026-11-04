# Qualified local model fleet

Moved to the node service on 2026-10-09, now `node/` in
[tare.tools.os](https://github.com/augusto-scarvalho/tare.tools.os/tree/main/node). The node ships the **catalog** of
the models tare.tools qualified (`node/tare_node/config/qualified_model_fleet.json`); a model qualified here enters
the catalog by a pull request there. Which models a node serves is its owner's **fleet**
(`~/.config/tare-node/fleet.json`, chosen with `tare fleet` or `/fleet` in the TUI): catalog models first, and the
owner's own. See `node/docs/QUALIFIED_MODEL_FLEET.md` there.

[`config/qualified_model_fleet.json`](../config/qualified_model_fleet.json) stays here as the registry the research
scripts and their evidence cite, as of the move. `tools/agents/modelctl.py` reads the node's catalog and needs
the node package installed:

```powershell
git clone https://github.com/augusto-scarvalho/tare.tools.os.git ../tare.tools.os   # the satellites are not needed
python -m pip install ../tare.tools.os/node
python tools/agents/modelctl.py status
```

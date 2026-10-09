# Qualified local model fleet

Moved to [tare.tools.node](https://github.com/augusto-scarvalho/tare.tools.node) on 2026-10-09: the gateway serves the
fleet, and the live registry ships with it (`tare_node/config/qualified_model_fleet.json`, documented in that
repository's `docs/QUALIFIED_MODEL_FLEET.md`). A model qualified here enters the fleet by a pull request there.

[`config/qualified_model_fleet.json`](../config/qualified_model_fleet.json) stays here as the registry the research
scripts and their evidence cite, as of the move. `tools/agents/modelctl.py` reads the node's registry and needs
tare.tools.node installed:

```powershell
python -m pip install "git+ssh://git@github.com/augusto-scarvalho/tare.tools.node"
python tools/agents/modelctl.py status
```

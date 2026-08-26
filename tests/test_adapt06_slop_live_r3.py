from tools.research.run_adapt06_slop_live_r3 import route
def test_r3_route_controls_unchanged():
 assert route("base")!=route("mlp")!=route("attn")

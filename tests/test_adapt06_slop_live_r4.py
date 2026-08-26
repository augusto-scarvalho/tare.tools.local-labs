from tools.research.run_adapt06_slop_live_r4 import route
def test_r4_route_controls_unchanged():
 assert route("base")[0]["scale"]==0.0
 assert route("mlp")[0]["scale"]==1.0 and route("attn")[1]["scale"]==1.0

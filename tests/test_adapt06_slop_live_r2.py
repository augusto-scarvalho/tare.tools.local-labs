from tools.research.run_adapt06_slop_live_r2 import route
def test_r2_routes_keep_two_adapter_scales_disjoint():
 assert route("base")[0]["scale"]==0 and route("base")[1]["scale"]==0
 assert route("mlp")[0]["scale"]==1 and route("attn")[1]["scale"]==1

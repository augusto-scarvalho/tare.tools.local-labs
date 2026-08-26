from tools.research.run_adapt06_slop_live import route
def test_routes_are_disjoint():
 assert route("base")==[{"id":0,"scale":0.0},{"id":1,"scale":0.0}]
 assert route("mlp")!=route("attn")

from tools.analysis.adaptive_mtp_controller import AdaptiveMTPController
def test_controller_mapping_domain():
 c=AdaptiveMTPController()
 assert 0<=c.get_recommended_depth()<=4

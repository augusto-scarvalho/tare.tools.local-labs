from tools.research import run_gdn02_learned_state_r3 as r3


def test_wsl_model_path_is_literal_posix_string():
    assert isinstance(r3.MODEL, str)
    assert r3.MODEL.startswith("/home/")
    assert "\\" not in r3.MODEL

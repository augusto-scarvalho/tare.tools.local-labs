from tools.research.run_adapt06_slop_live_r5 import stable_exec
def test_stable_exec_drops_only_volatile_systemd_suffix():
 a="{ path=/x ; argv[]=/x -m /m ; ignore_errors=no ; start_time=[a] ; pid=1 }"
 b="{ path=/x ; argv[]=/x -m /m ; ignore_errors=no ; start_time=[b] ; pid=2 }"
 assert stable_exec(a)==stable_exec(b)=="{ path=/x ; argv[]=/x -m /m"

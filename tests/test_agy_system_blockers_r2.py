from tools.research.run_agy_system_blockers_r2 import ITEMS,TASK_ID
def test_r2_scope_unchanged():
 assert TASK_ID.endswith("-02") and len(ITEMS)==6

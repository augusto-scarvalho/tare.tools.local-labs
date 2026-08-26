from tools.analysis.reasoning_loop_guard import ReasoningLoopGuard


def test_guard_triggers_repeated_reversal_pieces():
    guard = ReasoningLoopGuard()
    fired = False
    for piece in (["wait ", "let ", "me ", "reconsider ", "now "] * 8):
        fired, _ = guard.feed_token(piece)
        if fired:
            break
    assert fired


def test_guard_does_not_trigger_short_legitimate_trace():
    guard = ReasoningLoopGuard()
    assert not any(guard.feed_token(piece)[0] for piece in "First calculate the total then verify the result and provide the answer".split())

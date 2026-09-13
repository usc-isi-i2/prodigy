from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_tucker.sh"


def test_launcher_defaults_to_strict_gpu_2_3_scope() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'GPU_A="${GPU_A:-2}"' in text
    assert 'GPU_B="${GPU_B:-3}"' in text
    assert '[[ "$gpu" =~ ^[23]$ ]]' in text
    assert '[[ "$gpu" =~ ^[0-3]$ ]]' not in text


def test_overlap_gate_is_explicit_and_strict_by_default() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'GPU_MAX_USED_MIB="${GPU_MAX_USED_MIB:-2000}"' in text
    assert 'GPU_MAX_UTIL_PCT="${GPU_MAX_UTIL_PCT:-10}"' in text
    assert "used < GPU_MAX_USED_MIB && util < GPU_MAX_UTIL_PCT" in text
    assert "gpu_max_used_mib=$GPU_MAX_USED_MIB" in text
    assert "gpu_max_util_pct=$GPU_MAX_UTIL_PCT" in text


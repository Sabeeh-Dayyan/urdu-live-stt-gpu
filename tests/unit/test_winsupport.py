import sys

from live_stt import winsupport


def test_cuda_dll_dirs_finds_pip_nvidia_bins_then_toolkit(tmp_path):
    sp = tmp_path / "site-packages"
    for lib in ("cudnn", "cublas"):
        (sp / "nvidia" / lib / "bin").mkdir(parents=True)
    (sp / "nvidia" / "nvjitlink").mkdir(parents=True)          # no bin/: ignored
    toolkit = tmp_path / "CUDA" / "v12.4"
    (toolkit / "bin").mkdir(parents=True)
    got = winsupport.cuda_dll_dirs([str(sp), str(tmp_path / "missing")], {"CUDA_PATH": str(toolkit)})
    assert got == [sp / "nvidia" / "cublas" / "bin", sp / "nvidia" / "cudnn" / "bin", toolkit / "bin"]


def test_cuda_dll_dirs_skips_missing_toolkit_and_duplicates(tmp_path):
    sp = tmp_path / "sp"
    (sp / "nvidia" / "cudnn" / "bin").mkdir(parents=True)
    got = winsupport.cuda_dll_dirs([str(sp), str(sp)], {"CUDA_PATH": str(tmp_path / "nope")})
    assert got == [sp / "nvidia" / "cudnn" / "bin"]


def test_add_cuda_dll_dirs_is_a_noop_off_windows():
    if sys.platform != "win32":
        assert winsupport.add_cuda_dll_dirs() == []

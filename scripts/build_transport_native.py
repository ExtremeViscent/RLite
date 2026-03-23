"""Build the Linux native transport shared library with CMake."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def _build_with_cxx(
    repo_root: Path,
    source_dir: Path,
    output_dir: Path,
    build_type: str,
    cuda_include_dir: str,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("neither cmake nor c++ is available to build the native transport library")

    optimization_flags = ["-O0", "-g"] if build_type == "Debug" else ["-O3"]
    command = [
        compiler,
        "-std=c++17",
        "-shared",
        "-fPIC",
        *optimization_flags,
        "-I",
        str((source_dir / "include").resolve()),
        "-I",
        str((repo_root / "third-party" / "libfabric" / "include").resolve()),
        "-I",
        str((repo_root / "third-party" / "gdrcopy" / "include").resolve()),
    ]
    if cuda_include_dir:
        command.extend(["-I", cuda_include_dir])
    command.extend(
        [
            str((source_dir / "src" / "rlite_transport_native.cpp").resolve()),
            "-ldl",
            "-lpthread",
            "-o",
            str((output_dir / "librlite_transport_native.so").resolve()),
        ]
    )
    _run(command, repo_root)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=repo_root / "build" / "transport-native",
        help="Out-of-tree CMake build directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "src" / "rlite" / "transport" / "_native",
        help="Directory that should receive the built shared library",
    )
    parser.add_argument(
        "--build-type",
        default="Release",
        choices=("Debug", "Release", "RelWithDebInfo", "MinSizeRel"),
        help="CMake build type",
    )
    parser.add_argument(
        "--generator",
        default="",
        help="Optional CMake generator override",
    )
    parser.add_argument(
        "--cuda-include-dir",
        default=os.environ.get("CUDA_HOME", ""),
        help="Optional CUDA toolkit root or include directory",
    )
    args = parser.parse_args()

    source_dir = repo_root / "native" / "transport"
    output_dir = args.output_dir.resolve()
    build_dir = args.build_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    cuda_include_dir = ""
    if args.cuda_include_dir:
        candidate = Path(args.cuda_include_dir)
        include_dir = candidate / "include" if (candidate / "include").exists() else candidate
        if include_dir.exists():
            cuda_include_dir = str(include_dir.resolve())

    cmake_executable = shutil.which("cmake")
    if cmake_executable is not None:
        configure_command = [
            cmake_executable,
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={args.build_type}",
            f"-DRLITE_TRANSPORT_NATIVE_OUTPUT_DIR={output_dir}",
        ]
        if args.generator:
            configure_command.extend(["-G", args.generator])
        if cuda_include_dir:
            configure_command.append(f"-DRLITE_TRANSPORT_CUDA_INCLUDE_DIR={cuda_include_dir}")

        build_command = [
            cmake_executable,
            "--build",
            str(build_dir),
            "--config",
            args.build_type,
            "--parallel",
        ]

        _run(configure_command, repo_root)
        _run(build_command, repo_root)
        return

    _build_with_cxx(repo_root, source_dir, output_dir, args.build_type, cuda_include_dir)


if __name__ == "__main__":
    main()

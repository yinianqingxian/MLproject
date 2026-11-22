"""
用途：为 Spark 容器补齐缺失应用依赖并保护 NVIDIA 科学计算环境。
核心职责：
- 记录并比对 PyTorch、NumPy、SciPy 的版本、模块路径与 NVIDIA 约束文件。
- 复用健康的继承依赖，仅在独立虚拟环境安装缺失模块并核验元数据。
关键副作用/安全约束：
- 安装模式只允许在继承系统包的虚拟环境运行，不修改基础环境的 pip 配置。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

from packaging.requirements import Requirement


PROTECTED = ("torch", "torchvision", "numpy", "scipy")
APPLICATIONS = {"opencv-python-headless": ("cv2", (
    "opencv-python-headless", "opencv-python", "opencv-contrib-python-headless", "opencv-contrib-python")),
    "scikit-learn": ("sklearn", ("scikit-learn",)),
    "matplotlib": ("matplotlib", ("matplotlib",))}
NVIDIA_CONSTRAINTS = Path("/etc/pip/constraint.txt")


def snapshot() -> dict:
    packages = {}
    for name in PROTECTED:
        module = importlib.import_module(name)
        packages[name] = {"version": str(module.__version__),
                          "path": str(Path(module.__file__).resolve())}
    constraints = None
    if NVIDIA_CONSTRAINTS.is_file():
        constraints = {"path": str(NVIDIA_CONSTRAINTS),
                       "sha256": hashlib.sha256(NVIDIA_CONSTRAINTS.read_bytes()).hexdigest()}
    return {"packages": packages, "nvidia_constraints": constraints}


def require_unchanged(baseline: dict, current: dict) -> None:
    for name, expected in baseline["packages"].items():
        if current["packages"].get(name) != expected:
            raise RuntimeError(f"Protected NVIDIA package changed: {name}: "
                               f"{expected!r} -> {current['packages'].get(name)!r}")
    if current.get("nvidia_constraints") != baseline.get("nvidia_constraints"):
        raise RuntimeError("NVIDIA pip constraint file changed during dependency preparation")


def validate_dependency_metadata(distribution: metadata.Distribution) -> None:
    for raw in distribution.requires or []:
        requirement = Requirement(raw)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f"{distribution.metadata['Name']} has missing dependency: {requirement}") from error
        if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
            raise RuntimeError(f"{distribution.metadata['Name']} requires {requirement}; inherited "
                               f"{requirement.name} is {installed}. Do not replace NVIDIA NumPy/SciPy/torch.")


def fresh_snapshot() -> dict:
    output = subprocess.check_output([sys.executable, str(Path(__file__).resolve()), "snapshot"], text=True)
    return json.loads(output)


def inspect_applications() -> tuple[dict, list[str]]:
    available, missing = {}, []
    for target, (module_name, distributions) in APPLICATIONS.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(target)
            continue
        # An existing but broken binary import must stop; reinstalling could hide
        # a CUDA/NumPy ABI mismatch and shadow the supported NVIDIA stack.
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            raise RuntimeError(f"Inherited {module_name} exists but cannot import: {error}") from error
        distribution = None
        for candidate in distributions:
            try:
                distribution = metadata.distribution(candidate)
                break
            except metadata.PackageNotFoundError:
                pass
        if distribution is None:
            raise RuntimeError(f"Inherited {module_name} has no package metadata; cannot verify its dependencies")
        validate_dependency_metadata(distribution)
        available[target] = {"distribution": distribution.metadata["Name"],
                             "version": distribution.version,
                             "module_version": str(getattr(module, "__version__", distribution.version)),
                             "path": str(Path(module.__file__).resolve())}
    return available, missing


def pip_install_command(missing: list[str], requirements: dict[str, str], constraints: Path) -> list[str]:
    if not NVIDIA_CONSTRAINTS.is_file():
        raise RuntimeError(f"NVIDIA pip constraint file is missing: {NVIDIA_CONSTRAINTS}")
    return [sys.executable, "-m", "pip", "install", "--no-user", "--only-binary=:all:",
            "--upgrade-strategy", "only-if-needed", "--constraint", str(NVIDIA_CONSTRAINTS),
            "--constraint", str(constraints), *[requirements[name] for name in missing]]


def install_missing(baseline_path: Path, requirements_path: Path, report_path: Path) -> None:
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Dependency installation requires the isolated --system-site-packages venv")
    venv_config = Path(sys.prefix) / "pyvenv.cfg"
    if not venv_config.is_file() or "include-system-site-packages = true" not in venv_config.read_text().lower():
        raise RuntimeError("The application venv must inherit NVIDIA's system site packages")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("nvidia_constraints") is None or not NVIDIA_CONSTRAINTS.is_file():
        raise RuntimeError(f"NVIDIA pip constraint file is required: {NVIDIA_CONSTRAINTS}")
    require_unchanged(baseline, fresh_snapshot())
    available, missing = inspect_applications()
    requirements = {}
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            requirement = Requirement(line.strip())
            requirements[requirement.name] = str(requirement)
    if set(requirements) != set(APPLICATIONS):
        raise RuntimeError("Fallback requirements must contain exactly the three supported application packages")
    if missing:
        constraints = baseline_path.with_name("traffic-sign-protected-constraints.txt")
        constraints.write_text("".join(f"{name}=={values['version']}\n"
                                       for name, values in baseline["packages"].items()), encoding="utf-8")
        subprocess.run(pip_install_command(missing, requirements, constraints), check=True)
    # A fresh process catches packages shadowed by pip, despite imports cached in
    # this process. Its application imports likewise see the final installed tree.
    require_unchanged(baseline, fresh_snapshot())
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "verify",
                    "--baseline", str(baseline_path), "--report", str(report_path)], check=True)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("snapshot")
    capture.add_argument("--output", type=Path)
    install = subparsers.add_parser("install")
    install.add_argument("--baseline", required=True, type=Path)
    install.add_argument("--requirements", required=True, type=Path)
    install.add_argument("--report", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--baseline", required=True, type=Path)
    verify.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        state = snapshot()
        if args.output:
            write_json(args.output, state)
        else:
            print(json.dumps(state))
    elif args.command == "install":
        install_missing(args.baseline, args.requirements, args.report)
    else:
        state = snapshot()
        require_unchanged(json.loads(args.baseline.read_text(encoding="utf-8")), state)
        applications, missing = inspect_applications()
        if missing:
            raise RuntimeError(f"Application dependencies remain missing: {', '.join(missing)}")
        write_json(args.report, {"python": platform.python_version(), "machine": platform.machine(),
                                 "protected": state, "applications": applications,
                                 "python_executable": sys.executable})
        print(f"NVIDIA stack preserved; application imports and dependencies verified: {args.report}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ImportError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Spark dependency preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)

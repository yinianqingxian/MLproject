"""
用途：核验 Spark 启动器的参数、安全边界及依赖保护。
核心职责：
- 在无 Docker 的主机验证帮助、命令预览和带空格参数传递。
- 检查容器依赖验证会拒绝科学计算包替换及缺失依赖。
关键副作用/安全约束：
- 仅运行命令预览并在临时目录创建测试元数据，不启动容器或训练。
"""

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")
if os.name == "nt":
    candidate = Path("C:/Program Files/Git/bin/bash.exe")
    # Windows' system bash.exe can be a WSL launcher with no installed distro;
    # it cannot exercise these Windows paths as Git Bash does.
    BASH = str(candidate) if candidate.is_file() else None


@unittest.skipUnless(BASH, "Bash is needed to exercise the shell launcher")
class SparkLauncherTests(unittest.TestCase):
    def launch(self, *args, script=None, extra_environment=None):
        script = script or PROJECT / "SPARK.sh"
        self.assertTrue(script.is_file(), "Spark launcher has not been implemented")
        environment = dict(os.environ, SPARK_IMAGE="ds4023-c-spark:25.11")
        environment.update(extra_environment or {})
        return subprocess.run([BASH, str(script), *args], capture_output=True,
                              text=True, env=environment, check=False)

    def docker_commands(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return [shlex.split(line) for line in result.stdout.splitlines()
                if line.startswith("docker ")]

    def test_help_needs_no_linux_gpu_or_docker(self):
        result = self.launch("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("setup", result.stdout)
        self.assertIn("--dry-run", result.stdout)

    def test_train_preview_preserves_gpu_runtime_and_custom_arguments(self):
        commands = self.docker_commands(self.launch(
            "--dry-run", "train", "--output-dir", "results/custom study", "--resume"))
        run = next(command for command in commands if command[1] == "run")
        for flag, value in (("--platform", "linux/arm64"), ("--gpus", "all"),
                            ("--shm-size", "1g"), ("--workdir", "/workspace")):
            self.assertEqual(run[run.index(flag) + 1], value)
        self.assertIn("--rm", run)
        self.assertIn("--init", run)
        self.assertIn("NVIDIA_IMEX_CHANNELS=0", run)
        self.assertIn("NVIDIA_TF32_OVERRIDE=0", run)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", run)
        self.assertIn("tools/run_spark_study.py", run)
        self.assertEqual(run[run.index("--device") + 1], "cuda")
        self.assertEqual(run[run.index("--epochs") + 1], "10")
        self.assertEqual(run[run.index("--threads") + 1], "6")
        self.assertEqual(run[-3:], ["--output-dir", "results/custom study", "--resume"])
        self.assertNotIn("--inputs", run)
        self.assertNotIn("--augmentations", run)

    def test_project_path_with_spaces_is_one_bind_argument(self):
        with tempfile.TemporaryDirectory(prefix="spark project ") as temporary:
            script = Path(temporary) / "SPARK.sh"
            self.assertTrue((PROJECT / "SPARK.sh").is_file(),
                            "Spark launcher has not been implemented")
            shutil.copyfile(PROJECT / "SPARK.sh", script)
            commands = self.docker_commands(self.launch("smoke", "--dry-run", script=script))
            run = next(command for command in commands if command[1] == "run")
            mount = run[run.index("--mount") + 1]
            self.assertIn("spark project ", mount)
            self.assertTrue(mount.endswith(",target=/workspace"), mount)
            self.assertEqual(run[run.index("--epochs") + 1], "1")
            self.assertEqual(run[run.index("--inputs") + 1], "dual")
            self.assertEqual(run[run.index("--augmentations") + 1], "mixed")
            self.assertIn("tools/run_spark_study.py", run)

    def test_setup_preview_uses_only_spark_build_context_and_then_checks_readiness(self):
        commands = self.docker_commands(self.launch("setup", "--dry-run"))
        build = next(command for command in commands if command[1] == "build")
        self.assertEqual(build[build.index("--platform") + 1], "linux/arm64")
        self.assertTrue(build[-1].replace("\\", "/").endswith("/spark"), build)
        check = next(command for command in commands if command[1] == "run")
        self.assertIn("tools/check_spark_environment.py", check)
        report = check[check.index("--report") + 1]
        self.assertTrue(report.startswith("results/spark_environments/"), report)

    def test_verify_and_report_use_separate_spark_outputs(self):
        verification = self.docker_commands(self.launch("--dry-run", "verify"))[-1]
        self.assertIn("--verify-only", verification)
        self.assertIn("experiments/run_resnet_study.py", verification)
        self.assertEqual(verification[verification.index("--output-dir") + 1],
                         "results/study_resnet18_spark_v1")
        report = self.docker_commands(self.launch("--dry-run", "report"))[-1]
        self.assertIn("experiments/summarize_resnet_study.py", report)
        self.assertEqual(report[report.index("--report-dir") + 1], "report/c_part_resnet18_spark")

    def test_actual_run_uses_inspected_immutable_image_and_rejects_wrong_architecture(self):
        with tempfile.TemporaryDirectory(prefix="spark fake docker ") as temporary:
            root = Path(temporary)
            fixtures = {
                "uname": '#!/usr/bin/env bash\nif [[ "$1" == -s ]]; then echo Linux; else echo aarch64; fi\n',
                "id": '#!/usr/bin/env bash\nif [[ "$1" == -u ]]; then echo 1201; else echo 1202; fi\n',
                "docker": '''#!/usr/bin/env bash
case "$1" in
  info) exit 0 ;;
  image) printf '%s|sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\n' "$SPARK_TEST_ARCH" ;;
  run) printf '%s\\0' "$@" > "$SPARK_TEST_LOG" ;;
  *) exit 99 ;;
esac
''',
            }
            for name, contents in fixtures.items():
                path = root / name
                path.write_text(contents, encoding="utf-8", newline="\n")
                path.chmod(0o755)
            # Git Bash prepends its utilities to inherited PATH on Windows;
            # BASH_ENV places the executable fixtures first after that startup.
            shell_root = root.as_posix()
            if os.name == "nt":
                shell_root = "/" + root.drive[0].lower() + shell_root[2:]
            shell_environment = root / "fixtures.sh"
            shell_environment.write_text("export PATH=" + shlex.quote(shell_root) + ':"$PATH"\n',
                                         encoding="utf-8", newline="\n")
            environment = {"BASH_ENV": str(shell_environment), "SPARK_TEST_ARCH": "arm64",
                           "SPARK_TEST_LOG": shell_root + "/run.args"}
            result = self.launch("check", extra_environment=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = (root / "run.args").read_bytes().decode().rstrip("\0").split("\0")
            expected_id = "sha256:" + "a" * 64
            self.assertIn("SPARK_CONTAINER_IMAGE_ID=" + expected_id, arguments)
            self.assertIn("SPARK_CONTAINER_IMAGE_REF=ds4023-c-spark:25.11", arguments)
            self.assertIn(expected_id, arguments)
            self.assertEqual(arguments[arguments.index("--user") + 1], "1201:1202")
            (root / "run.args").unlink()
            result = self.launch("train", extra_environment=dict(environment, SPARK_TEST_ARCH="amd64"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("architecture", result.stderr)
            self.assertFalse((root / "run.args").exists(), "Wrong-architecture image must never run")


class SparkDependencyProtectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = PROJECT / "spark/validate_overlay.py"
        if not path.is_file():
            raise AssertionError("Spark dependency validator has not been implemented")
        spec = importlib.util.spec_from_file_location("spark_dependencies", path)
        cls.validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.validator)

    def test_protected_module_relocation_and_version_changes_are_rejected(self):
        baseline = {"packages": {"numpy": {"version": "2.1.0", "path": "/base/numpy.py"}},
                    "nvidia_constraints": {"path": "/etc/pip/constraint.txt", "sha256": "original"}}
        self.validator.require_unchanged(baseline, baseline)
        for changed in ({"version": "2.2.0", "path": "/base/numpy.py"},
                        {"version": "2.1.0", "path": "/venv/numpy.py"}):
            with self.subTest(changed=changed), self.assertRaisesRegex(RuntimeError, "numpy"):
                self.validator.require_unchanged(baseline, dict(baseline, packages={"numpy": changed}))
        with self.assertRaisesRegex(RuntimeError, "constraint"):
            self.validator.require_unchanged(baseline, dict(baseline, nvidia_constraints=None))

    def distribution(self, root, requirements):
        path = root / "spark_test_app-1.0.dist-info"
        path.mkdir(exist_ok=True)
        metadata = "Metadata-Version: 2.1\nName: spark-test-app\nVersion: 1.0\n"
        metadata += "".join(f"Requires-Dist: {item}\n" for item in requirements)
        (path / "METADATA").write_text(metadata, encoding="utf-8")
        return importlib.metadata.Distribution.at(path)

    def test_dependency_metadata_honors_markers_and_rejects_missing_or_wrong_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for requirements, expected_error in (
                (["packaging>=0", 'spark-impossible-test-dependency; python_version < "2"'], None),
                (["spark-impossible-test-dependency>=1"], "missing"),
                (["packaging>=999"], "requires"),
            ):
                distribution = self.distribution(root, requirements)
                with self.subTest(requirements=requirements):
                    if expected_error:
                        with self.assertRaisesRegex(RuntimeError, expected_error):
                            self.validator.validate_dependency_metadata(distribution)
                    else:
                        self.validator.validate_dependency_metadata(distribution)

    def test_install_requires_and_passes_both_nvidia_and_protected_constraints(self):
        with tempfile.TemporaryDirectory() as temporary:
            nvidia_constraints = Path(temporary) / "nvidia-constraint.txt"
            protected_constraints = Path(temporary) / "protected-constraint.txt"
            with patch.object(self.validator, "NVIDIA_CONSTRAINTS", nvidia_constraints):
                with self.assertRaisesRegex(RuntimeError, "NVIDIA.*constraint"):
                    self.validator.pip_install_command(["matplotlib"],
                        {"matplotlib": "matplotlib>=3.8,<3.11"}, protected_constraints)
                nvidia_constraints.write_text("numpy==2.1.0\n", encoding="utf-8")
                command = self.validator.pip_install_command(["matplotlib"],
                    {"matplotlib": "matplotlib>=3.8,<3.11"}, protected_constraints)
            self.assertEqual([command[index + 1] for index, argument in enumerate(command)
                              if argument == "--constraint"],
                             [str(nvidia_constraints), str(protected_constraints)])
            self.assertEqual(command[-1], "matplotlib>=3.8,<3.11")
            self.assertIn("--no-user", command)
            self.assertIn("--only-binary=:all:", command)


if __name__ == "__main__":
    unittest.main()

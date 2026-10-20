import errno
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.local_storage import configure_runtime, ensure_free_space, runtime_environment
from pipelines.train_local_datasets import storage_failure, write_summary


class LocalStorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / "Extreme SSD/results/freeart"

    def test_runtime_and_child_temporary_files_stay_inside_target(self):
        with patch.dict(os.environ), patch.object(tempfile, "tempdir", tempfile.tempdir), \
             patch("utils.local_storage.MIN_FREE_BYTES", 0):
            env = configure_runtime(self.target)
            with tempfile.TemporaryDirectory() as directory:
                self.assertTrue(Path(directory).is_relative_to(self.target))
            result = subprocess.run([sys.executable, "-B", "-c",
                "import json, os, tempfile; print(json.dumps([tempfile.gettempdir(), "
                "os.environ['TORCH_EXTENSIONS_DIR'], os.environ['TORCH_HOME']]))"],
                capture_output=True, text=True, check=True)
            for directory in json.loads(result.stdout):
                self.assertTrue(Path(directory).is_relative_to(self.target))
            self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")

    def test_dry_run_bootstrap_and_shell_quoting_write_nothing(self):
        helper = Path(__file__).resolve().parents[1] / "utils/local_storage.py"
        result = subprocess.run([sys.executable, "-B", str(helper), "--outputs-root", str(self.target),
                                 "--objects", "scene", "--dry-run"], capture_output=True, text=True, check=True)
        env = runtime_environment(self.target)
        # Check the same eval used by the Bash launcher, including a path containing spaces.
        shell = subprocess.run(["bash", "-c", result.stdout + '\nprintf "%s" "$TMPDIR"'],
                               capture_output=True, text=True, check=True)
        self.assertEqual(shell.stdout, env["TMPDIR"])
        self.assertFalse(self.target.exists())
        result = subprocess.run([sys.executable, "-B", str(helper), "--outputs-root", str(self.target),
                                 "--help"], capture_output=True, text=True, check=True)
        self.assertIn("export TMPDIR=", result.stdout)
        self.assertFalse(self.target.exists())

    def test_runtime_rejects_symlinks_that_escape_target(self):
        self.target.mkdir(parents=True)
        (self.target / ".runtime").symlink_to(Path(self.temp.name))
        with self.assertRaisesRegex(ValueError, "escapes"):
            configure_runtime(self.target)

    def test_free_space_guard_and_subprocess_disk_errors(self):
        usage = shutil._ntuple_diskusage(total=20 * 1024**3, used=19 * 1024**3, free=1024**3)
        with patch("utils.local_storage.shutil.disk_usage", return_value=usage):
            with self.assertRaises(OSError) as error:
                ensure_free_space(self.target)
            self.assertEqual(error.exception.errno, errno.ENOSPC)
        log = Path(self.temp.name) / "job.log"
        command_error = subprocess.CalledProcessError(1, ["bash"])
        for message in ("OSError: [Errno 28] No space left on device", "No usable temporary directory found"):
            log.write_text(message)
            self.assertTrue(storage_failure(command_error, log))
        log.write_text("torch.OutOfMemoryError: CUDA out of memory")
        self.assertFalse(storage_failure(command_error, log))
        self.assertTrue(storage_failure(OSError(errno.ENOSPC, "full"), log))

    def test_failed_summary_write_preserves_previous_results(self):
        path = Path(self.temp.name) / "summary.json"
        old = [{"job": "completed", "status": "passed"}]
        path.write_text(json.dumps(old))
        with patch.object(Path, "write_text", side_effect=OSError(errno.ENOSPC, "full")):
            with self.assertRaises(OSError):
                write_summary(path, [{"job": "new"}])
        self.assertEqual(json.loads(path.read_text()), old)
        write_summary(path, [{"job": "new"}])
        self.assertEqual(json.loads(path.read_text()), [{"job": "new"}])
        self.assertFalse(path.with_suffix(".tmp").exists())

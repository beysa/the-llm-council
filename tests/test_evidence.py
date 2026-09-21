"""Tests for local capability evidence collection."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from llm_council.engine import evidence as evidence_module
from llm_council.engine.capabilities import CapabilityPlan
from llm_council.engine.evidence import (
    _candidate_files,
    _load_git_diff,
    collect_capability_evidence,
)


class TestEvidenceCollection:
    """Tests for bounded local evidence collection."""

    @pytest.mark.asyncio
    async def test_collects_planning_and_repo_evidence(self, tmp_path):
        """Planning runs should gather checklist and repo-analysis evidence."""
        (tmp_path / "src").mkdir()
        (tmp_path / "docs" / "architecture").mkdir(parents=True)
        (tmp_path / "src" / "planner.py").write_text("def plan_migration():\n    return True\n")
        (tmp_path / "README.md").write_text("# Project\n")
        (tmp_path / "ROADMAP.md").write_text("# Roadmap\n")
        (tmp_path / "docs" / "index.md").write_text("# Docs\n")

        plan = CapabilityPlan(
            execution_profile="light_tools",
            budget_class="normal",
            required_capabilities=["planning-assess", "repo-analysis"],
        )

        bundle = await collect_capability_evidence(
            "Plan the migration for planner rollout",
            "planner",
            "plan",
            plan,
            repo_root=tmp_path,
        )

        assert bundle.executed_capabilities == ["planning-assess", "repo-analysis"]
        assert bundle.pending_capabilities == []
        assert len(bundle.items) == 2
        prompt_block = bundle.to_prompt_block()
        assert "Collected Evidence" in prompt_block
        assert "Planning artifacts" in prompt_block

    @pytest.mark.asyncio
    async def test_collects_security_matches_and_leaves_unsupported_pending(self, tmp_path):
        """Legacy security-audit alias should still work and leave unsupported packs pending."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "auth.py").write_text(
            "TOKEN='dev'\ndef run_auth(password, session):\n    return password, session, TOKEN\n"
        )

        plan = CapabilityPlan(
            execution_profile="deep_analysis",
            budget_class="premium",
            required_capabilities=["security-audit", "docs-research"],
        )

        bundle = await collect_capability_evidence(
            "Assess the authentication attack surface",
            "critic",
            "security",
            plan,
            repo_root=tmp_path,
        )

        assert bundle.executed_capabilities == ["security-audit"]
        assert bundle.pending_capabilities == ["docs-research"]
        assert any(item.capability == "security-audit" for item in bundle.items)
        assert "Capability packs not executed in this runtime" in bundle.to_prompt_block()

    @pytest.mark.asyncio
    async def test_collects_split_security_evidence(self, tmp_path):
        """Split security packs should gather attack-surface and code-audit evidence."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "auth.py").write_text(
            "TOKEN='dev'\n"
            "router.post('/login')\n"
            "def run_auth(password, session):\n"
            "    return password, session, TOKEN\n"
        )

        plan = CapabilityPlan(
            execution_profile="deep_analysis",
            budget_class="premium",
            required_capabilities=["red-team-recon", "security-code-audit"],
        )

        bundle = await collect_capability_evidence(
            "Assess the authentication attack surface",
            "critic",
            "security",
            plan,
            repo_root=tmp_path,
        )

        assert bundle.executed_capabilities == ["red-team-recon", "security-code-audit"]
        assert bundle.pending_capabilities == []
        assert any(item.capability == "red-team-recon" for item in bundle.items)
        assert any(item.capability == "security-code-audit" for item in bundle.items)

    @pytest.mark.asyncio
    async def test_collects_docs_research_for_known_targets(self, tmp_path, monkeypatch):
        """Docs research should execute when the task maps to supported official docs."""
        fetched_urls: list[str] = []

        def fake_fetch(url: str) -> tuple[str, str] | None:
            fetched_urls.append(url)
            return ("React Docs", "Server and client components guidance.")

        monkeypatch.setattr("llm_council.engine.evidence._fetch_docs_page", fake_fetch)

        plan = CapabilityPlan(
            execution_profile="grounded",
            budget_class="normal",
            required_capabilities=["docs-research"],
        )

        bundle = await collect_capability_evidence(
            "Research React server components best practices",
            "researcher",
            None,
            plan,
            repo_root=tmp_path,
        )

        assert bundle.executed_capabilities == ["docs-research"]
        assert bundle.pending_capabilities == []
        assert fetched_urls == ["https://react.dev/reference/react"]
        assert bundle.items[0].capability == "docs-research"
        assert "React Docs" in bundle.items[0].details[0]

    @pytest.mark.asyncio
    async def test_collects_diff_review_from_git_diff(self, tmp_path, monkeypatch):
        """Review runs should gather changed-file and hunk evidence from a diff."""
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "src" / "service.py").write_text("def old():\n    return False\n")
        (tmp_path / "tests" / "test_service.py").write_text("def test_old():\n    assert True\n")

        monkeypatch.setattr(
            "llm_council.engine.evidence._load_git_diff",
            lambda repo_root: (
                "diff --git a/src/service.py b/src/service.py\n"
                "--- a/src/service.py\n"
                "+++ b/src/service.py\n"
                "@@ -1,2 +1,2 @@\n"
                "-def old():\n"
                "+def old(flag=True):\n"
            ),
        )

        plan = CapabilityPlan(
            execution_profile="light_tools",
            budget_class="normal",
            required_capabilities=["diff-review"],
        )

        bundle = await collect_capability_evidence(
            "Review the latest implementation changes",
            "critic",
            "review",
            plan,
            repo_root=tmp_path,
        )

        assert bundle.executed_capabilities == ["diff-review"]
        assert bundle.pending_capabilities == []
        assert bundle.items[0].capability == "diff-review"
        assert "Changed files: src/service.py" in bundle.items[0].details[0]
        assert any("@@ -1,2 +1,2 @@" in detail for detail in bundle.items[0].details)
        assert any("Likely impacted tests" in detail for detail in bundle.items[0].details)

    @pytest.mark.asyncio
    async def test_diff_review_stays_pending_without_diff(self, tmp_path, monkeypatch):
        """Without diff context, diff-review should remain pending."""
        monkeypatch.setattr("llm_council.engine.evidence._load_git_diff", lambda repo_root: "")

        plan = CapabilityPlan(
            execution_profile="light_tools",
            budget_class="normal",
            required_capabilities=["diff-review"],
        )

        bundle = await collect_capability_evidence(
            "Review the pending changes",
            "critic",
            "review",
            plan,
            repo_root=tmp_path,
        )

        assert bundle.executed_capabilities == []
        assert bundle.pending_capabilities == ["diff-review"]


class TestEvidenceRobustness:
    """Regression tests: evidence is optional and must degrade, never abort a run."""

    def test_load_git_diff_tolerates_missing_stdout(self, tmp_path, monkeypatch):
        """Regression: CompletedProcess.stdout=None must read as 'no diff', not crash."""

        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args=["git"], returncode=0, stdout=None, stderr=None)

        monkeypatch.setattr("llm_council.engine.evidence.subprocess.run", fake_run)

        assert _load_git_diff(tmp_path) == ""

    def test_load_git_diff_uses_replacement_safe_decoding(self, tmp_path, monkeypatch):
        """The git call must capture output as utf-8 text with non-raising decoding."""

        captured_kwargs: list[dict] = []

        def fake_run(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return subprocess.CompletedProcess(args=["git"], returncode=0, stdout="", stderr="")

        monkeypatch.setattr("llm_council.engine.evidence.subprocess.run", fake_run)

        _load_git_diff(tmp_path)

        assert captured_kwargs
        for kwargs in captured_kwargs:
            assert kwargs["capture_output"] is True
            assert kwargs["text"] is True
            assert kwargs["encoding"] == "utf-8"
            assert kwargs["errors"] == "replace"

    @pytest.mark.asyncio
    async def test_failing_collector_degrades_to_pending(self, tmp_path, monkeypatch):
        """A crashing collector leaves its capability pending while others still run."""

        def broken_diff(repo_root):
            raise AttributeError("'NoneType' object has no attribute 'strip'")

        monkeypatch.setattr("llm_council.engine.evidence._load_git_diff", broken_diff)

        plan = CapabilityPlan(
            execution_profile="light_tools",
            budget_class="normal",
            required_capabilities=["diff-review", "repo-analysis"],
        )

        bundle = await collect_capability_evidence(
            "Review the latest implementation changes",
            "critic",
            "review",
            plan,
            repo_root=tmp_path,
        )

        assert bundle.pending_capabilities == ["diff-review"]
        assert bundle.executed_capabilities == ["repo-analysis"]

    @pytest.mark.asyncio
    async def test_skipped_capability_never_runs_collector(self, tmp_path, monkeypatch):
        """Capabilities in skip_capabilities never invoke git, and are reported as left out
        by the run's scope - not as pending, which reads as a runtime failure."""

        calls: list[str] = []

        def record_diff(repo_root):
            calls.append("git-diff")
            return ""

        monkeypatch.setattr("llm_council.engine.evidence._load_git_diff", record_diff)

        plan = CapabilityPlan(
            execution_profile="light_tools",
            budget_class="normal",
            required_capabilities=["diff-review"],
        )

        bundle = await collect_capability_evidence(
            "Review the user-provided files",
            "critic",
            "review",
            plan,
            repo_root=tmp_path,
            skip_capabilities={"diff-review"},
        )

        assert calls == []
        assert bundle.executed_capabilities == []
        assert bundle.pending_capabilities == []
        assert bundle.skipped_capabilities == ["diff-review"]
        block = bundle.to_prompt_block()
        assert "left out by the run's scope (not a failure): diff-review" in block
        assert "not executed in this runtime" not in block


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _write(root: Path, rel: str, text: str = "x = 1\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _repo(root: Path, *rels: str) -> Path:
    """A git work tree holding `rels`, all tracked."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    for rel in rels:
        _write(root, rel)
    if rels:
        _git(root, "add", "--", *rels)
    return root


def _relative(root: Path, files: list[Path]) -> list[str]:
    return [path.relative_to(root).as_posix() for path in files]


class TestCandidateFiles:
    """Which files the evidence scan considers, in what order, and within what bounds."""

    def test_a_git_repo_is_scanned_wherever_its_code_lives(self, tmp_path: Path) -> None:
        """Any layout is covered and work in progress too; .gitignore keeps build output out."""
        _repo(
            tmp_path, "plugin/Nwiro/Source/Nwiro/Private/Tool.cpp", "api/src/server.ts", "README.md"
        )
        _write(tmp_path, ".gitignore", "Intermediate/\nnode_modules/\n")
        _write(tmp_path, "notes/new_idea.py")  # untracked, not ignored
        _write(tmp_path, "plugin/Nwiro/Intermediate/Build/Gen.cpp")  # ignored build output
        _write(tmp_path, "app/node_modules/lib/index.js")  # ignored dependency

        assert _relative(tmp_path, _candidate_files(tmp_path)) == [
            "api/src/server.ts",
            "notes/new_idea.py",
            "plugin/Nwiro/Source/Nwiro/Private/Tool.cpp",
            "README.md",
        ]

    def test_outside_git_the_conventional_directories_are_walked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
        for rel in ("src/core.py", "src/native/core.cpp", "plugin/Source/Tool.cpp", "docs/a.md"):
            _write(tmp_path, rel)

        assert _relative(tmp_path, _candidate_files(tmp_path)) == [
            "src/core.py",
            "src/native/core.cpp",
            "docs/a.md",
        ]

    @pytest.mark.parametrize("failure", ["git missing", "git timed out", "git refused"])
    def test_a_failing_git_falls_back_to_the_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
    ) -> None:
        _repo(tmp_path, "src/core.py", "plugin/Source/Tool.cpp")

        def failing_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            if failure == "git missing":
                raise FileNotFoundError("git")
            if failure == "git timed out":
                raise subprocess.TimeoutExpired(cmd="git", timeout=5.0)
            return subprocess.CompletedProcess(args=["git"], returncode=128, stdout=b"")

        monkeypatch.setattr("llm_council.engine.evidence.subprocess.run", failing_run)

        assert _relative(tmp_path, _candidate_files(tmp_path)) == ["src/core.py"]

    def test_git_lists_with_no_repository_program_and_only_below_repo_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """core.fsmonitor names a program git runs, so a scanned repo must not set it; the
        `-- .` pathspec keeps the listing below repo_root (the resolve check is the backstop)."""
        commands: list[list[str]] = []

        def recording_run(
            command: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            commands.append(command)
            return subprocess.CompletedProcess(args=command, returncode=0, stdout=b"")

        monkeypatch.setattr("llm_council.engine.evidence.subprocess.run", recording_run)

        _candidate_files(tmp_path)

        assert commands
        assert commands[0][:3] == ["git", "-c", "core.fsmonitor=false"]
        assert commands[0][-2:] == ["--", "."]

    def test_generated_files_are_left_out(self, tmp_path: Path) -> None:
        _repo(
            tmp_path,
            "Source/Tool.h",
            "Source/Tool.generated.h",
            "web/app.min.js",
            "k8s/zz_generated.deepcopy.go",
            "proto/api.pb.go",
            "ui/Form.Designer.cs",
        )

        assert _relative(tmp_path, _candidate_files(tmp_path)) == ["Source/Tool.h"]

    def test_oversized_files_are_left_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _repo(tmp_path, "src/small.py")
        _write(tmp_path, "src/big.py", "x = 1\n" * 100)
        _git(tmp_path, "add", "--", "src/big.py")
        monkeypatch.setattr(evidence_module, "MAX_CANDIDATE_BYTES", 50)

        assert _relative(tmp_path, _candidate_files(tmp_path)) == ["src/small.py"]

    def test_the_cap_keeps_code_before_other_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _repo(tmp_path, "docs/a.md", "docs/b.md", "src/z.py", "src/y.ts")
        monkeypatch.setattr(evidence_module, "MAX_CANDIDATE_FILES", 2)

        assert _relative(tmp_path, _candidate_files(tmp_path)) == ["src/y.ts", "src/z.py"]

    def test_a_symlink_is_not_followed_out_of_the_repository(self, tmp_path: Path) -> None:
        secret = _write(tmp_path / "outside", "secret.py", "TOKEN = 'x'\n")
        repo = _repo(tmp_path / "repo", "src/real.py")
        try:
            (repo / "src" / "link.py").symlink_to(secret)
        except (OSError, NotImplementedError):
            pytest.skip("this system does not let the tests create symlinks")
        _git(repo, "add", "--", "src/link.py")

        assert _relative(repo, _candidate_files(repo)) == ["src/real.py"]

    def test_a_link_is_skipped_even_where_symlinks_cannot_be_made(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same rule on any system: lstat reports a link, so the file is left out."""
        repo = _repo(tmp_path, "src/real.py", "src/link.py")
        link = repo / "src" / "link.py"
        lstat = Path.lstat

        def reporting_a_link(path: Path, *args: object, **kwargs: object) -> os.stat_result:
            if path == link:
                return os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 1, 0, 0, 10, 0, 0, 0))
            return lstat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "lstat", reporting_a_link)

        assert _relative(repo, _candidate_files(repo)) == ["src/real.py"]

    @staticmethod
    async def _first_hit(repo: Path, task: str) -> tuple[str, str]:
        plan = CapabilityPlan(
            execution_profile="light_tools",
            budget_class="normal",
            required_capabilities=["repo-analysis"],
        )
        bundle = await collect_capability_evidence(task, "critic", "review", plan, repo_root=repo)
        item = bundle.items[0]
        return item.details[1].replace("\\", "/"), item.summary

    @pytest.mark.asyncio
    async def test_repo_analysis_leads_with_the_files_the_task_names(self, tmp_path: Path) -> None:
        """Only a sample and ten hits reach the prompt, so files the task names go first -
        judged by the path inside the repository, not by the folders the repository sits in."""
        repo = _repo(tmp_path / "evidence-scan-workspace" / "repo")
        _write(repo, "api/alpha.ts", "// evidence of nothing in particular\n")
        _write(repo, "src/evidence_scan.py", "def evidence_scan():\n    return []\n")

        first_hit, summary = await self._first_hit(repo, "Review the evidence scan")

        assert first_hit.startswith("src/evidence_scan.py:1:")
        assert summary.startswith("Scanned 2 text files in the repository")

    @pytest.mark.asyncio
    async def test_repo_analysis_ranks_paths_naming_more_task_keywords_first(
        self, tmp_path: Path
    ) -> None:
        """A generic keyword ("tool") must not outrank a path that names the whole task."""
        repo = _repo(tmp_path)
        _write(repo, "api/tool_usage.ts", "// tool usage\n")
        _write(repo, "plugin/tool_bridge.cpp", "// tool bridge\n")

        first_hit, _summary = await self._first_hit(repo, "Review the tool bridge")

        assert first_hit.startswith("plugin/tool_bridge.cpp:1:")

    def test_ranking_comes_before_the_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file the task names survives the cap even when others sort before it."""
        _repo(tmp_path, "a/code.py", "z/evidence.py")
        monkeypatch.setattr(evidence_module, "MAX_CANDIDATE_FILES", 1)

        assert _relative(tmp_path, _candidate_files(tmp_path, ["evidence"])) == ["z/evidence.py"]

    def test_data_comes_after_prose_and_config(self, tmp_path: Path) -> None:
        """Tracked JSON collections must not crowd documentation out of the capped list."""
        _repo(tmp_path, "data/a.json", "docs/b.md", "config/c.yaml")

        assert _relative(tmp_path, _candidate_files(tmp_path)) == [
            "config/c.yaml",
            "docs/b.md",
            "data/a.json",
        ]

    def test_the_scan_stops_after_its_examination_budget_whatever_it_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tree of mostly oversized or linked files must not turn the loop into a long walk."""
        _repo(tmp_path, "a.py", "b.py", "c.py", "d.py")
        monkeypatch.setattr(evidence_module, "MAX_CANDIDATE_BYTES", 0)  # every file too big

        monkeypatch.setattr(evidence_module, "MAX_EXAMINED_FILES", 4)  # exactly enough
        assert not _candidate_files(tmp_path).truncated

        monkeypatch.setattr(evidence_module, "MAX_EXAMINED_FILES", 3)  # one short
        listed = _candidate_files(tmp_path)

        assert list(listed) == []
        assert listed.truncated

    def test_a_scan_cut_short_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _repo(tmp_path, "a.py", "b.py", "c.py")
        assert not _candidate_files(tmp_path).truncated

        monkeypatch.setattr(evidence_module, "MAX_LISTING_BYTES", 12)  # "a.py\0b.py\0" fits
        listed = _candidate_files(tmp_path)
        assert _relative(tmp_path, listed) == ["a.py", "b.py"]
        assert listed.truncated

        monkeypatch.setattr(evidence_module, "MAX_LISTING_BYTES", 16_000_000)
        monkeypatch.setattr(evidence_module, "MAX_CANDIDATE_FILES", 1)
        assert _candidate_files(tmp_path).truncated

    @pytest.mark.asyncio
    async def test_the_summary_reports_a_truncated_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _repo(tmp_path, "a.py", "b.py")
        monkeypatch.setattr(evidence_module, "MAX_CANDIDATE_FILES", 1)
        plan = CapabilityPlan(
            execution_profile="light_tools",
            budget_class="normal",
            required_capabilities=["repo-analysis"],
        )

        bundle = await collect_capability_evidence(
            "Review the module", "critic", "review", plan, repo_root=repo
        )

        summary = bundle.items[0].summary
        assert summary.startswith("Scanned 1 text files in the repository")
        assert "truncated" in summary


class TestCandidateFilesStayInsideTheRepository:
    """Whatever git prints or the file system links to, nothing outside repo_root is read."""

    def test_a_repo_root_below_the_work_tree_lists_only_its_own_files(self, tmp_path: Path) -> None:
        _repo(tmp_path, "outside.py", "sub/inside.py")

        assert _relative(tmp_path / "sub", _candidate_files(tmp_path / "sub")) == ["inside.py"]

    def test_names_git_prints_outside_the_repository_are_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        _write(repo, "src/ok.py")
        secret = _write(tmp_path, "secret.py", "TOKEN = 'x'\n")
        listing = b"../secret.py\0" + str(secret).encode() + b"\0src/ok.py\0"

        def listing_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(args=["git"], returncode=0, stdout=listing)

        monkeypatch.setattr("llm_council.engine.evidence.subprocess.run", listing_run)

        assert _relative(repo, _candidate_files(repo)) == ["src/ok.py"]

    def test_a_callers_git_environment_does_not_redirect_the_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = _repo(tmp_path / "target", "mine.py")
        other = _repo(tmp_path / "other", "theirs.py")
        monkeypatch.setenv("GIT_DIR", str(other / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(other))

        assert _relative(target, _candidate_files(target)) == ["mine.py"]

    def test_a_linked_directory_does_not_lead_out_of_the_repository(self, tmp_path: Path) -> None:
        """git still lists lib/x.py from its index after lib/ became a link to elsewhere."""
        _write(tmp_path / "elsewhere", "x.py", "TOKEN = 'x'\n")
        repo = _repo(tmp_path / "repo", "lib/x.py", "src/ok.py")
        (repo / "lib" / "x.py").unlink()
        (repo / "lib").rmdir()
        try:
            os.symlink(tmp_path / "elsewhere", repo / "lib", target_is_directory=True)
        except OSError:
            if sys.platform != "win32":
                pytest.skip("this system does not let the tests create directory links")
            import _winapi  # a junction needs no privilege

            _winapi.CreateJunction(str(tmp_path / "elsewhere"), str(repo / "lib"))

        assert _relative(repo, _candidate_files(repo)) == ["src/ok.py"]

    def test_the_git_diff_runs_no_program_the_repository_configures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        commands: list[list[str]] = []

        def recording_run(
            command: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="")

        monkeypatch.setattr("llm_council.engine.evidence.subprocess.run", recording_run)

        _load_git_diff(tmp_path)

        assert len(commands) == 2
        for command in commands:
            assert command[:3] == ["git", "-c", "core.fsmonitor=false"]
            assert "--no-ext-diff" in command and "--no-textconv" in command

    @staticmethod
    def _inject_config(monkeypatch: pytest.MonkeyPatch, key: str, value: str) -> None:
        """A caller's GIT_CONFIG_* reaching git: config the scanned repository never set."""
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", key)
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", value)

    def test_a_callers_git_config_does_not_change_the_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _repo(tmp_path / "repo", "mine.py")
        _write(repo, "notes.py")  # untracked work in progress
        hide = _write(tmp_path, "hide.txt", "*.py\n")
        self._inject_config(monkeypatch, "core.excludesFile", str(hide))
        # Positive control: the injected config does reach a plain git call.
        raw = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "notes.py" not in raw

        assert _relative(repo, _candidate_files(repo)) == ["mine.py", "notes.py"]

    def test_a_callers_git_config_does_not_change_the_diff(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _repo(tmp_path / "repo", "mine.py")
        _write(repo, "mine.py", "x = 'changed'\n")
        self._inject_config(monkeypatch, "color.diff", "always")
        # Positive control: the injected config does reach a plain git call.
        raw = subprocess.run(
            ["git", "-C", str(repo), "diff"], capture_output=True, text=True, check=True
        ).stdout
        assert "\x1b[" in raw

        diff = _load_git_diff(repo)

        assert "changed" in diff
        assert "\x1b[" not in diff, "escape codes would reach the diff parser"

    def test_a_callers_git_environment_does_not_redirect_the_diff(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = _repo(tmp_path / "target", "mine.py")
        other = _repo(tmp_path / "other", "theirs.py")
        _write(target, "mine.py", "x = 'changed in target'\n")
        _write(other, "theirs.py", "x = 'changed in other'\n")
        monkeypatch.setenv("GIT_DIR", str(other / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(other))

        diff = _load_git_diff(target)

        assert "changed in target" in diff
        assert "changed in other" not in diff

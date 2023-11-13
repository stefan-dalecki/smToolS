import dataclasses
import os
from typing import List, Optional, Self, Tuple

import git.diff
import pytest
from git import Repo

script_path = os.path.realpath(__file__)
tool_path = os.path.realpath(os.path.join(script_path, "..", ".."))


@pytest.fixture
def test_data_loc():
    return os.path.join("tests", "simo_tools", "test_tables")


class ArtifactError(Exception):
    pass


@dataclasses.dataclass
class RepoState:
    tracked_files_list: dataclasses.InitVar[List[git.diff.Diff]]
    untracked_files_list: dataclasses.InitVar[List[str]]

    def __post_init__(
        self, tracked_files_list: List[git.diff.Diff], untracked_files_list: List[str]
    ):
        self.tracked_files = {file.b_path for file in tracked_files_list}
        self.untracked_files = set(untracked_files_list)

    def __eq__(self, other_state: Self) -> Tuple[bool, Optional[str]]:
        tracked_diff = self.tracked_files.symmetric_difference(
            other_state.tracked_files
        )
        untracked_diff = self.untracked_files.symmetric_difference(
            other_state.untracked_files
        )
        if not tracked_diff and untracked_diff:
            return (
                False,
                f"File artifacts: `{', '.join(tracked_diff | untracked_diff)}`",
            )
        return True, None


@pytest.fixture(scope="session", autouse=True)
def verify_repo_unmodified():
    CURRENT_REPO = Repo(tool_path)
    CURRENT_COMMIT = CURRENT_REPO.head.commit
    starting_state = RepoState(CURRENT_COMMIT.diff(None), CURRENT_REPO.untracked_files)
    yield
    ending_state = RepoState(CURRENT_COMMIT.diff(None), CURRENT_REPO.untracked_files)
    repo_unmodified, file_changes = starting_state == ending_state
    if not repo_unmodified:
        message = f"Tests have modified repository state: {file_changes}."
        raise ArtifactError(message)

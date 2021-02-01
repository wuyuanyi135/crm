import pytest
from crm.utils.git import get_commit_hash


def test_get_commit_hash():
    print(get_commit_hash())

from git import Repo
from crm.utils.git import get_commit_hash
import argparse
import pytest


def check_uncommitted_change() -> bool:
    repo = Repo(search_parent_directories=True)
    return len(repo.index.diff(None)) > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-commit-check", action="store_true")
    parser.add_argument("-m", "--mark", help="mark", type=str)
    args = parser.parse_args()

    if not args.no_commit_check and check_uncommitted_change():
        raise RuntimeError("Uncommitted change detected.")

    test_args = []
    if args.mark is not None:
        test_args.extend(["-m", args.mark])
        html_name = f".test_report/{get_commit_hash()}_{args.mark}.html"
    else:
        html_name = f".test_report/{get_commit_hash()}.html"
    test_args.extend(["--benchmark-autosave", f"--html={html_name}", "--self-contained-html"])
    pytest.main(test_args)


if __name__ == '__main__':
    main()

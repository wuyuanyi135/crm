import git


def get_commit_hash():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha

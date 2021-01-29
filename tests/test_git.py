import unittest
from crm.utils.git import get_commit_hash

class TestGit(unittest.TestCase):
    def test_get_commit_hash(self):
        print(get_commit_hash())

if __name__ == '__main__':
    unittest.main()
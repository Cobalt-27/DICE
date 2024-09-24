# Some test are multi-process, may not stop the entire test if one of them failed.
# Thus, watch the output carefully, to see if all tests are passed.
# Better run all tests before commiting any changes to the codebase.
python3 -m tests.test_ep
python3 -m tests.test_prof
python3 -m tests.test_etrim
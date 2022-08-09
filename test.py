import subprocess
import sys

script_name = 'main.py'


subprocess.call(['python', script_name, "--scenario-name=simple_test", "--run-index=0", "--load-meta=0"])
subprocess.call(['python', script_name, "--scenario-name=simple_test", "--run-index=1", "--load-meta=0"])
subprocess.call(['python', script_name, "--scenario-name=simple_test", "--run-index=2", "--load-meta=0"])
subprocess.call(['python', script_name, "--scenario-name=simple_test", "--run-index=3", "--load-meta=0"])
subprocess.call(['python', script_name, "--scenario-name=simple_test", "--run-index=4", "--load-meta=0"])

subprocess.call(['python', script_name, "--scenario-name=simple_test", "--run-index=0", "--load-meta=1"])
subprocess.call(['python', script_name, "--scenario-name=simple_test", "--run-index=1", "--load-meta=1"])
subprocess.call(['python', script_name, "--scenario-name=simple_test", "--run-index=2", "--load-meta=1"])
subprocess.call(['python', script_name, "--scenario-name=simple_test", "--run-index=3", "--load-meta=1"])
subprocess.call(['python', script_name, "--scenario-name=simple_test", "--run-index=4", "--load-meta=1"])
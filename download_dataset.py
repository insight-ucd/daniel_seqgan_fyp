import subprocess
import os

def exec():
    file = open("dataset.sh", "r")
    commands = file.readlines()

    for cmd in check_files_exist(commands):
        cmd = cmd.rstrip().split(" ")
        list_files = subprocess.run(cmd)
        # print("The exit code was: %d" % list_files.returncode)

def check_files_exist(commands):
    file = "lastfm-dataset-1K//content/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"

    if os.path.exists(file):
        print("Last.fm file exists")
        return commands[2:]

    return commands
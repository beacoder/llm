import subprocess
import time


"""
This script keeps monitoring containers, and start killing longest running container every 5 minutes.
"""


MAX_CONTAINER_LIMIT = 10

def get_container_uptime():
    containers = subprocess.check_output(["docker", "ps", "--format", "{{.ID}} {{.RunningFor}}"])
    uptime_data = []
    for line in containers.decode().strip().split('\n'):
        try:
            cid, uptime, time_unit, *_ = line.split()
            uptime_data.append((cid, uptime, time_unit))
        except:
            continue
    return uptime_data

def kill_longest_running_container(uptime_data):
    longest_cid, uptime, time_unit = max(uptime_data, key=lambda x: x[1])
    subprocess.run(["docker", "kill", longest_cid])
    print(f"kill container {longest_cid}, since it's been up for longest time: {uptime} {time_unit}")


def main():
    while True:
        uptime_data = get_container_uptime()
        while (uptime_data and len(uptime_data) > MAX_CONTAINER_LIMIT):
            kill_longest_running_container(uptime_data)
            time.sleep(1)
            uptime_data = get_container_uptime()
        time.sleep(300)  # Check every 5 minutes


if __name__ == "__main__":
    main()

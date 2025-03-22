import docker
import subprocess
import time


MAX_CONTAINER_LIMIT = 10

client = docker.from_env()

def kill_longest_running_container():
    containers = client.containers.list()
    longest_running_container = min(containers, key=lambda c: c.attrs['State']['StartedAt'])
    container_id = longest_running_container.attrs['Id']
    subprocess.run(["docker", "kill", container_id])
    print(f"kill container {container_id}, since it's been up for longest time.")

def main():
    while True:
        containers = client.containers.list()
        while (containers and len(containers) > MAX_CONTAINER_LIMIT):
            kill_longest_running_container()
            containers = client.containers.list()
        time.sleep(300)  # Check every 5 minutes


if __name__ == "__main__":
    main()

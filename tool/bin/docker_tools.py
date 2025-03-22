from langchain_core.tools import tool
import docker
import os
import streamlit as st
import subprocess
import uuid


# tools definitions
# @note remember to return either a result or a message to inform the LLM
@tool
def create_file(path: str, filename: str, content: str):
    """Create a new file in a path with the specified content.

    Args:
        path:     The directory where to create the file
        filename: The name of the file to create
        content:  The content to write to the file
    """
    try:
        user_id= get_user_id()
        # make sure sandbox is ready
        get_or_create_sandbox(user_id)
        sandbox_name = f"sandbox_{user_id}"
        file_path = f"{sandbox_name}:{path}/{filename}"
        temp_file = os.path.expanduser(f"~/{user_id}_{filename}")
        with open(temp_file, 'w') as file:
            file.write(content)
        result = subprocess.run(["docker", "cp", temp_file, file_path], capture_output=True, text=True)
        os.remove(temp_file)
        if result.returncode == 0:
            return f"Created file {filename} in {path}"
        else:
            return f"Failed to create file {filename} in {path}: {result.stderr}"
    except Exception as e:
        return f"Failed to create file {filename} in {path}: {e}"

@tool
def make_directory(name: str, parent: str):
    """Create a new directory with the given name in the specified parent directory.

    Args:
        name:   The name of the new directory to create, e.g. testdir
        parent: The parent directory where the new directory should be created, e.g. /tmp
    """
    try:
        sandbox = get_or_create_sandbox(get_user_id())
        dir_path = parent + "/" +name
        result = execute_in_sandbox(sandbox, f"mkdir {dir_path}")
        if result == "":
            return f"Directory {name} created/verified in {parent}"
        else:
            return result
    except Exception as e:
        return f"Failed to create directory {name} in {parent}: {e}"

# super-powerful, capable of replacing numerous existing tools.
@tool
def run_command(command: str):
    """Run a command.

    Args:
        command: Command to run
    """
    try:
        sandbox = get_or_create_sandbox(get_user_id())
        return execute_in_sandbox(sandbox, command)
    except Exception as e:
        return f"Failed to run {command}: {e}"

@tool
def run_script(program: str, file: str, args: str = ""):
    """Run the script along with its specified arguments using the program.

    Args:
        program: Program to run the the script
        file:    Path to the script to run.  Supports relative paths and ~
        args:    Args for script to run
    """
    try:
        sandbox = get_or_create_sandbox(get_user_id())
        command = " ".join([program, file, args])
        return execute_in_sandbox(sandbox, command)
    except Exception as e:
        return f"Failed to run script {file}: {e}"


# docker APIs
def get_user_id():
    if 'user_uuid' not in st.session_state:
        st.session_state.user_uuid = uuid.uuid4()
        print(f"current user id is: {st.session_state.user_uuid}")
    return st.session_state.user_uuid

def get_or_create_sandbox(user_id):
    container = None
    client = docker.from_env()
    try:
        container = client.containers.get(f"sandbox_{user_id}")
        if container.status != 'running':
            cleanup_sandbox(user_id)
            container = None
    except Exception as e:
        pass
    if container is None:
        print(f"Start creating sandbox for user {user_id}")
        container = client.containers.run(
            image="python:3.10.12",
            command="sleep infinity",
            detach=True,
            name=f"sandbox_{user_id}"
        )
    return container

def execute_in_sandbox(sandbox, command):
    print(f"Start executing {command} in sandbox.")
    result = sandbox.exec_run(command, stdout=True, stderr=True, stream=False)
    return result.output.decode()

def cleanup_sandbox(user_id):
    client = docker.from_env()
    try:
        print(f"Start cleanup sandbox for user {user_id}")
        container = client.containers.get(f"sandbox_{user_id}")
        container.stop()
        container.remove()
        st.session_state.current_sandbox = None
    except Exception as e:
        return print(f"Failed to cleanup sandbox for user {user_id}: {e}")

def check_file_existence(file_path: str):
    sandbox = get_or_create_sandbox(get_user_id())
    result = sandbox.exec_run(f"test -f {file_path} && echo 'exists' || echo 'not found'")
    return "exists" in result.output.decode()

def check_path_existence(path: str):
    sandbox = get_or_create_sandbox(get_user_id())
    result = sandbox.exec_run(f"test -e {path} && echo 'exists' || echo 'not found'")
    return "exists" in result.output.decode()

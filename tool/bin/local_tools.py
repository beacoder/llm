from langchain_core.tools import tool
import os, subprocess, sys


"""
This script contains LLM tools suited for local file system.
"""


# default timeout for executing command/script is 5 minutes
SHELL_TIMEOUT=300

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
        expanded_path = os.path.expanduser(path + "/" + filename)
        with open(expanded_path, 'w') as file:
            file.write(content)
            return f"Created file {filename} in {path}"
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
        dir_path = parent + "/" +name
        expanded_dir_path = os.path.expanduser(dir_path)
        os.makedirs(expanded_dir_path, exist_ok=True)
        if os.path.exists(expanded_dir_path):
            return f"Directory {name} created/verified in {parent}"
        else:
            return f"Failed to create directory {name} in {parent}"
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
        result = subprocess.run(command,
                                shell=True,
                                capture_output=True,
                                text=True,
                                timeout=SHELL_TIMEOUT)
        if result.returncode == 0:
            return result.stdout
        else:
            return result.stderr
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
        command = " ".join([program, file, args])
        result = subprocess.run(command,
                                shell=True,
                                capture_output=True,
                                text=True,
                                timeout=SHELL_TIMEOUT)
        if result.returncode == 0:
            return result.stdout
        else:
            return result.stderr
    except Exception as e:
        return f"Failed to run {file}: {e}"

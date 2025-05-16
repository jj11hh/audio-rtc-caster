import os
import platform
import subprocess
import zipfile

def get_git_info():
    """Fetches Git branch/tag and commit hash."""
    try:
        # Try to get tag, fallback to branch
        tag_or_branch = subprocess.check_output(
            ['git', 'describe', '--tags', '--exact-match'], 
            stderr=subprocess.DEVNULL
        ).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        try:
            tag_or_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
            ).strip().decode('utf-8')
        except subprocess.CalledProcessError:
            tag_or_branch = "unknown"
    
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        commit_hash = "unknown"
        
    # Sanitize branch/tag name for filesystem
    tag_or_branch = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in tag_or_branch)

    return tag_or_branch, commit_hash

def build_go_server():
    """Builds the Go server."""
    go_server_dir = "server-go"
    executable_name = "server-go"
    if platform.system() == "Windows":
        executable_name += ".exe"
    
    output_path = os.path.join(go_server_dir, executable_name) # e.g., server-go/server-go.exe

    print(f"Building Go server. Output: {output_path}...")
    try:
        # Ensure GOPATH and GOROOT are sensible if running from unusual environments
        env = os.environ.copy()
        # Users might need to set these if go is not in PATH or standard locations
        # For this script, assume 'go' command is in PATH

        # Build command is run from the project root, targeting the ./server-go package
        # The output path -o needs to specify the desired location and name.
        process = subprocess.Popen(
            ['go', 'build', '-o', output_path, './server-go'], # Corrected build command
            cwd=os.getcwd(), # Run from the current working directory (project root)
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error building Go server:")
            print(f"Stdout: {stdout.decode('utf-8', errors='ignore')}")
            print(f"Stderr: {stderr.decode('utf-8', errors='ignore')}")
            return None
        print("Go server built successfully.")
        return output_path # This is already the relative path like "server-go/server-go.exe"
    except Exception as e:
        print(f"An exception occurred during Go build: {e}")
        return None

def create_zip_package(go_executable_path, client_folder_path):
    """Creates the zip package."""
    os_platform = platform.system().lower()
    tag_or_branch, commit_hash = get_git_info()
    
    zip_filename = f"rtc-caster-{os_platform}-{tag_or_branch}-{commit_hash}.zip"
    
    print(f"Creating zip package: {zip_filename}...")
    
    if not os.path.exists(go_executable_path):
        print(f"Error: Go executable not found at {go_executable_path}")
        return

    if not os.path.isdir(client_folder_path):
        print(f"Error: Client folder not found at {client_folder_path}")
        return

    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add Go server executable
            executable_basename = os.path.basename(go_executable_path)
            zf.write(go_executable_path, arcname=executable_basename)
            print(f"Added {executable_basename} to zip.")
            
            # Add client folder
            for root, _, files in os.walk(client_folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Create an archive name relative to the client_folder_path's parent
                    # e.g., client/index.html becomes client/index.html in the zip
                    archive_name = os.path.relpath(file_path, start=os.path.dirname(client_folder_path))
                    zf.write(file_path, arcname=archive_name)
            print(f"Added {client_folder_path} to zip.")
            
        print(f"Zip package created successfully: {zip_filename}")
    except Exception as e:
        print(f"Error creating zip package: {e}")

if __name__ == "__main__":
    # Assuming the script is run from the root of the repository
    workspace_root = os.getcwd() 
    
    go_executable_path_relative = build_go_server()
    
    if go_executable_path_relative:
        go_executable_full_path = os.path.join(workspace_root, go_executable_path_relative)
        client_folder_full_path = os.path.join(workspace_root, "client")
        create_zip_package(go_executable_full_path, client_folder_full_path)
    else:
        print("Skipping packaging due to build failure.")
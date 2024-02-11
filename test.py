import os
import time

def get_file_creation_time(file_path):
    try:
        # Get the creation time of the file in seconds since the epoch
        creation_time = os.path.getctime(file_path)
        return creation_time
    except OSError:
        # Handle the case where the file doesn't exist or other OSError
        return None

# Example usage
file_path = 'ge'  # Replace with the actual path to your file
creation_time = get_file_creation_time(file_path)

if creation_time is not None:
    print(f"The creation time of the file is: {creation_time} seconds since the epoch.")
    # If you want to convert to a human-readable format, you can use the time module
    creation_time_readable = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time))
    print(f"Human-readable format: {creation_time_readable}")
else:
    print("File not found or error occurred.")

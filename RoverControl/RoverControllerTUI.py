import subprocess
import sys

def display_menu():
    print("\n=== Rover Control Menu ===")
    print("1. Forward")
    print("2. Reverse")
    print("3. Stop")
    print("4. Turn")
    print("5. Brake")
    print("6. Toggle Autobrake (current: " + ("On" if autobrake_enabled else "Off") + ")")
    print("7. Exit")
    print("====================")

def get_numeric_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = float(input(prompt))
            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid numeric value")

def execute_command(command, distance=None, speed=None, angle=None, autobrake=None):
    cmd = ["./RoverControl", "-execute", command]
    if distance is not None:
        cmd.extend(["-distance", str(distance)])
    if speed is not None:
        cmd.extend(["-speed", str(speed)])
    if angle is not None:
        cmd.extend(["-angle", str(angle)])
    if autobrake is not None:
        cmd.extend(["-autobrake", "on" if autobrake else "off"])

    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffering
        )

        # Stream output as it comes
        while True:
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()

            if stdout_line:
                print(stdout_line, end='', flush=True)
            if stderr_line:
                print(stderr_line, end='', flush=True)

            # Check if process has finished
            if process.poll() is not None and not stdout_line and not stderr_line:
                break

        # Ensure process is complete and check return code
        process.wait()
        if process.returncode != 0:
            print(f"Command failed with exit code {process.returncode}")
    except FileNotFoundError:
        print("Error: 'RoverControl' executable not found. Ensure it's in the same directory.")
    except Exception as e:
        print(f"Error executing command: {e}")

def main():
    global autobrake_enabled
    autobrake_enabled = False  # Default to off

    while True:
        display_menu()
        choice = input("Select an option (1-7): ")

        if choice == "1":  # Forward
            distance = get_numeric_input("Enter distance (meters, >0): ", min_val=0.1)
            speed = get_numeric_input("Enter speed (0-100): ", min_val=0, max_val=100)
            execute_command("forward", distance=distance, speed=speed, autobrake=autobrake_enabled)

        elif choice == "2":  # Reverse
            distance = get_numeric_input("Enter distance (meters, >0): ", min_val=0.1)
            speed = get_numeric_input("Enter speed (0-100): ", min_val=0, max_val=100)
            execute_command("reverse", distance=distance, speed=speed, autobrake=autobrake_enabled)

        elif choice == "3":  # Stop
            execute_command("stop", autobrake=autobrake_enabled)

        elif choice == "4":  # Turn
            angle = get_numeric_input("Enter angle (degrees, positive for right, negative for left): ")
            speed = get_numeric_input("Enter speed (0-100): ", min_val=0, max_val=100)
            execute_command("turn", angle=angle, speed=speed, autobrake=autobrake_enabled)

        elif choice == "5":  # Brake
            execute_command("brake")

        elif choice == "6":  # Toggle Autobrake
            autobrake_enabled = not autobrake_enabled
            print(f"Autobrake is now: {'On' if autobrake_enabled else 'Off'}")

        elif choice == "7":  # Exit
            print("Exiting Rover Control App...")
            break

        else:
            print("Invalid choice. Please select 1-7.")

if __name__ == "__main__":
    main()
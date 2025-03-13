import subprocess
import sys

def display_menu():
    print("\n=== Rover Control Menu ===")
    print("1. Forward")
    print("2. Reverse")
    print("3. Stop")
    print("4. Turn")
    print("5. Exit")
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

def execute_command(command, distance=None, speed=None, angle=None):
    cmd = ["./rovercontrol", "-execute", command]
    if distance is not None:
        cmd.extend(["-distance", str(distance)])
    if speed is not None:
        cmd.extend(["-speed", str(speed)])
    if angle is not None:
        cmd.extend(["-angle", str(angle)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Command executed successfully:")
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}:")
        print(e.output)
    except FileNotFoundError:
        print("Error: 'rovercontrol' executable not found. Ensure it's in the same directory.")
    except Exception as e:
        print(f"Error executing command: {e}")

def main():
    while True:
        display_menu()
        choice = input("Select an option (1-5): ")

        if choice == "1":  # Forward
            distance = get_numeric_input("Enter distance (meters, >0): ", min_val=0.1)
            speed = get_numeric_input("Enter speed (0-100): ", min_val=0, max_val=100)
            execute_command("forward", distance=distance, speed=speed)

        elif choice == "2":  # Reverse
            distance = get_numeric_input("Enter distance (meters, >0): ", min_val=0.1)
            speed = get_numeric_input("Enter speed (0-100): ", min_val=0, max_val=100)
            execute_command("reverse", distance=distance, speed=speed)

        elif choice == "3":  # Stop
            execute_command("stop")

        elif choice == "4":  # Turn
            angle = get_numeric_input("Enter angle (degrees, positive for right, negative for left): ")
            speed = get_numeric_input("Enter speed (0-100): ", min_val=0, max_val=100)
            execute_command("turn", angle=angle, speed=speed)

        elif choice == "5":  # Exit
            print("Exiting Rover Control App...")
            break

        else:
            print("Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()
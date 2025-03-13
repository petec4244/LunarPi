#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <krpc.hpp>
#include <krpc/services/space_center.hpp>

void print_usage() {
    std::cout << "Usage:\n";
    std::cout << "  ./rovercontrol -execute forward -distance <meters> -speed <value>\n";
    std::cout << "  ./rovercontrol -execute reverse -distance <meters> -speed <value>\n";
    std::cout << "  ./rovercontrol -execute stop\n";
    std::cout << "  ./rovercontrol -execute turn -angle <degrees> -speed <value>\n";
    std::cout << "Examples:\n";
    std::cout << "  ./rovercontrol -execute forward -distance 10 -speed 10\n";
    std::cout << "  ./rovercontrol -execute turn -angle 30 -speed 10\n";
}

int execute_command(krpc::Client conn, krpc::services::SpaceCenter &spaceCenter, const std::string &command, float distance, float speed, float angle) {
    try {
        // Get the active vessel (rover)
        auto vessel = spaceCenter.active_vessel();
        auto control = vessel.control();
        auto flight = vessel.flight(vessel.orbit().body().reference_frame());

        if (command == "forward" || command == "reverse") {
            // Validate inputs
            if (distance <= 0 || speed <= 0 || speed > 100) {
                std::cerr << "Error: Distance and speed must be positive; speed must be <= 100\n";
                return 1;
            }

            // Set direction
            float direction = (command == "forward") ? 1.0f : -1.0f;
            control.set_forward(direction);
            control.set_throttle(speed / 100.0f);

            // Estimate duration based on distance and speed
            // Assume speed of 1 (100%) = 5 m/s for simplicity
            float estimated_speed = (speed / 100.0f) * 5.0f; // Adjust based on rover's actual speed
            float duration = distance / estimated_speed;

            std::cout << "Executing " << command << ": distance=" << distance << "m, speed=" << speed << "%\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(duration * 1000)));

            // Stop
            control.set_forward(0.0f);
            control.set_throttle(0.0f);
            std::cout << command << " completed.\n";
        }
        else if (command == "stop") {
            control.set_forward(0.0f);
            control.set_throttle(0.0f);
            control.set_yaw(0.0f);
            std::cout << "Stopped rover.\n";
        }
        else if (command == "turn") {
            if (speed <= 0 || speed > 100) {
                std::cerr << "Error: Speed must be positive and <= 100\n";
                return 1;
            }

            // Current heading
            float initial_heading = flight.heading();

            // Target heading
            float target_heading = initial_heading + angle;
            if (target_heading >= 360.0f) target_heading -= 360.0f;
            if (target_heading < 0.0f) target_heading += 360.0f;

            // Set yaw (angle normalized to -1 to 1)
            float yaw = (angle > 0) ? 1.0f : -1.0f; // Right or left turn
            control.set_yaw(yaw);
            control.set_throttle(speed / 100.0f);

            // Wait until heading is reached (or timeout after 5 seconds)
            float tolerance = 2.0f; // Allow 2-degree tolerance
            auto start_time = std::chrono::steady_clock::now();
            while (true) {
                float current_heading = flight.heading();
                float heading_diff = std::abs(current_heading - target_heading);
                if (heading_diff <= tolerance || heading_diff >= 360.0f - tolerance) break;

                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - start_time).count();
                if (elapsed > 5) {
                    std::cout << "Turn timeout after 5 seconds\n";
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            // Reset controls
            control.set_yaw(0.0f);
            control.set_throttle(0.0f);
            std::cout << "Turn completed: angle=" << angle << " degrees\n";
        }
        else {
            std::cerr << "Invalid command: " << command << "\n";
            return 1;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string execute, distance_str, speed_str, angle_str;
    float distance = 0.0f, speed = 0.0f, angle = 0.0f;

    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];
        if (i + 1 >= argc) {
            print_usage();
            return 1;
        }
        if (arg == "-execute") {
            execute = argv[i + 1];
        } else if (arg == "-distance") {
            distance_str = argv[i + 1];
            distance = std::stof(distance_str);
        } else if (arg == "-speed") {
            speed_str = argv[i + 1];
            speed = std::stof(speed_str);
        } else if (arg == "-angle") {
            angle_str = argv[i + 1];
            angle = std::stof(angle_str);
        } else {
            print_usage();
            return 1;
        }
    }

    // Validate command and parameters
    if (execute == "forward" || execute == "reverse") {
        if (distance_str.empty() || speed_str.empty()) {
            std::cerr << "Error: Forward/reverse requires -distance and -speed\n";
            return 1;
        }
    } else if (execute == "turn") {
        if (angle_str.empty() || speed_str.empty()) {
            std::cerr << "Error: Turn requires -angle and -speed\n";
            return 1;
        }
    } else if (execute != "stop") {
        std::cerr << "Invalid command: " << execute << "\n";
        return 1;
    }

    // Connect to kRPC server
    krpc::Client conn;
    try {
        conn = krpc::connect("192.168.1.11", 50000);
        std::cout << "Connected to kRPC server\n";
    } catch (const std::exception &e) {
        std::cerr << "Failed to connect to kRPC server: " << e.what() << "\n";
        return 1;
    }

    krpc::services::SpaceCenter spaceCenter(conn);
    return execute_command(conn, spaceCenter, execute, distance, speed, angle);
}
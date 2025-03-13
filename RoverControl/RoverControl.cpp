#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <krpc.hpp>
#include <krpc/services/space_center.hpp>

// Global autobrake state
bool autobrake_enabled = false; // Default to off

void print_usage() {
    std::cout << "Usage:\n";
    std::cout << "  ./RoverControl -execute forward -distance <meters> -speed <value> [-autobrake on/off]\n";
    std::cout << "  ./RoverControl -execute reverse -distance <meters> -speed <value> [-autobrake on/off]\n";
    std::cout << "  ./RoverControl -execute stop [-autobrake on/off]\n";
    std::cout << "  ./RoverControl -execute turn -angle <degrees> -speed <value> [-autobrake on/off]\n";
    std::cout << "  ./RoverControl -execute brake\n";
    std::cout << "Examples:\n";
    std::cout << "  ./RoverControl -execute forward -distance 10 -speed 10 -autobrake on\n";
    std::cout << "  ./RoverControl -execute brake\n";
}

int execute_command(krpc::Client* conn, const std::string &command, float distance, float speed, float angle) {
    try {
        std::cout << "Constructing SpaceCenter...\n" << std::flush;
        krpc::services::SpaceCenter spaceCenter(conn);

        std::cout << "Getting active vessel...\n" << std::flush;
        krpc::services::SpaceCenter::Vessel vessel;
        try {
            vessel = spaceCenter.active_vessel();
            std::cout << "Active vessel retrieved successfully\n" << std::flush;
        } catch (const std::exception &e) {
            std::cerr << "Error: No active vessel found: " << e.what() << "\n";
            return 1;
        }

        auto control = vessel.control();
        auto reference_frame = vessel.orbit().body().reference_frame();
        auto flight = vessel.flight(reference_frame);
        std::cout << "Vessel and controls initialized\n" << std::flush;

        std::cout << "Executing command: " << command << "\n" << std::flush;

        if (command == "forward" || command == "reverse") {
            if (distance <= 0 || speed <= 0 || speed > 100) {
                std::cerr << "Error: Distance and speed must be positive; speed must be <= 100\n";
                return 1;
            }

            float direction = (command == "forward") ? 1.0f : -1.0f;
            std::cout << "Setting forward: " << direction << ", throttle: " << (speed / 100.0f) << "\n" << std::flush;
            control.set_forward(direction);
            control.set_throttle(speed / 100.0f);

            float estimated_speed = (speed / 100.0f) * 5.0f;
            float duration = distance / estimated_speed;
            std::cout << "Sleeping for " << duration << " seconds\n" << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(duration * 1000)));

            control.set_forward(0.0f);
            control.set_throttle(0.0f);
            std::cout << command << " completed.\n" << std::flush;

            // Apply autobrake if enabled
            if (autobrake_enabled) {
                std::cout << "Applying autobrake...\n" << std::flush;
                control.set_brakes(true);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                control.set_brakes(false);
                std::cout << "Autobrake released\n" << std::flush;
            }
        }
        else if (command == "stop") {
            control.set_forward(0.0f);
            control.set_throttle(0.0f);
            control.set_yaw(0.0f);
            std::cout << "Stopped rover.\n" << std::flush;
        }
        else if (command == "turn") {
            if (speed <= 0 || speed > 100) {
                std::cerr << "Error: Speed must be positive and <= 100\n";
                return 1;
            }

            float initial_heading = flight.heading();
            std::cout << "Initial heading: " << initial_heading << "\n" << std::flush;

            float target_heading = initial_heading + angle;
            if (target_heading >= 360.0f) target_heading -= 360.0f;
            if (target_heading < 0.0f) target_heading += 360.0f;
            std::cout << "Target heading: " << target_heading << "\n" << std::flush;

            float yaw = (angle > 0) ? 1.0f : -1.0f;
            control.set_yaw(yaw);
            control.set_throttle(speed / 100.0f);
            std::cout << "Set yaw: " << yaw << ", throttle: " << (speed / 100.0f) << "\n" << std::flush;

            float tolerance = 2.0f;
            auto start_time = std::chrono::steady_clock::now();
            int steps = 0;
            while (true) {
                float current_heading = flight.heading();
                float heading_diff = std::abs(current_heading - target_heading);
                if (heading_diff <= tolerance || heading_diff >= 360.0f - tolerance) {
                    std::cout << "Reached target heading: " << current_heading << " after " << steps << " steps\n" << std::flush;
                    break;
                }

                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - start_time).count();
                if (elapsed > 5) {
                    std::cout << "Turn timeout after 5 seconds, current heading: " << current_heading << "\n" << std::flush;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                steps++;
            }

            control.set_yaw(0.0f);
            control.set_throttle(0.0f);
            std::cout << "Turn completed: angle=" << angle << " degrees\n" << std::flush;

            // Apply autobrake if enabled
            if (autobrake_enabled) {
                std::cout << "Applying autobrake...\n" << std::flush;
                control.set_brakes(true);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                control.set_brakes(false);
                std::cout << "Autobrake released\n" << std::flush;
            }
        }
        else if (command == "brake") {
            control.set_brakes(true);
            std::cout << "Brakes applied\n" << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            control.set_brakes(false);
            std::cout << "Brakes released\n" << std::flush;
        }
        else {
            std::cerr << "Invalid command: " << command << "\n";
            return 1;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n" << std::flush;
        return 1;
    }
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string execute, distance_str, speed_str, angle_str, autobrake_str;
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
        } else if (arg == "-autobrake") {
            autobrake_str = argv[i + 1];
            if (autobrake_str == "on") {
                autobrake_enabled = true;
            } else if (autobrake_str == "off") {
                autobrake_enabled = false;
            } else {
                std::cerr << "Error: -autobrake must be 'on' or 'off'\n";
                return 1;
            }
        } else {
            print_usage();
            return 1;
        }
    }

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
    } else if (execute != "stop" && execute != "brake") {
        std::cerr << "Invalid command: " << execute << "\n";
        return 1;
    }

    // Connect to kRPC server
    krpc::Client conn;
    std::string host = "192.168.1.11";
    unsigned int rpc_port = 50000;
    unsigned int stream_port = 50001;
    try {
        conn = krpc::connect("lunarPi", host, rpc_port, stream_port);
        std::cout << "Connected to kRPC server at " << host << ":" << rpc_port << "\n" << std::flush;
    } catch (const std::exception &e) {
        std::cerr << "Failed to connect to kRPC server at " << host << ":" << rpc_port << ": " << e.what() << "\n" << std::flush;
        return 1;
    }

    return execute_command(&conn, execute, distance, speed, angle);
}
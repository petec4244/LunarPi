import zmq
import gphoto2 as gp
import time
import pickle
import krpc
import sys
from PIL import Image
import io

# ZMQ setup
context = zmq.Context()
socket_to_pc = context.socket(zmq.REQ)
socket_to_pc.connect("tcp://<PC_IP>:5555")  # Command PC IP
socket_from_pc = context.socket(zmq.REP)
socket_from_pc.bind("tcp://*:5556")

# kRPC setup
conn = krpc.connect(name="LunarPi", address="<PC_IP>", rpc_port=50000, stream_port=50001)
vessel = conn.space_center.active_vessel

# Canon EOS 6D setup
camera = gp.Camera()
camera.init()

def capture_image():
    file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
    cam_file = camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    image_data = cam_file.get_data_and_size()
    camera.file_delete(file_path.folder, file_path.name)
    return image_data

def is_night(image_data):
    img = Image.open(io.BytesIO(image_data)).convert("L")  # Grayscale
    brightness = sum(img.getdata()) / (img.width * img.height)
    return brightness < 50  # Threshold for "night" (adjustable)

def explore():
    print("Exploring...")
    while True:
        image_data = capture_image()
        socket_to_pc.send(pickle.dumps(image_data))
        instruction = pickle.loads(socket_to_pc.recv())
        execute_instruction(instruction)
        time.sleep(1)

def turn_360():
    print("Performing 360 turn...")
    vessel.control.wheel_steering = 0.5
    vessel.control.throttle = 0.3
    time.sleep(6)  # ~360° at this speed (calibrate in KSP)
    vessel.control.throttle = 0
    vessel.control.wheel_steering = 0

def stay_put():
    print("Staying put...")
    vessel.control.throttle = 0
    vessel.control.wheel_steering = 0
    time.sleep(5)  # Wait before next command

def sleep():
    print("Sleeping (night detected)...")
    image_data = capture_image()
    while is_night(image_data):
        vessel.control.throttle = 0
        time.sleep(5)  # Check every 5s
        image_data = capture_image()
    print("Waking up...")

def establish_link():
    print("Establishing link...")
    socket_to_pc.send(pickle.dumps("ping"))
    response = pickle.loads(socket_to_pc.recv())
    if response == "pong":
        print("Link established!")
        return True
    return False

def mark_location():
    print("Marking location...")
    # Simulate by logging position (KSP flags need manual placement)
    pos = vessel.position(vessel.orbit.body.reference_frame)
    print(f"Marked: {pos}")

def emergency_stop():
    print("Emergency stop!")
    vessel.control.throttle = 0
    vessel.control.wheel_steering = 0
    sys.exit(0)

def execute_instruction(instruction):
    if instruction == "move_left":
        vessel.control.wheel_steering = 0.5
        vessel.control.throttle = 0.3
    elif instruction == "move_right":
        vessel.control.wheel_steering = -0.5
        vessel.control.throttle = 0.3
    elif instruction == "stop":
        vessel.control.throttle = 0
        vessel.control.wheel_steering = 0
    socket_from_pc.send(pickle.dumps("executed"))
    socket_from_pc.recv()

def main():
    functions = {
        "explore": explore,
        "360": turn_360,
        "stay": stay_put,
        "sleep": sleep,
        "link": establish_link,
        "mark": mark_location,
        "emergency": emergency_stop
    }
    if len(sys.argv) > 1 and sys.argv[1] in functions:
        functions[sys.argv[1]]()
    else:
        print("Usage: python lunarpi.py [explore|360|stay|sleep|link|mark|emergency]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        camera.exit()
        conn.close()



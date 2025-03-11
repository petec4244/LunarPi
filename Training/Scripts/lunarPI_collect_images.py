import zmq
import gphoto2 as gp
import time
import pickle

# ZMQ setup
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://<PC_IP>:5555")  

# Canon EOS 6D setup
camera = gp.Camera()
camera.init()

def capture_and_send():
    print("Starting image collection...")
    image_count = 0
    try:
        while True:
            # Capture image
            file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
            cam_file = camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
            image_data = cam_file.get_data_and_size()
            camera.file_delete(file_path.folder, file_path.name)

            # Send to PC
            socket.send(pickle.dumps(image_data))
            response = pickle.loads(socket.recv())
            if response == "saved":
                image_count += 1
                print(f"Image {image_count} sent and saved")
            else:
                print("Error saving image")

            time.sleep(0.5)  # ~2 images/second, adjust as needed
    except KeyboardInterrupt:
        print(f"Collected {image_count} images")
        camera.exit()
        socket.close()

if __name__ == "__main__":
    capture_and_send()
#
# This application tests some command executions with the RoverControl app
#

import subprocess

def execute_command(command, distance=None, speed=None, angle=None):
    cmd = ["./rovercontrol", "-execute", command]
    if distance is not None:
        cmd.extend(["-distance", str(distance)])
    if speed is not None:
        cmd.extend(["-speed", str(speed)])
    if angle is not None:
        cmd.extend(["-angle", str(angle)])
    subprocess.run(cmd)

# Examples
execute_command("forward", distance=10, speed=10)
execute_command("turn", angle=30, speed=10)
execute_command("stop")
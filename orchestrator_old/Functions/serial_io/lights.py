import serial
import serial.tools.list_ports
from dependencies import BaseTool
import time


def detect_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "USB" in port.description or "UART" in port.description:
            return port.device
    return None


def led(serial_port, command='0'):

    SerialObj = serial.Serial(serial_port, 9600, timeout=1)

    time.sleep(1)

    try:
        print("in function")
        while True:
            # LED on
            if command == '1':
                print("turning on")
                SerialObj.write(b'1')
                return "The lights has been turned on"
                # LED off
            elif command == '0':
                print("turning off")
                SerialObj.write(b'0')
                return "The lights has been turned off"
            else:
                print("Invalid command.")
                return "an error has taken place"

    except KeyboardInterrupt:
        print("\nSerial Port Closed")
    finally:
        SerialObj.close()
        print("\nSerial Port Closed")



class LightsOn(BaseTool):
    name = "lights_on"
    description = "Useful to illuminate the area by turning on the lights, gives the ability to turn on the lights and make the room brighter"

    def __init__(self):
        super().__init__()

    def _run(self, tool_input: str, **kwargs) -> str:
        """Send data to via serial port."""
        print(tool_input)
        print("turning on")
        port = detect_port()
        print(f"port{port}")
        if port:
            message = led(port, command='1')
            print(message)
        else:
            message = "unable to find any lights"
        return f"\noutput: {message}"


class LightsOff(BaseTool):
    name = "lights_off"
    description = "Useful in case light is undesirable turning off the lights, can off the lights and make the room darker"

    def __init__(self):
        super().__init__()

    def _run(self, tool_input: str, **kwargs) -> str:
        """Send data to via serial port."""
        print(tool_input)
        print("turning off")
        port = detect_port()
        if port:
            message = led(port, command='0')
        else:
            message = "unable to find any lights"
        return f"\noutput: {message}"


lights_on = LightsOn()
lights_off = LightsOff()

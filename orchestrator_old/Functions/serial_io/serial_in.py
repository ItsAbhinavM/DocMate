from dependencies import BaseTool
import serial
import time


class SerialInput:
    def __init__(self, port, baud_rate):
        self.serial_obj = serial.Serial(port, baud_rate, timeout=1)

    def read(self):
        out = ''
        # Read the output for 5 seconds
        end_time = 5
        while end_time > 0:
            time.sleep(1)
            end_time -= 1
            output = self.serial_obj.readline().decode('utf-8').strip()
            if output:
                print(f"Received: {output}")
                out += output + "\n"
        return out

    def close_serial(self):
        self.serial_obj.close()


class SerialIn(BaseTool):
    name = "read serial port input"
    description = "Useful to read data from serial port"

    def __init__(self):
        super().__init__()

    def _run(self, tool_input: str, **kwargs) -> str:
        """Send data to via serial port."""
        message = tool_input
        serial_controller = SerialInput('/dev/ttyUSB0', 9600)
        try:
            out = serial_controller.read()
            return f"\nquery: {message}\noutput: {out}"

        except KeyboardInterrupt:
            serial_controller.close_serial()
            print("\nSerial Port Closed")
        return f"\nquery: {message}\noutput: canceled operation"


serial_in = SerialIn()

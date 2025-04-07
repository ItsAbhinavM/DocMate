from dependencies import BaseTool
import serial
import time


class SerialInput:
    def __init__(self, port, baud_rate):
        self.serial_obj = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(3)

    def write(self, message):
        self.serial_obj.write(message.encode('utf-8') + b'\n')
        self.serial_obj.write(b'1\n')
        time.sleep(1)
        self.serial_obj.write(b'0\n')
        time.sleep(1)

        # Read the output after write operation
        while self.serial_obj.in_waiting > 0:
            time.sleep(1)
            output = self.serial_obj.readline().decode('utf-8').strip()
            print(f"Received: {output}")

    def close_serial(self):
        self.serial_obj.close()


class SerialOut(BaseTool):
    name = "write to serial port out"
    description = "Useful to write to serial port"

    def __init__(self):
        super().__init__()

    def _run(self, tool_input: str, **kwargs) -> str:
        """read data from serial port."""
        message = tool_input
        serial_controller = SerialInput('/dev/ttyUSB0', 9600)
        out = serial_controller.write(message)
        serial_controller.close_serial()
        return f"\nquery: {message}\noutput: data transferred successfully, output: {out}"


serial_out = SerialOut()

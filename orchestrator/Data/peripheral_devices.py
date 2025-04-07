import re
import subprocess


def usb_devices():
    device_re = re.compile("Bus\s+(?P<bus>\d+)\s+Device\s+(?P<device>\d+).+ID\s(?P<id>\w+:\w+)\s(?P<tag>.+)$", re.I)
    df = subprocess.check_output("lsusb").decode("utf-8")
    devices = []
    for i in df.split('\n'):
        if i:
            info = device_re.match(i)
            if info:
                info = info.groupdict()
                info['device'] = '/dev/bus/usb/%s/%s' % (info.pop('bus'), info.pop('device'))
                devices.append(info)

    return devices


def pcie_devices():
    df = subprocess.check_output(["lspci", "-vv"]).decode("utf-8")
    devices = [i for i in df.split("\n") if len(i) > 7]  # 7 -> len(00:00.0)

    return devices

data = dict()
device, current_section = None, None

for line in pcie_devices():
    if line[0].isdigit():
        spit_data = line.split(' ', 1)
        device = spit_data[0].strip()
        current_section = spit_data[1].strip()

        data[device] = {}
        data[device] = {"device": current_section, "data": {}}
    elif ':' in line:
        key, value = map(str.strip, line.split(':', 1))
        data[device]["data"][key] = value

for i in data:
    print(f"{i}:{data[i]}")

from vna.scipiCommands import get_corrected_data_array, SParam
from timeit import default_timer as timer


scpi_commands = [
    ":CALCulate1:PARameter1:DEFine S11",
    ":CALCulate1:PARameter2:DEFine S12",
    ":CALCulate1:PARameter3:DEFine S13",
    ":CALCulate1:PARameter4:DEFine S14",
    ":CALCulate1:PARameter5:DEFine S21",
    ":CALCulate1:PARameter6:DEFine S22",
    ":CALCulate1:PARameter7:DEFine S23",
    ":CALCulate1:PARameter8:DEFine S24",
    ":CALCulate1:PARameter9:DEFine S31",
    ":CALCulate1:PARameter10:DEFine S32",
    ":CALCulate1:PARameter11:DEFine S33",
    ":CALCulate1:PARameter12:DEFine S34",
    ":CALCulate1:PARameter13:DEFine S41",
    ":CALCulate1:PARameter14:DEFine S42",
    ":CALCulate1:PARameter15:DEFine S43",
    ":CALCulate1:PARameter16:DEFine S44",
]

import socket
import struct
import numpy as np

# Replace with your VNA’s IP address
VNA_IP = "192.168.255.2"
PORT = 5025  # Standard SCPI over TCP/IP


def await_completion(sock: socket):
    sock.sendall("*OPC?\n".encode())
    rx = int(s.recv(100).decode().strip())
    if rx == 1:
        return
    else:
        rx = int(s.recv(100).decode().strip())


def send_scpi_command(sock, command):
    """Sends an SCPI command to the VNA."""
    sock.sendall((command + "\n").encode())


def receive_binary_data(sock, num_points, number_of_ports):
    """Receives binary S-parameter data and handles SCPI headers."""
    num_floats = (
        num_points * number_of_ports * 2
    )  # 16 S-parameters, each with Real+Imag
    expected_bytes = num_floats * 8  # Convert floats to bytes

    # Read the response header
    raw_response = sock.recv(100)  # Read initial bytes
    print("Response Start:", raw_response[:20])  # Print first 20 bytes to check header

    # If header starts with #, it indicates IEEE 488.2 binary format
    if raw_response.startswith(b"#"):
        header_length = int(raw_response[1:2])  # Extract length of size field
        data_length = int(
            raw_response[2 : 2 + header_length]
        )  # Extract actual data size
        binary_data = raw_response[2 + header_length :]
        while len(binary_data) < data_length:
            binary_data = binary_data + sock.recv(
                data_length
            )  # Read remaining binary data
    else:
        binary_data = raw_response  # No header, assume direct binary data

    # # Ensure we read full expected data
    # while len(binary_data) < expected_bytes:
    #     chunk = sock.recv(expected_bytes - len(binary_data))
    #     if not chunk:
    #         raise ValueError("Socket closed before full data received.")
    #     binary_data += chunk

    # don't use .strip()
    binary_data = binary_data[:-1]
    if len(binary_data) != expected_bytes:

        print(
            f"Received {len(binary_data)} bytes (Expected {expected_bytes})"
        )  # Debugging info

    # Convert binary to float array
    data = struct.unpack(f"{num_floats}d", binary_data)
    s_matrix = np.array(data).reshape(number_of_ports, num_points, 2)

    return s_matrix


# Connect to VNA
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(1000)
    s.connect((VNA_IP, PORT))

    # need to add the traces first

    # Define all 16 S-parameters
    for cmd in scpi_commands:
        print(f"Sending {cmd}")
        send_scpi_command(s, cmd)

    # Request number of points
    send_scpi_command(s, ":SENSe:SWEep:POINts?")
    num_points = int(s.recv(100).decode().strip())  # Read the response

    print(f"Number of measurement points: {num_points}")

    # # send_scpi_command(s, ":FORMat:DATA REAL,32")  # Set binary mode
    # response = send_scpi_command(s, ":FORMat:DATA?")  # Verify format
    #
    # print(response)

    # send_scpi_command(s, ":INITiate:IMMediate")  # Start a new sweep
    # send_scpi_command(s, "*WAI")  # Wait until the measurement is complete

    n_repeats = 100
    # Request binary data (all 16 S-parameters)
    for i in range(n_repeats):
        times = []
        start_time = timer()
        for sparam in SParam:

            send_scpi_command(
                s, get_corrected_data_array(channel_number=1, sparam=sparam)
            )
            # Receive and process binary data
            s_parameters = receive_binary_data(s, num_points, 1)
        elapsed_time = timer() - start_time
        times.append(elapsed_time)
    print(f"Execution time : {sum(times)/n_repeats:.6f} seconds")
    # Print sample output
    # for i, s_param in enumerate(
    #     [
    #         "S11",
    #         "S12",
    #         "S13",
    #         "S14",
    #         "S21",
    #         "S22",
    #         "S23",
    #         "S24",
    #         "S31",
    #         "S32",
    #         "S33",
    #         "S34",
    #         "S41",
    #         "S42",
    #         "S43",
    #         "S44",
    #     ]
    # ):
    #     print(f"{s_param} (Real, Imag):", s_parameters[i][:5])  # Print first 5 points

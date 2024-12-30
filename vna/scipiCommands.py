from enum import Enum
from pyvisa import Resource


class SParam(Enum):
    S11 = "S11"
    S21 = "S21"
    S31 = "S31"
    S41 = "S41"
    S12 = "S12"
    S22 = "S22"
    S32 = "S32"
    S42 = "S42"
    S13 = "S13"
    S23 = "S23"
    S33 = "S33"
    S43 = "S43"
    S14 = "S14"
    S24 = "S24"
    S34 = "S34"
    S44 = "S44"


class DisplayAddCommands(Enum):
    ADD_TRACE = "TRC"
    ADD_CHANNEL_TRACE = "CH_TRC"
    ADD_WINDOW_TRACE = "WIN_TRC"
    ADD_WINDOW_CHANNEL_TRACE = "WIN_CH_TRC"


def await_completion(inst: Resource):
    while True:
        if int(inst.query("*OPC?")) == 1:
            break


def set_start_or_stop_freq_command_string(
    channel_number: int, freq_hz: int, stop_flag=False
) -> str:
    if stop_flag:
        start_or_stop = "STOP"
    else:
        start_or_stop = "STARt"
    channel_number = validate_channel_number(channel_number)
    return f":SENSe{channel_number}:FREQuency:{start_or_stop} {freq_hz}"


def validate_channel_number(channel_number):
    if channel_number < 1 or channel_number > 256:
        channel_number = 1
    return channel_number


def preset_system_command_string() -> str:
    return ":SYSTem:PRESet"


def add_to_display_command_string(add_command: DisplayAddCommands) -> str:
    return f":DISPlay:ADD:FUNCtion:EXECute {add_command.value}"


def add_s_param_measurement_command_string(
    *, sparam: SParam, channel_number: int = 1, trace_number: int = 1
) -> str:
    """
     This command sets and gets the measurement parameter of the selected trace, for the selected chann
    <tnum>:={[1]-256}, represents the measurement trace number. If not specified, <tnum> defaults to 1.
    :param sparam:
    :param channel_number: channel_number {[1] -256}, represents the measurement channel number. If not specified, <cnum> defaults to 1.
    :return:
    """
    channel_number = validate_channel_number(channel_number)
    return f":CALCulate{channel_number}:PARameter{trace_number}:DEFine {sparam.value}"


def get_corrected_data_array(*, channel_number: int, sparam: SParam) -> str:
    """
    3.4.85
    Get Corrected Data Array (:SENSe{[1]-200}:DATA:CORRdata? S<XY>)
    :return:
    """
    return f":SENSe{channel_number}:DATA:CORRdata? {sparam.value}"


def debug_command(inst: Resource):
    """
    Assuming you were able to confirm that the instrument understood the command you sent,
    it means the reading part is the issue, which is easier to troubleshoot. You can try
    different standard values for the read_termination, but if nothing works you can use
    the read_bytes() method. This method will read at most the number of bytes specified.
    So you can try reading one byte at a time till you encounter a time out. When that happens most
    likely the last character you read is the termination character.
    :param inst:
    :return:
    """
    while True:
        print(inst.read_bytes(1))

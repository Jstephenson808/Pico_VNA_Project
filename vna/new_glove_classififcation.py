from vna.VNA_enums import SParam
from vna.VNA_utils import mhz_to_hz

frequency_ranges = [[mhz_to_hz(200), mhz_to_hz(350)], [mhz_to_hz(250), mhz_to_hz(300)]]
s_parameter_sets = [
    [SParam.S21, SParam.S31, SParam.S41],
    [SParam.S11, SParam.S41],
    [SParam.S21, SParam.S41],
    [SParam.S11],
]

import pyvisa
from pyvisa.resources import MessageBasedResource

from vna.VNA_defaults import NI_VISA_DLL_PATH

VNA_SOCKET_ADDRESS = ""
rm = pyvisa.ResourceManager(NI_VISA_DLL_PATH)
print(rm.list_resources())

vna_handle: MessageBasedResource = rm.open_resource(VNA_SOCKET_ADDRESS)
print(vna_handle.query("*IDN?"))

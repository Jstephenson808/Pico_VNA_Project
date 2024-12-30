class Frequency:
    @staticmethod
    def mhz_to_hz(mhz):
        """
        utility function to convert mhz to hz
        :param mhz: MHz value
        :return: value in Hz
        """
        return mhz * 1_000_000

    @staticmethod
    def hz_to_mhz(hz):
        """
        utility function to convert hz to Mhz
        :param hz: Hz value
        :return: value in MHz
        """
        return hz / 1_000_000

    @staticmethod
    def ghz_to_hz(ghz):
        """
        utility function to convert GHz to Hz
        :param ghz: GHz value
        :return: value in Hz
        """
        return ghz * 1_000_000_000

    @staticmethod
    def hz_to_ghz(hz):
        """
        utility function to convert Hz to GHz
        :param hz: Hz value
        :return: value in GHz
        """
        return hz / 1_000_000_000

    def __init__(self, freq_value_Hz):
        self.freq_value_hz = freq_value_Hz

    def __repr__(self):
        return f"Frequency Object {self.freq_value_hz}Hz"

    def __str__(self):
        return f"{self.freq_value_hz}Hz"

    def __eq__(self, other):
        return self.freq_value_hz == other.freq_value_hz

    def __ne__(self, other):
        return self.freq_value_hz != other.freq_value_hz

    def __gt__(self, other):
        return self.freq_value_hz > other.freq_value_hz

    def __lt__(self, other):
        return self.freq_value_hz < other.freq_value_hz

    def __ge__(self, other):
        return self.freq_value_hz >= other.freq_value_hz

    def __le__(self, other):
        return self.freq_value_hz <= other.freq_value_hz

    def __add__(self, other):
        return Frequency(self.freq_value_hz + other.freq_value_hz)

    def __sub__(self, other):
        return Frequency(self.freq_value_hz - other.freq_value_hz)

    def __mul__(self, other):
        return Frequency(self.freq_value_hz * other)

    def __int__(self):
        return int(self.freq_value_hz)

    def get_freq_hz(self):
        return self.freq_value_hz

    def get_freq_mhz(self):
        return Frequency.hz_to_mhz(self.freq_value_hz)

    def get_freq_ghz(self):
        return Frequency.hz_to_ghz(self.freq_value_hz)

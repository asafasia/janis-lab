# Copyright 2014-2021 Keysight Technologies
"""
Software demodulation with custom windows
"""


import numpy as np
from dataclasses import dataclass


@dataclass
class Window:
    channel_number: int
    sample_rate: float
    length: int
    filename: str
    frequency: float = 0
    phase: float = 0
    shift: bool = False

    def __post_init__(self) -> None:
        self.times = np.arange(self.length) / self.sample_rate

    def load_window(self) -> np.ndarray:
        """Load window from file or create boxcar"""
        if self.filename == '':
            w = np.ones_like(self.times, dtype=complex)
        elif self.filename.endswith('.npy'):
            w = np.load(self.filename)
        elif self.filename.endswith('.npz'):
            # load dictionary which may contain multiple arrays
            # can be used to include all windows in one file
            # use the one called 'window_{n}' or 'window_{n}_I'
            d = np.load(self.filename)
            win_name = f'window_{self.channel_number+1}'
            if win_name in d.keys():
                w = d[win_name]
            elif win_name + '_I' in d.keys():
                # not yet in use, but can allow different weights for I and Q outputs
                w = d[win_name + '_I']
            else:
                raise ValueError(f'Expected array name {win_name} '
                                 f'not found in file {self.filename}')
        else:
            raise ValueError(f'Unexpected custom window file {self.filename}')
        return w

    def get_window(self) -> np.ndarray:
        """
        Construct complex window function
        """
        w = self.load_window()
        w = self._normalize_window(w)
        do_shift = (not self.filename) or self.shift
        if do_shift:
            # modulate and phase-shift the window
            w = (w * np.exp(-2j * np.pi * self.times * self.frequency)
                 * np.exp(-1j * np.pi * self.phase / 180))
        return w

    @staticmethod
    def _normalize_window(w: np.ndarray) -> np.ndarray:
        """normalize a window by the RMS"""
        norm = np.sqrt( np.mean( np.abs(w)**2 ) )
        return w / norm


@dataclass
class WindowManager:
    """
    Creates and manages demodulation windows

    Parameters
    ----------
    nFDM: int
        Numer of independent demod channels
    sample_rate: float
        Acquisiton rample rate in Sa/s

    """
    nFDM: int
    sample_rate: float

    def __post_init__(self) -> None:
        self.dt = 1/self.sample_rate
        self._windows = [None for n in range(self.nFDM)]

    def setup_demod_channel(self,
                            channel_number: int,
                            frequency: float,
                            length: int,
                            filename: str = '',
                            phase: float = 0,
                            shift_custom_window: bool = False,
                            ) -> None:
        """
        Assign settings to a frequency-domain demodulation channel

        Parameters
        ----------
        channel_number: int
            FDM channel index. Zero-indexed
        frequency: float
            Demodulation center frequency in Hz
        length: int
            Window length in samples
        filename: str
            If a Custom window is desired, this is the absolute path to it
            Can be .npy or .npz file.
            If .npz, the keys should include channel_number
        phase: float
            Demodulation phase
        shift_custom_window: bool
            If window_type is Custom, this parameter sets whether it is
            shifted by frequency and phase 
        """
        window = Window(
            channel_number=channel_number,
            sample_rate=self.sample_rate,
            length=length,
            frequency=frequency,
            phase=phase,
            filename=filename,
            shift=shift_custom_window
        )
        self._windows[channel_number] = window

    def get_window(self, n: int) -> np.ndarray:
        """
        Get complex integration window for a given FDM channel

        Parameters
        ----------
        channel_number: int
            Demodulation chanel index. Zero-indexed
        
        Returns
        -------
        window: np.ndarray
            Complex window
        """
        return self._windows[n].get_window()



def demodulate(window, acq1, acq2=None, noise_amp=None):
    """
    Perform the demodulation for a single FDM channel.

    Parameters
    ----------
    window : [complex]
        Complex demodulation window
    acq1 : [float]
        Signal acquired on input channel 1
    acq2 : [float]
        Signal acquired on input channel 2,
        or None for single-channel acquisition
    noise_amp: float
        Standard deviation of noise to add to each sample
    """
    # construct the signal
    iq_mode = acq2 is not None
    if iq_mode:
        if acq2.shape != acq1.shape:
            raise ValueError(
                'Traces acq1 and acq2 must be of same length')
    # add noise
    if noise_amp is not None:
        noise1 = noise_amp * np.random.randn(*acq1.shape)
        acq1 = acq1 + noise1
        if iq_mode:
            noise2 = noise_amp * np.random.randn(*acq2.shape)
            acq2 = acq2 + noise2

    sig = acq1
    if iq_mode:
        sig = sig + 1j * acq2

    # perform the demodulation and integration
    sig_demod = sig * window
    sig_int = sig_demod.mean(axis=-1)
    if not iq_mode:
        sig_int *= 2
    return sig_int


def threshold(value, threshold, axis='I'):
    """
    Threshold the IQ value.

    Parameters
    ----------
    value: complex or [complex]
        values(s) to threshold

    threshold : float
        voltage threshold for discrimination

    axis : string
        axis to threshold along (either 'I' or 'Q')

    """
    # get the threshhold value
    if axis == 'I':
        op = np.real
    else:
        op = np.imag
    threshed = op(value) > threshold
    return threshed

import numpy as np
import random
from torch import Tensor
from typing import Optional
import librosa
import torch
import torchaudio
from torch_audiomentations import Compose,PitchShift,AddColoredNoise,Shift
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict
#Fixed seed for reproduci
random.seed(42)
# Default data augmentation
class CustomFade(BaseWaveformTransform):
    def __init__(self, max_fade_in_ratio, max_fade_out_ratio, fade_shapes, mode='per_example', 
            sample_rate=16000, p=1.0):
        """A custom time-series fade class built upon torchaudio's Fade. Allows 
            for fade from both front and back in a variety of shapes.

        Args:
            max_fade_in_ratio (float): Maximum fade in ratio for the signal
            max_fade_out_ratio (float): Minimum fade in ratio for the signal
            fade_shapes (list): List of shapes to apply fade with
            mode (str, optional): Mode for which to apply the augmentations. 
                Defaults to 'per_example'.
            sample_rate (int, optional): Sample rate of the signal. Defaults to 16000.
            p (float, optional): Probability with which to apply the augmentation. 
                Defaults to 1.0.
        """
        super().__init__(
            sample_rate=sample_rate,
            mode=mode,
            p=p
        )

        self.fade_shapes = fade_shapes
        self.max_fade_out_ratio = max_fade_out_ratio
        self.max_fade_in_ratio = max_fade_in_ratio

    def randomise_params(self, samples):
        """Randomises params for incoming batch of augmentations

        Args:
            samples (Tensor): Incoming data batch of signals
        """
        # Get data details
        batch_size, num_channels, sample_length = samples.shape

        # Pre-calculate the appropriate maxes of fades form both sides
        fade_in_size = int(sample_length*self.max_fade_in_ratio)
        fade_out_size = int(sample_length*self.max_fade_out_ratio)

        # Create a transform for every sample in batch
        if self.mode == 'per_example':
            shapes = random.choices(self.fade_shapes, k=batch_size)

            # Generate a new transform for every sample
            self.batch_transforms = []
            for i in range(batch_size):
                in_fade =random.randint(0, fade_in_size)
                out_fade =random.randint(0, fade_out_size)

                new_trans = torchaudio.transforms.Fade(fade_in_len=in_fade,
                    fade_out_len=out_fade, fade_shape=shapes[i])
                # Store generated per example transforms
                self.batch_transforms.append(new_trans)

        # Want to use the same transform across the batch
        elif self.mode == 'per_batch':
            shape_idx = random.randint(0, len(self.fade_shapes))
            in_fade = random.randint(0, fade_in_size)
            out_fade = random.randint(0, fade_out_size)
            self.transform = torchaudio.transforms.Fade(fade_in_len=fade_in_size,
                fade_out_len=fade_out_size, fade_shape=self.fade_shapes[shape_idx])


    def apply_transform(self,
            samples: Tensor = None,
            sample_rate: Optional[int] = None,
            targets: Optional[Tensor] = None,
            target_rate: Optional[int] = None) -> ObjectDict:
        """Mandatory apply augmentation function. Is called by super class when 
            augs are composed together Actually in charge of applying the semantics 
            of the specific augmentation to the batch of samples.  

        Args:
            samples (Tensor, optional): Incoming data batch of signals. Defaults to None.
            sample_rate (Optional[int], optional): Sample rate of signal. Defaults to None.
            targets (Optional[Tensor], optional): Target data samples. Defaults to None.
            target_rate (Optional[int], optional): Target sample rate. Defaults to None.

        Returns:
            ObjectDict: Dictionary of samples and other meta data. Should be accessed as needed
        """
        batch_size, num_channels, num_samples = samples.shape

        self.randomise_params(samples)
        
        if self.mode == 'per_example':
            for i in range(batch_size):
                samples[i, ...] = self.batch_transforms[i](samples[i][None])

        elif self.mode == 'per_batch':
            for i in range(batch_size):
                samples[i, ...] = self.transform(samples[i][None])

        return ObjectDict(samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate)
    
class CustomTimeMasking(BaseWaveformTransform):
    def __init__(self, max_signal_ratio, options, mode='per_example', 
            sample_rate=16000, p=1.0):
        """Custom time masking class wrapped around torchaudio TimeMasking. 
            Can either place signal with pure white noise or make constant 0. 
            Both the length of the mask and where it starts is randomly selected. 
            Location is not bound.

            (As of right nw only 0 masking works not white noise!)

        Args:
            max_signal_ratio (float): Maximum ratio of the signal that can be masked at any given time
            options (list): List of application options. White Noise or 0 padding 
            mode (str, optional): Mode for which to apply the augmentations. 
                Defaults to 'per_example'.
            sample_rate (int, optional): Sample rate of the signal. Defaults to 16000.
            p (float, optional): Probability with which to apply the augmentation. 
                Defaults to 1.0.
        """
        super().__init__(
            sample_rate=sample_rate,
            mode=mode,
            p=p
        )

        self.max_signal_ratio = max_signal_ratio
        self.options = options 

    def randomise_params(self, samples):
        """Randomises params for incoming batch of augmentations

        Args:
            samples (Tensor): Incoming data batch of signals
        """
        # should add in noise masking later as well

        # Get data details
        batch_size, num_channels, sample_length = samples.shape

        # Pre-calculate the appropriate max mask
        max_mask = self.max_signal_ratio * sample_length

        # Generates a random use constant value masking, every time called new mask applied
        self.constant_aug = torchaudio.transforms.TimeMasking(
                time_mask_param=max_mask)

    
    def apply_transform(self,
            samples: Tensor = None,
            sample_rate: Optional[int] = None,
            targets: Optional[Tensor] = None,
            target_rate: Optional[int] = None) -> ObjectDict:
        """Mandatory apply augmentation function. Is called by super class when 
            augs are composed together Actually in charge of applying the semantics 
            of the specific augmentation to the batch of samples.  

        Args:
            samples (Tensor, optional): Incoming data batch of signals. Defaults to None.
            sample_rate (Optional[int], optional): Sample rate of signal. Defaults to None.
            targets (Optional[Tensor], optional): Target data samples. Defaults to None.
            target_rate (Optional[int], optional): Target sample rate. Defaults to None.

        Returns:
            ObjectDict: Dictionary of samples and other meta data. Should be accessed as needed
        """
        batch_size, num_channels, num_samples = samples.shape

        self.randomise_params(samples)
        
        if self.mode == 'per_example':
            for i in range(batch_size):
                samples[i, ...] = self.constant_aug(samples[i][None])

        return ObjectDict(samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate)
# class CustomTimeStretch(BaseWaveformTransform):
#     def __init__(self, min_stretch, max_stretch, n_fft=2048, mode='per_example', 
#             sample_rate=16000, p=1.0):
#         super().__init__(
#             sample_rate=sample_rate,
#             mode=mode,
#             p=p
#         )

#         # Pre-calculated the necessary stft variables
#         self.n_fft = n_fft
#         # This calculation is a rearranged form of an equation in torch stft source (init function)
#         self.n_freq = int((self.n_fft / 2) + 1)

#         self.min_stretch = min_stretch
#         self.max_stretch = max_stretch

    
#     def randomise_params(self, samples):
#         # Get data details
#         batch_size, num_channels, sample_length = samples.shape

#         # Gets a batch size worth of valid stretching parameters
#         self.rand_nums = []
#         for _ in range(batch_size):
#             rand_num = random.uniform(self.min_stretch, self.max_stretch)
#             self.rand_nums.append(round(rand_num, 2))


#     def apply_transform(self,
#             samples: Tensor = None,
#             sample_rate: Optional[int] = None,
#             targets: Optional[Tensor] = None,
#             target_rate: Optional[int] = None) -> ObjectDict:
#         """Mandatory apply augmentation function. Is called by super class when 
#             augs are composed together Actually in charge of applying the semantics 
#             of the specific augmentation to the batch of samples.  

#         Args:
#             samples (Tensor, optional): Incoming data batch of signals. Defaults to None.
#             sample_rate (Optional[int], optional): Sample rate of signal. Defaults to None.
#             targets (Optional[Tensor], optional): Target data samples. Defaults to None.
#             target_rate (Optional[int], optional): Target sample rate. Defaults to None.

#         Returns:
#             ObjectDict: Dictionary of samples and other meta data. Should be accessed as needed
#         """

#         batch_size, num_channels, num_samples = samples.shape
#         # Make a call to randomise params or batch of augs
#         self.randomise_params(samples)

#         if self.mode == 'per_example':
#             for i in range(batch_size):
#                 # Generates real and imaginary stft data
#                 stft_data = torch.stft(samples[i], n_fft=2048)
#                 # Stretches the stft data using a phase vocoder
#                 stretched_stft = phase_vocoder(stft_data, self.rand_nums[i],phase_advance=self.n_freq )
#                 # Recombines the real and imaginary component of stft with an inverse 
#                 recombined_data = torch.istft(stretched_stft, n_fft=self.n_fft)
#                 # Uses circular padding or cropping to return to expected signal length
#                 samples[i, ...] = enforce_length(recombined_data, num_samples)


#         return ObjectDict(samples=samples,
#             sample_rate=sample_rate,
#             targets=targets,
#             target_rate=target_rate)
def select_possible_augs(sample_rate=20000):
    aug_list = []
    p=1
    sample_rate=sample_rate
    augs_to_include = ['pitch_shift','fade','white_noise','time_shift']
    for name in augs_to_include:
        if name == 'pitch_shift':
            aug = PitchShift(min_transpose_semitones=-15,max_transpose_semitones=15, p=p, sample_rate=sample_rate)
        elif name == 'fade':
            aug = CustomFade(max_fade_in_ratio=0.5,max_fade_out_ratio=0.5,fade_shapes=['linear', 'logarithmic', 'exponential'], p=p, sample_rate=sample_rate)
        elif name == 'white_noise':
            aug = AddColoredNoise(min_snr_in_db=3,max_snr_in_db=30,min_f_decay=-1,max_f_decay=0,p=p, sample_rate=sample_rate)
        elif name == 'mixed_noise':
            aug = AddColoredNoise(min_snr_in_db=3,max_snr_in_db=30,min_f_decay=-2,max_f_decay=2, p=p, sample_rate=sample_rate)
        # elif name == 'time_masking':
        #     aug = CustomTimeMasking(max_signal_ratio=0.125, p=p, sample_rate=sample_rate)
        elif name == 'time_shift':
            aug = Shift(max_shift=0.5,min_shift=-0.5,shift_unit='fraction', p=p, sample_rate=sample_rate)
        # elif name == 'time_stretch':
        #     aug = CustomTimeStretch(min_stretch=0.5,max_stretch=1.5, p=p, sample_rate=sample_rate)
        elif name == 'None':
            continue
        else:
            raise ValueError(f'Augmentation name: {name} not recognised')

        aug_list.append(aug)
    max_k = len(aug_list)
    #rand_augs = random.choices(aug_list,k=max_k)
    random.shuffle(aug_list)
    comp_augs = Compose(aug_list,shuffle=False)
    comp_augs1 = comp_augs
    return comp_augs,comp_augs1
def padding(pad):
    def f(sound):
        return np.pad(sound, pad, 'constant')

    return f


def random_crop(size):
    def f(sound):
        org_size = len(sound)
        start = random.randint(0, org_size - size)
        return sound[start: start + size]

    return f

def multrandom_crop(input_length,n_crops):
    def f(sound):
        sounds = []
        org_size = len(sound)
        for i in range(n_crops):
            start = random.randint(0,org_size - input_length)
            sounds.append(sound[start:start+input_length])
        sounds = np.array(sounds)
        return sounds
        
    return f

def normalize(factor):
    def f(sound):
        return sound / factor

    return f


# For strong data augmentation
def random_scale(max_scale, interpolate='Linear'):
    def f(sound):
        scale = np.power(max_scale, random.uniform(-1, 1))
        output_size = int(len(sound) * scale)
        ref = np.arange(output_size) / scale
        if interpolate == 'Linear':
            ref1 = ref.astype(np.int32)
            ref2 = np.minimum(ref1 + 1, len(sound) - 1)
            r = ref - ref1
            scaled_sound = sound[ref1] * (1 - r) + sound[ref2] * r
        elif interpolate == 'Nearest':
            scaled_sound = sound[ref.astype(np.int32)]
        else:
            raise Exception('Invalid interpolation mode {}'.format(interpolate))

        return scaled_sound

    return f


def random_gain(db):
    def f(sound):
        return sound * np.power(10, random.uniform(-db, db) / 20.0)

    return f


# For testing phase
def multi_crop(input_length, n_crops):
    def f(sound):
        stride = (len(sound) - input_length) // (n_crops - 1)
        sounds = [sound[stride * i: stride * i + input_length] for i in range(n_crops)]
        return np.array(sounds)

    return f

def center_crop(input_length):
    def f(sound):
        start = (len(sound) - input_length) // 2
        return sound[start: start + input_length]

    return f


# For BC learning
def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if fs == 16000 or fs == 20000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    #no xrange anymore supported
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

    return sound

# Convert time representation
def to_hms(time):
    h = int(time // 3600)
    m = int((time - h * 3600) // 60)
    s = int(time - h * 3600 - m * 60)
    if h > 0:
        line = '{}h{:02d}m'.format(h, m)
    else:
        line = '{}m{:02d}s'.format(m, s)

    return line

class MFCCProcessor:#(20,189)
    def __init__(self, name='mfcc', sr=20000, order=20, n_fft=2048, hop=160, der_order=0):
        self.name = name
        self.sr = sr
        self.order = order
        self.n_fft = n_fft
        self.hop = hop
        self.der_order = der_order
    #def __call__(self, pkg, cached_file=None):
    def __call__(self, wav, cached_file=None):
        #pkg = self.format_package(pkg)
        #wav = pkg['chunk']
        #y = wav.data.numpy()
        y = wav
        #y = y.squeeze(-1)
        max_frames = y.shape[0] // self.hop

        # if cached_file is not None:
        #     # 加载预先计算的数据
        #     mfcc = torch.load(cached_file)
        #     #beg_i = pkg['chunk_beg_i'] // self.hop
        #     #end_i = pkg['chunk_end_i'] // self.hop
        #     mfcc = mfcc[:, beg_i:end_i]
        #     #pkg[self.name] = mfcc
        # else:
        #     # 计算MFCC
        #     #y = y.astype(np.float32)
        #     #y = y.cpu().numpy()
        #     #y = np.array(y)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr,
                                    n_mfcc=self.order,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop)[:, :max_frames]
        if self.der_order > 0:
            deltas = [mfcc]
            for n in range(1, self.der_order + 1):
                deltas.append(librosa.feature.delta(mfcc, order=n))
            mfcc = np.concatenate(deltas)

            #pkg[self.name] = torch.tensor(mfcc.astype(np.float32))

        # 将分辨率改为跳帧长度
        #pkg['dec_resolution'] = self.hop
        return torch.tensor(mfcc)

class MFCCProcessor1:
    def __init__(self,sample_rate=20000,n_mfcc=20,n_fft=2048,hop_length=160):
        self.sample_rate = sample_rate
        self.n_mfcc =n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def __call__(self,wav):
        transform = torchaudio.transforms.MFCC(self.sample_rate,self.n_mfcc,melkwargs={"n_fft":self.n_fft,"hop_length":self.hop_length})
        max_frames = 189
        mfcc = transform(wav.cpu())[:,:max_frames]
        return mfcc
class LPS(object):#(1025,189)

    def __init__(self, n_fft=2048, hop=160,
                 win=400, der_order=0,
                 name='lps',
                 device='cpu'):
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.name = name
        self.der_order=der_order
        self.device = device

    # @profile
    def __call__(self, wav):
        y = wav
        # max_frames = y.shape[0] // self.hop
        max_frames = y.shape[0] // self.hop
        # if cached_file is not None:
        #     # load pre-computed data
        #     X = torch.load(cached_file)
        #     beg_i = pkg['chunk_beg_i'] // self.hop
        #     end_i = pkg['chunk_end_i'] // self.hop
        #     X = X[:, beg_i:end_i]
        #     pkg['lps'] = X
        # else:
            #print ('Chunks wav shape is {}'.format(wav.shape))
            #wav = wav.to(self.device)
            #y = y.astype(np.float32)
            #y = torch.tensor(y)
        X = torch.stft(y, self.n_fft,
                        self.hop, self.win, return_complex=True)
        X = torch.abs(X)[:, :max_frames]
        #X = torch.abs(X)[:, :, :max_frames]
        X = torch.abs(X)[:,:max_frames]
        X = 10 * torch.log10(X ** 2 + 10e-20)
        if self.der_order > 0 :
            deltas=[X]
            for n in range(1,self.der_order+1):
                deltas.append(librosa.feature.delta(X.numpy(),order=n))
            X=torch.from_numpy(np.concatenate(deltas))
     
        #     pkg[self.name] = X
        # # Overwrite resolution to hop length
        # pkg['dec_resolution'] = self.hop
        return X

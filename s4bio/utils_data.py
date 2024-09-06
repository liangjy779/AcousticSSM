import sys
import torch
import torchaudio

def batch_to_log_mel_spec(samples):
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=20000,n_mels=128,n_fft=1024,hop_length=512).to(samples.device)
    mel_specs = transform(samples)
    power = 2
    log_mel_specs = 20.0/power * torch.log10(mel_specs + sys.float_info.epsilon)
    return log_mel_specs

def batch_get_mag_phase(stft_data):
    for i in range(stft_data.shape[0]):
        sub_data = stft_data[i]
        phase = torch.atan2(sub_data[...,1],sub_data[...,0])
        magnitudes = torch.sqrt(torch.pow(sub_data[...,0],2)+torch.pow(sub_data[...,1],2))
        stft_data[i,...,0] = magnitudes
        stft_data[i,...,1] = phase
    return stft_data

def batch_to_log_mel_spec_plus_stft(samples):
    mel_specs = batch_to_log_mel_spec(samples)#output shape[batch_size,mel,time]
    #mel_specs = mel_specs.squeeze()

    #need to remove channel dimension
    
    #samples = samples.squeeze()
    stft_data = torch.stft(samples,n_fft=1024,hop_length=512,return_complex=False)
    transform = torchaudio.transforms.MelScale(n_mels=128,sample_rate=20000,n_stft=stft_data.shape[1]).to(samples.device)

    trans_real = transform(stft_data[...,0])
    trans_img = transform(stft_data[...,1])
    trans_stft_data = torch.stack((trans_real,trans_img),dim=-1)#(batch_size,n_mels,timestep,2)
    
    #get the phase and magnitude of the stft data
    mag_phase = batch_get_mag_phase(trans_stft_data)
    magnitude = mag_phase[...,0]
    phase = mag_phase[...,1]
    
    #convert amplitude spectrogram to log scale
    log_mel_scale_stft_amplitude = 20.0/2 * torch.log10(magnitude+sys.float_info.epsilon)
    #Finally stacks the 3 channels of input data together along new channel 
    final_samples = torch.stack((mel_specs,log_mel_scale_stft_amplitude,phase),dim=1)
    #final_samples = mel_specs.unsqueeze(1)
    return final_samples

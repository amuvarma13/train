
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import numpy as np
import IPython.display as ipd
import librosa
import soundfile as sf

from naturalspeech3_facodec.ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download





fa_encoder = FACodecEncoder(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

fa_encoder.load_state_dict(torch.load(encoder_ckpt))
fa_decoder.load_state_dict(torch.load(decoder_ckpt))

def process_audio_and_get_vq_id():

    test_wav_path = "spks.wav"
    test_wav = librosa.load(test_wav_path, sr=16000)[0]
    test_wav = torch.from_numpy(test_wav).float()
    test_wav = test_wav.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        # encode
        enc_out = fa_encoder(test_wav)
        print(enc_out.device)
        print(enc_out.shape)

        # quantize
        _, _, _, _, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)

        return spk_embs

spk_embs = process_audio_and_get_vq_id()


def decode_tensor(tensor_input):
  tensor_input.to("cpu")
  vq_post_emb = fa_decoder.vq2emb(tensor_input)
  recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)
  sf.write("e_tok.wav", recon_wav[0][0].cpu().detach().numpy(), 16000)


def convert_to_wav(generated_ids, tokenizer):
   
    sos_indices = (generated_ids[0] == 128000).nonzero(as_tuple=True)[0]
    second_sos_index = sos_indices[-1].item()

    # Find the next occurrence of 128009 (EOS) after the second SOS
    eos_index = (generated_ids[0][second_sos_index:] == 128009).nonzero(as_tuple=True)[0][0].item() + second_sos_index

    # Extract the tokens between SOS and EOS
    extracted_tokens = generated_ids[0][second_sos_index : eos_index]

    decoded_text = tokenizer.decode(extracted_tokens)
    token_to_find = 128257
    token_to_remove = 128263

    # Check if the token exists in the tensor
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()  # Get the last occurrence index
        # Crop the tensor to the values after the last occurrence
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        # If token not found, no cropping is done
        cropped_tensor = generated_ids

    mask = cropped_tensor != token_to_remove
    cropped_tensor = cropped_tensor[mask].view(cropped_tensor.size(0), -1)

    print(cropped_tensor.tolist())

    processed_tensor = cropped_tensor - 128266
    original_shape = processed_tensor.shape
    new_dim_1 = (original_shape[1] // 6) * 6
    processed_tensor = processed_tensor[:, :new_dim_1]
    processed_tensor_content = processed_tensor[:, ::6]

    processed_tensor_prosody = processed_tensor[:, 1::6]
    processed_tensor_prosody = processed_tensor_prosody - 1024

    processed_tensor_content_1 = processed_tensor[:, 2::6]
    processed_tensor_content_1 = processed_tensor_content_1 - 2*1024

    processed_tensor_acoustic_1 = processed_tensor[:, 3::6]
    processed_tensor_acoustic_1 = processed_tensor_acoustic_1 - 3*1024

    processed_tensor_acoustic_2 = processed_tensor[:, 4::6]
    processed_tensor_acoustic_2 = processed_tensor_acoustic_2 - 4*1024

    processed_tensor_acoustic_3 = processed_tensor[:, 5::6]
    processed_tensor_acoustic_3 = processed_tensor_acoustic_3 - 5*1024
    stacked_tensor = torch.stack([processed_tensor_prosody, processed_tensor_content,processed_tensor_content_1, processed_tensor_acoustic_1,processed_tensor_acoustic_2, processed_tensor_acoustic_3, ], dim=0)
    test_0 = stacked_tensor[:, 0, :].unsqueeze(1)
    stacked_tensor = stacked_tensor.cpu()
    test_0 = test_0.cpu()


    decode_tensor(test_0)



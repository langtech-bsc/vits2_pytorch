## LJSpeech
import torch
import os
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import argparse


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def tts(model_path, config_path, texts, output_path):
    model_name = os.path.basename(model_path)
    hps = utils.get_hparams_from_file(config_path)

    if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder == True
    ):
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80  # vits2
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(model_path, net_g, None)


    for i,text in enumerate(texts):
        print(text)
        stn_tst = get_text(text, hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            audio = (
                net_g.infer(
                    x_tst, x_tst_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )

        write(data=audio, rate=hps.data.sampling_rate, filename=os.path.join(output_path, f"sample_{model_name}_{i}.wav"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model file.')
    parser.add_argument('--config_path', type=str, default=None, help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default=None, help='Path to output the files.')
    parser.add_argument('--text_file', type=str, default=None, help='Text file to synthesize')
    args = parser.parse_args()
    
    if args.text_file:
        with open(args.text_file, "r") as f:
            texts = f.read().splitlines()
    else:
        texts = [
        "El Tribunal Suprem espanyol (TS) ha sol·licitat a la fiscalia que informi sobre la competència i el contingut de l’exposició raonada on el jutge de l’Audiència espanyola (AN) Manuel García-Castellón demanava d’investigar el president Carles Puigdemont i Marta Rovira per delictes de terrorisme pel Tsunami Democràtic a la tardor del 2019 en reacció a la sentència del Primer d’Octubre.",
        "Vull recordar com a exemple la intervenció en el tram final del Barranc del Carraixet.",
        "Canvi de temps a la vista! La previsió meteorològica anuncia novetats importants per als propers dies i setmanes a Catalunya. ",
        "Fora d'això, era un jove encantador; i en cas de dubte, bastava preguntar-li-ho a sa mare.",
        "Un reproductor de vídeo integrat, extracció de CD amb un clic i suport millorat per als formats multimèdia.",
        "Va veure un Toyota aparcat al carrer i el va fotografiar des de tots els angles."
        ]

    tts(model_path=args.model_path,
        config_path=args.config_path,
        texts=texts,
        output_path=args.output_path)
 
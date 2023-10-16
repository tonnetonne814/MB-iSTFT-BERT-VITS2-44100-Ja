import glob
import os
from tqdm import tqdm
import random
import argparse
import torch, torchaudio
import subprocess

target_sr = 44100
hop_size = 512
song_min_s = 5000 # ms
split_ratio = 0.005
filelist_DIR = "./filelists/"
dataset_dir = "./dataset/"

os.makedirs(filelist_DIR ,exist_ok=True)
os.makedirs(dataset_dir ,exist_ok=True)

def process_one (wav_dir, 
                 txt_path, # = {/""/""/wav_filename.wav}|{sentence} 
                 results_folder, 
                 speaker_name, 
                 split_symbol = "|",
                 language = "JP", # ["ZH", "JP"]
                 silence_threshold=1e-2,
                 normalize=True):
    
    # Search wavfiles in {wav_dir}
    wav_lists = list()
    lists = glob.glob(f"{wav_dir}/**/*.wav", recursive=True)
    if len(lists) != 0:
        wav_lists.extend(lists)
    lists = glob.glob(f"{wav_dir}/*.wav", recursive=True)
    if len(lists) != 0:
        wav_lists.extend(lists)

    wav_list_new = list()
    for line in wav_lists:
        wav_list_new.append(str(line))
    wav_lists = wav_list_new

    # Read txt
    with open(txt_path, mode="r", encoding="utf-8") as f:
        lines = f.readlines()

    # Data PreProcess
    text_list = list()
    for wav_path in tqdm(wav_lists):
        path = wav_path.replace("\\", "/")
        path = path.replace("//", "/")
        filename = path.split("/")[-1]

        # checker
        sentence = ""
        target_filename = ""
        count = 0

        q = filename.replace(".wav", "") # ファイル名のテキストが
        for line in lines:               # ある行の文章を
            if q in line:                # 引っ張ってくる
                data = line.split(split_symbol)
                # sentence 
                if len(data) == 2:
                    sentence = data[1]
                elif len(data) > 2:
                    n = len(data)-1
                    sentence = str()
                    for idx in range(n):
                        sentence += data[idx+1]
                target_filename = filename
                # 重複確認用
                count += 1

        assert sentence != "" and target_filename != "" ,  (f".wavファイルとtextの記述が対応していない項目があります。text記述のファイル名と、実際に配置されているwavファイル名を確認してください。\nファイル:{wav_path}")
        assert count==1 ,  (f"textの記述に重複したファイル名があります。\nファイル:{wav_path}")

        if normalize is True:
            out_path = wav_path.replace(".wav", "_norm.wav")
            audio_norm_process(wav_path, out_path)
            wav_path = out_path

        # read wav
        y, sr = torchaudio.load(wav_path)

        # resampling (to 44100)
        y_converted = torchaudio.functional.resample(waveform=y, orig_freq=sr, new_freq=target_sr)

        # silence remover
        start = 0
        end = 0
        z_onoff = 0
        onoff = 0
        for idx in range(int(y_converted.size(1)/hop_size)):
            s = int(idx*hop_size)
            e = int((idx+1)*hop_size)
            value = torch.sum(torch.square(y_converted[:, s:e]))
            #print(value)
            
            # whether silent
            if value > silence_threshold:
                onoff = 1
            else:
                onoff = 0

            # start and end point
            if onoff==1 and z_onoff==0 and start==0:
                start = idx
            elif onoff==0 and z_onoff==1 and start!=0:
                end = idx
            
            z_onoff = onoff
        y_extract = y_converted[:, int(start*hop_size):int(end*hop_size)]
        if start == end:
            print("")
        save_path = os.path.join(results_folder, target_filename)
        save_path = save_path.replace("\\", "/")
        save_path = save_path.replace("//", "/")
        torchaudio.save(save_path, y_extract, target_sr) 

        if normalize is True:
            os.remove(wav_path)
        
        # {wav_path}|{speaker_name}|{language}|{text}
        out_txt = save_path.replace("\n","")+"|"+speaker_name+"|"+language+"|"+sentence.replace("\n","")+"\n"
        text_list.append(out_txt)

    return text_list

# MとFを両方DLして、ja0JPフォルダにfemaleとmaleを配置する
def preprocess(dataset_dir:str = "./hi-fi-captain/", 
               text_path_dir = "/path/to/.txt",
               results_folder="./dataset/hi-fi-captain/", 
               speaker_name = "name",
               language =  "JP", # ["ZH", "JP"]
               silence_threshold = 0.01,
               split_symbol = "|",
               normalize        :bool= False ):

    os.makedirs(results_folder, exist_ok=True)

    # txtフォルダパスであれば統合する
    if os.path.isfile(text_path_dir):
        in_txt_path = text_path_dir
    elif os.path.isdir(text_path_dir):
        in_txt_path = os.path.join(filelist_DIR, "merge.txt") 
        merge_txt = list()
        for filename in os.listdir(text_path_dir):
            path = os.path.join(text_path_dir, filename)
            with open(path, mode="r", encoding="utf-8") as f:
                lines = f.readlines()
            merge_txt.extend(lines)
        with open(in_txt_path, mode="w", encoding="utf-8") as f:
            f.writelines(merge_txt)

    text_list = list()
    text_list.extend( process_one (wav_dir = dataset_dir,
                                   txt_path = in_txt_path,
                                   results_folder=results_folder,
                                   speaker_name=speaker_name,
                                   split_symbol=split_symbol, 
                                   language=language,
                                   silence_threshold=silence_threshold,
                                   normalize=normalize))
    # train val test分割用
    #max_n = len(text_list)
    #test_list = list()
    #for _ in range(int(max_n * split_ratio)):
    #    n = len(text_list)
    #    idx = random.randint(9, int(n-1))
    #    txt = text_list.pop(idx)
    #    test_list.append(txt)

    #max_n = len(text_list)
    #val_list = list()
    #for _ in range(int(max_n * split_ratio)):
    #    n = len(text_list)
    #    idx = random.randint(9, int(n-1))
    #    txt = text_list.pop(idx)
    #    val_list.append(txt)

    add_write_txt(f"./filelists/transcription.txt", text_list)
    #add_write_txt(f"./filelists/train_{target_sr}.txt", text_list)
    #add_write_txt(f"./filelists/val_{target_sr}.txt",   val_list)
    #add_write_txt(f"./filelists/test_{target_sr}.txt",  test_list)

    return 0


def read_txt(path):
    with open(path, mode="r", encoding="utf-8")as f:
        lines = f.readlines()
    return lines

def write_txt(path, lines):
    with open(path, mode="w", encoding="utf-8")as f:
        f.writelines(lines)

def add_write_txt(path, lines):
    with open(path, mode="a", encoding="utf-8")as f:
        f.writelines(lines)

def audio_norm_process(in_path, out_path):
    if os.path.isfile(in_path) is True:
        pass
    else:
        print("[ERROR] File is not existed : ", in_path)
        exit()
    cmd = ["ffmpeg-normalize", in_path,   "-o",  out_path]
    subprocess.run(cmd, encoding='utf-8', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name',
                        type=str,
                        #required=True, 
                        default="jsut_basic5000",
                        help='単一話者のデータセットにすること。この名前は話者名にも流用される')
    
    parser.add_argument('--dataset_folder',
                        type=str,
                        #required=True, 
                        default="./data/basic5000",
                        help='データセットフォルダのパス。このパス以下にあるwavを全て読み込む')
    
    parser.add_argument('--dataset_language',
                        type=str,
                        #required=True, 
                        default="JP", #["JP" or "ZH"]
                        help='言語。基本的に日本語のみ。')
    
    parser.add_argument('--text_path',
                        type=str,
                        #required=True, 
                        default="./data/basic5000/transcript_utf8.txt",
                        help='Path to jvs corpus folder')
    
    parser.add_argument('--split_symbol',
                        type=str,
                        #required=True, 
                        default=":",
                        help='Path to jvs corpus folder')
    
    parser.add_argument('--silence_threshold',
                        type=float,
                        #required=True, 
                        default=0.01,
                        help='Path to jvs corpus folder')
    
    parser.add_argument('--normalize',
                        type=bool,
                        #required=True, 
                        default=False,
                        help='Path to jvs corpus folder')

    args = parser.parse_args()
    results_folder = os.path.join(dataset_dir, args.dataset_name)
    os.makedirs (results_folder,exist_ok=True)

    preprocess(dataset_dir=args.dataset_folder,
               text_path_dir=args.text_path,
               results_folder=results_folder,
               speaker_name=args.dataset_name,
               language=args.dataset_language,
               split_symbol=args.split_symbol,
               silence_threshold=args.silence_threshold,
               normalize=args.normalize)




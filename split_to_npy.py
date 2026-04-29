import numpy as np
from pathlib import Path
from tqdm import tqdm

def split_to_npy(input_path: Path, output_path: Path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    input_files = list(input_path.glob("*.npy"))[11:]
    # print(input_files)
    input_files_dict = {}
    for input_file in input_files:
        file_name = input_file.stem
        seq_name = file_name.split("_")[0]
        if seq_name not in input_files_dict:
            input_files_dict[seq_name] = []
        input_files_dict[seq_name].append(input_file)
    for seq_name, input_files in tqdm(input_files_dict.items(), total=len(input_files_dict)):
        # print(seq_name, input_files)
        datas = [np.load(input_file) for input_file in input_files]
        datas = np.concatenate(datas, axis=0)
        np.save(output_path / f"{seq_name}.npy", datas)
    # for input_file in input_files:
    #     data = np.load(input_file)
    #     for i in range(data.shape[0]):
    #         np.save(output_path / f"{input_file.stem}_{i}.npy", data[i])

if __name__ == "__main__":
    split_to_npy("/media/knocking/Seagate_Basic/calib_spike_250804_seqs/planes_split", "/media/knocking/Seagate_Basic/calib_spike_250804_seqs/planes")
'''
This file attempts to process raw pcap dataset like CrossPlatform (China android).
Split them by session flows and generate the dataset.
'''

import binascii
import math
from data_process import dataset_generation
import os
import tqdm
import scapy.all as scapy
import random
import json
import numpy as np
import csv
from sklearn.model_selection import StratifiedShuffleSplit

random.seed(40)

def list_packets(pcap_path):
    '''
    check the packet number for each label to see if it meets the sample number
    
    input:
        pcap_path: root path for all pcaps
    
    return:
        flow_number_dict: { label: number }. A dict to list the packet number for each label
    '''
    print("list_packets...")
    flow_number_dict = {}
    for parent, dirs, files in os.walk(pcap_path):
        for dir in dirs:
            for parent_, dirs_, files_ in os.walk(os.path.join(parent, dir)):
                if not flow_number_dict.get(dir):
                    flow_number_dict[dir] = 0
                flow_number_dict[dir] += len(files_)
        break
    print("list_packets done!")
    return flow_number_dict


def get_feature_flow(label_pcap, payload_len, header_to_remove, payload_pac=5):
    header_len = {
        'ETHERNET_II' : 14,
        'IP': 20,
        'PORT': 4
    }
    for h in header_to_remove:
        if h not in header_len.keys():
            raise Exception("wrong header to remove!")

    header_len_to_remove = sum([header_len[h] for h in header_to_remove])
    
    feature_data = []
    packets = scapy.rdpcap(label_pcap, count=payload_pac)
    flow_data_string = '' 

    for i in range(payload_pac):
        packet = packets[i]
        packet_data = packet.copy()
        data = (binascii.hexlify(bytes(packet_data)))
        # packet_string = data.decode()[2 * header_len_to_remove:]
        packet_string = data.decode()
        flow_data_string += dataset_generation.bigram_generation(packet_string, packet_len=payload_len, flag = True)
    feature_data.append(flow_data_string)

    return feature_data


def generate_dataset(pcap_path,
               samples,
               dataset_save_path,
               type,
               few_shot,
               header_to_remove):
    '''
    generate dataset for pcaps
    
    input:
        pcap_path: root path for all pcaps
        samples: a dict of sample numbers for every label
        dataset_save_path: the path to save datasets
        type: flow or packet
        few_shot: the ratio of the chosen training samples to the complete training set
        header_to_remove: choose the headers to remove
    '''
    if not isinstance(few_shot, list) and not isinstance(few_shot, tuple):
        print("few_shot should be a list or tuple")
        return
    
    if len(few_shot) == 0:
        print("few_shot should not be empty")
        return
    
    for r in few_shot:
        if not (r > 0 and r <= 1):
            print(f"wrong ratio in few_shot: {r}")
            return
    
    print("generate_dataset_json...")

    dataset = {}

    label_name_list = [] # label id -> label name
    label_id = {} # label name -> label id
    label_pcap_path = {} # directory path for each label

    for parent, dirs, files in os.walk(pcap_path):
        label_name_list.extend(dirs)
        for dir in dirs:
            label_pcap_path[dir] = os.path.join(parent, dir)
        break
    #breakpoint()
    label_name_list.sort()
    for idx, name in enumerate(label_name_list):
        label_id[name] = idx

    r_file_record = []
    print("\nBegin to generate features.")
    #breakpoint()

    for key in tqdm.tqdm(label_name_list, leave=False):
        # key is a label name
        if label_id[key] not in dataset:
            dataset[label_id[key]] = {
                "samples": 0,
                "payload": {}
            }

        label_count = key
        target_all_files = [os.path.join(x[0], y) for x in [(p, f) for p, d, f in os.walk(label_pcap_path[key])] for y in x[1]]
        #breakpoint()
        need = samples[label_count]
        available = len(target_all_files)
        if need > available:
            print(f"[Warning] label {label_count}: need {need}, but only {available} available. Using all available.")
            need = available

        r_files = random.sample(target_all_files, need)
        # r_files = random.sample(target_all_files, samples[key])
        for r_f in tqdm.tqdm(r_files, leave=False):
            feature_data = get_feature_flow(r_f, payload_len=64, header_to_remove=header_to_remove, payload_pac=5)
            if feature_data == -1:
                continue
            dataset[label_id[key]]["samples"] += 1
            if len(dataset[label_id[key]]["payload"].keys()) > 0:
                dataset[label_id[key]]["payload"][str(dataset[label_id[key]]["samples"])] = \
                    feature_data[0]
            else:
                dataset[label_id[key]]["payload"]["1"] = feature_data[0]
            r_file_record.append((label_id[key], dataset[label_id[key]]["samples"], r_f))

    all_data_number = 0
    for index in range(len(label_name_list)):
        print("%s\t%s\t%d" %
              (label_id[label_name_list[index]], label_name_list[index],
               dataset[label_id[label_name_list[index]]]["samples"]))
        all_data_number += dataset[label_id[label_name_list[index]]]["samples"]
    print("all\t%d" % (all_data_number))

    info_save_path = os.path.join(dataset_save_path, 'dataset-'+type)
    if not os.path.exists(info_save_path):
        os.makedirs(info_save_path)
        print(f"create dir: {info_save_path}")
    with open(os.path.join(info_save_path, "chosen_files.txt"), "w", encoding='utf-8') as p_f:
        for id, sample_id, file in r_file_record:
            p_f.write(str(id)+"\t"+str(sample_id)+"\t"+file)
            p_f.write("\n")
    with open(os.path.join(info_save_path, "label_info.txt"),
              "w",
              encoding='utf-8') as p_f:
        p_f.write('idx' + '\t' + 'name' + '\t' + 'number' + '\n')
        for idx, (name, num) in enumerate(samples.items()):
            p_f.write(str(idx) + '\t' + name + '\t' + str(num) + '\n')
        p_f.write(f"\nall\t{all_data_number}\n")
    with open(os.path.join(info_save_path, "dataset.json"), "w") as f:
        json.dump(dataset, fp=f, ensure_ascii=False, indent=4)
    
    
    X,Y = dataset_generation.obtain_data(
        pcap_path=None, samples=[samples[label_name_list[id]] for id in dataset.keys()], features=["payload"], dataset_save_path=None, json_data=dataset
    )
    
    print("generate npy...")
    X_payload= []
    Y_all = []
    for app_label in Y:
        for label in app_label:
            Y_all.append(int(label))

    for index_label in range(len(X[0])):
        for index_sample in range(len(X[0][index_label])):
            X_payload.append(X[0][index_label][index_sample])

    split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=41) 
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42) 

    x_payload = np.array(X_payload)
    dataset_label = np.array(Y_all)

    x_payload_train = []
    y_train = []

    x_payload_valid = []
    y_valid = []

    x_payload_test = []
    y_test = []

    for train_index, test_index in split_1.split(x_payload, dataset_label):
        x_payload_train, y_train = \
            x_payload[train_index], \
            dataset_label[train_index]
        x_payload_test,y_test = \
            x_payload[test_index], \
            dataset_label[test_index]
    for test_index, valid_index in split_2.split(x_payload_test, y_test):
        x_payload_valid, y_valid = \
            x_payload_test[valid_index], y_test[valid_index]
        x_payload_test, y_test = \
            x_payload_test[test_index], y_test[test_index]

    npy_path = os.path.join(dataset_save_path, 'npy', type)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    train_len = len(x_payload_train)
    for ratio in few_shot:
        np.save(os.path.join(npy_path, f'x_datagram_train_{ratio}.npy'), x_payload_train[:math.floor(train_len * ratio)])
    np.save(os.path.join(os.path.join(npy_path, 'x_datagram_test.npy')), x_payload_test)
    np.save(os.path.join(os.path.join(npy_path, 'x_datagram_valid.npy')), x_payload_valid)

    for ratio in few_shot:
        np.save(os.path.join(npy_path, f'y_train_{ratio}.npy'), y_train[:math.floor(train_len * ratio)])
    np.save(os.path.join(os.path.join(npy_path, 'y_test.npy')), y_test)
    np.save(os.path.join(os.path.join(npy_path, 'y_valid.npy')), y_valid)

    tsv_path = os.path.join(dataset_save_path, 'tsv', type)
    if not os.path.exists(tsv_path):
        os.makedirs(tsv_path)
    
    def write_dataset_tsv(data,label,file_dir,type):
        dataset_file = [["label", "text_a"]]
        for index in range(len(label)):
            dataset_file.append([label[index], data[index]])
        with open(os.path.join(file_dir, type + "_dataset.tsv"), 'w',newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerows(dataset_file)
        return 0
    
    print("Begin to write tsv...")
    
    save_dir = tsv_path
    for ratio in few_shot:
        write_dataset_tsv(x_payload_train[:math.floor(train_len * ratio)], y_train[:math.floor(train_len * ratio)], save_dir, f"train_{ratio}")
    write_dataset_tsv(x_payload_test, y_test, save_dir, "test")
    write_dataset_tsv(x_payload_valid, y_valid, save_dir, "valid")
    
    print("finish generating pre-train's datagram dataset.\nPlease check in %s" % save_dir)


if __name__ == '__main__':
    
    few_shot = (0.1, 0.5, 1.0)
    header_to_remove = ('ETHERNET_II', 'IP', 'PORT')
    
    # # CIC-IoT
    # pcap_path = "/home/spa/traffic_cls/resources/CIC-IoT-pcap-flow/"
    # dataset_save_path = "/home/spa/traffic_cls/resources/CIC-IoT/"
    
    # # ISCX-Tor-service
    # pcap_path = "/home/spa/traffic_cls/resources/ISCX-Tor-pcap-flow/"
    # dataset_save_path = "/home/spa/traffic_cls/resources/ISCX-Tor/"
    
    # ISCX-Tor-app
    pcap_path = "/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/CSTNET-TLS1.3/pcap/"
    dataset_save_path = "/3241903007/workstation/AnomalyTrafficDetection/ET-BERT/datasets/own_lyj/CSTNET/tt/"
    
    MAX_SAMPLE = 500
    samples = {} # the number of actual picked samples for each label in the dataset
    label_num = None # the number of labels in this dataset

    packet_info = list_packets(pcap_path)
    for label, num in packet_info.items():
        samples[label] = min(MAX_SAMPLE, num)
    print("sample number for each label:")
    print(json.dumps(samples, indent=4))
    
    # genearte datatet
    #breakpoint()
    generate_dataset(pcap_path, samples, dataset_save_path, type='flow', few_shot=few_shot, header_to_remove=header_to_remove)

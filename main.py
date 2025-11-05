"""
This script provides an exmaple to wrap UER-py for classification.
改动记录：
 - read_dataset 返回分开的统计特征（length_ids list, time_ids list, direction list）
 - main 中加载 vocab，将 train/dev/test 读取为包含统计特征的 dataset
 - 将统计特征分别转为 Tensor（length_idx, time_idx, direction_vec）
"""
import pickle
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        # self.length_emb = nn.Embedding(args.len_vocab_size, args.hidden_size)
        # self.time_emb = nn.Embedding(args.iat_vocab_size, args.hidden_size)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha

        self.hidden_size = args.hidden_size

        # === 各模态特征映射 ===
        self.payload_fc = nn.Linear(self.hidden_size, 512)
        self.stat_cnn = nn.Conv2d(3, 1, kernel_size=(40, 1))  # 3×40×768 → 1×1×768
        self.stat_fc = nn.Linear(self.hidden_size, 512)

        # === 注意力融合 ===
        self.attention_fc = nn.Linear(512 * 2, 1)  # 输入 [payload; stat] 输出注意力权重 α

        # === 分类层（增强版） ===
        self.classifier = nn.Sequential(
            nn.Linear(512, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 2, self.labels_num)
        )

    def forward(self, src, tgt, seg, soft_tgt=None, stat_tensor=None):
        # ===== 模态开关 =====
        mode = getattr(self, "ablation_mode", "full")

        # ===== Payload 模态 =====
        if mode in ["full", "payload"]:
            emb = self.embedding(src, seg)
            enc_out = self.encoder(emb, seg)

            if self.pooling == "mean":
                payload_vec = torch.mean(enc_out, dim=1)
            elif self.pooling == "max":
                payload_vec = torch.max(enc_out, dim=1)[0]
            elif self.pooling == "last":
                payload_vec = enc_out[:, -1, :]
            else:
                payload_vec = enc_out[:, 0, :]

            payload_vec = torch.tanh(self.payload_fc(payload_vec))  # [B, 512]
        else:
            payload_vec = torch.zeros((src.size(0), 512), device=src.device)

        # ===== Stat 模态 =====
        if mode in ["full", "stat"] and stat_tensor is not None:
            stat_out = self.stat_cnn(stat_tensor).squeeze(2).squeeze(1)  # [B, 768]
            stat_vec = torch.tanh(self.stat_fc(stat_out))  # [B, 512]
        else:
            stat_vec = torch.zeros_like(payload_vec)

        # ===== 模态融合 =====
        if mode == "payload":
            fusion_vec = payload_vec
        elif mode == "stat":
            fusion_vec = stat_vec
        else:  # full
            fusion_input = torch.cat([payload_vec, stat_vec], dim=1)
            alpha = torch.sigmoid(self.attention_fc(fusion_input))
            fusion_vec = alpha * payload_vec + (1 - alpha) * stat_vec
            # 多种情况concat

        # ===== 分类 =====
        logits = self.classifier(fusion_vec)
        # import pdb; pdb.set_trace()
        # print(logits.shape)
        # ===== Loss =====
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                    (1 - self.soft_alpha) * nn.NLLLoss()(F.log_softmax(logits, dim=-1), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(F.log_softmax(logits, dim=-1), tgt.view(-1))
            return loss, logits
        else:
            return None, logits



def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters_with_path(model, model_path=None):
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'), strict=False)
    else:
        for n, p in model.named_parameters():
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size: (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    # leftover
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size:, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


import numpy as np
import torch

def read_dataset(args, path):
    """
    返回 dataset，每个 entry 是：
    - (src_ids, tgt, seg, lengths, iats, directions)
      或
    - (src_ids, tgt, seg, lengths, iats, directions, soft_tgt) 如果 args.soft_targets=True 且 TSV 中有 logits

    注意：
    lengths, iats, directions 是原始 list，训练时再转为 embedding。
    """
    dataset, columns = [], {}

    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue

            line = line.strip().split("\t")
            tgt = int(line[columns["label"]])

            # 解析 soft targets
            if args.soft_targets and "logits" in columns:
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]

            # payload -> token ids
            text_a = line[columns["text_a"]]
            try:
                flow_dict = json.loads(text_a.replace("'", "\""))
            except Exception as e:
                print(f"[WARN] JSON parse error at line {line_id}: {e}")
                continue

            payload = flow_dict.get('payload', "")
            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(payload))
            seg = [1] * len(src)
            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)

            # 只保存原始统计特征
            packet_num = getattr(args, "packet_num", 40)
            lengths = flow_dict.get('length', [])
            if len(lengths) > packet_num:
                lengths = lengths[:packet_num]
            else:
                lengths = lengths + [0] * (packet_num - len(lengths))

            iats = flow_dict.get('time', [])
            if len(iats) > packet_num:
                iats = iats[:packet_num]
            else:
                iats = iats + [0] * (packet_num - len(iats))

            directions = flow_dict.get('direction', [])
            if len(directions) > packet_num:
                directions = directions[:packet_num]
            else:
                directions = directions + [0] * (packet_num - len(directions))

            # 最终 append
            entry = (src, tgt, seg, lengths, iats, directions)
            if args.soft_targets and "logits" in columns:
                entry += (soft_tgt,)
            dataset.append(entry)

    return dataset 

import numpy as np
import torch

def build_stat_features(trainset, len_dict, iat_dict, len_emb, iat_emb, hidden_size, packet_num=40):
    """
    构建统计特征 embedding，如果遇到新特征则动态扩充词典和 embedding。
    
    Args:
        trainset: list，每个元素包含 src, tgt, seg, lengths, iats, directions
        len_dict: 长度词典 {length_value: index}
        iat_dict: 时间间隔词典 {iat_value: index}
        len_emb: np.array, 现有长度 embedding [词典大小, hidden_size]
        iat_emb: np.array, 现有时间 embedding [词典大小, hidden_size]
        hidden_size: embedding 维度
        packet_num: 每条 flow 取多少 packet
    Returns:
        stat_tensor: [N, 3, packet_num, hidden_size] 的 Tensor
        len_emb, iat_emb: 可能扩充后的 embedding
    """
    length_vec_list, time_vec_list, direction_vec_list = [], [], []

    for ex in trainset:
        l_seq = ex[3][:packet_num] + [0]*(packet_num - len(ex[3]))
        t_seq = ex[4][:packet_num] + [0]*(packet_num - len(ex[4]))
        d_seq = ex[5][:packet_num] + [0]*(packet_num - len(ex[5]))

        # 查 embedding，如果查不到就新增
        length_emb_seq = []
        for l in l_seq:
            if l not in len_dict:
                new_idx = len(len_dict)
                len_dict[l] = new_idx
                new_emb = np.random.normal(0, 0.02, size=(hidden_size,))
                len_emb = np.vstack([len_emb, new_emb])
            length_emb_seq.append(len_emb[len_dict[l]])
        length_emb_seq = np.array(length_emb_seq, dtype=np.float32)

        time_emb_seq = []
        for t in t_seq:
            if t not in iat_dict:
                new_idx = len(iat_dict)
                iat_dict[t] = new_idx
                new_emb = np.random.normal(0, 0.02, size=(hidden_size,))
                iat_emb = np.vstack([iat_emb, new_emb])
            time_emb_seq.append(iat_emb[iat_dict[t]])
        time_emb_seq = np.array(time_emb_seq, dtype=np.float32)

        direction_emb = np.array(
            [[1.0]*hidden_size if d==1 else [-1.0]*hidden_size if d==-1 else [0.0]*hidden_size for d in d_seq],
            dtype=np.float32
        )

        length_vec_list.append(length_emb_seq)
        time_vec_list.append(time_emb_seq)
        direction_vec_list.append(direction_emb)

    length_tensor = torch.FloatTensor(np.stack(length_vec_list))    # [N, packet_num, hidden_size]
    time_tensor   = torch.FloatTensor(np.stack(time_vec_list))      # [N, packet_num, hidden_size]
    direction_tensor = torch.FloatTensor(np.stack(direction_vec_list)) # [N, packet_num, hidden_size]

    # 堆叠成 [N, 3, packet_num, hidden_size]
    stat_tensor = torch.stack([length_tensor, time_tensor, direction_tensor], dim=1)

    return stat_tensor, len_emb, iat_emb

def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None, stat_tensor=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, stat_features, print_confusion_matrix=False):
    """
    Evaluate model on given dataset and optional statistical features.
    Outputs confusion matrix and precision/recall/f1 with actual label names in a well-aligned table.

    Args:
        args: 训练/模型参数
        dataset: list of tuples (src, tgt, seg, ...)
        stat_features: [N, 3, packet_num, hidden_size] 的 Tensor
        print_confusion_matrix: 是否打印并保存混淆矩阵
    Returns:
        accuracy: float
        confusion: Tensor [labels_num, labels_num]
    """
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size
    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    args.model.eval()

    # 遍历 batch
    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        stat_batch = stat_features[i*batch_size:(i+1)*batch_size].to(args.device)

        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch, stat_tensor=stat_batch)

        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch

        # 更新混淆矩阵
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    # Label 映射字典
    # label_map = {
    #     0: "Chat",
    #     1: "Email",
    #     2: "FileTransfer",
    #     3: "P2P",
    #     4: "Streaming",
    #     5: "VoIP"
    # }
    # label_map = {
    # 0: "Geodo",
    # 1: "Neris",
    # 2: "FTP",
    # 3: "Cridex",
    # 4: "WorldOfWarcraft",
    # 5: "Miuref",
    # 6: "Virut",
    # 7: "Weibo",
    # 8: "Htbot",
    # 9: "SMB",
    # 10: "Skype",
    # 11: "Shifu",
    # 12: "Zeus",
    # 13: "BitTorrent",
    # 14: "Gmail",
    # 15: "MySQL",
    # 16: "Tinba",
    # 17: "Nsis-ay",
    # 18: "Outlook",
    # 19: "Facetime"
    # }

    # 构建 label 名称列表，按 confusion matrix 的索引顺序（0~labels_num-1）
    # label_names = [label_map[i] for i in range(args.labels_num)]
    # 自动生成标签名称（不使用映射字典）
    label_names = [f"Label_{i}" for i in range(args.labels_num)]
    if print_confusion_matrix:
        print("Confusion matrix (rows=predicted, cols=true):")

        # 打印表头
        col_width = max(len(name) for name in label_names) + 2
        header = " " * (col_width) + "".join(name.ljust(col_width) for name in label_names)
        print(header)

        # 打印每行
        for i, row in enumerate(confusion):
            row_str = "".join(str(val).ljust(col_width) for val in row.tolist())
            print(f"{label_names[i].ljust(col_width)}{row_str}")

        # 保存到文件
        cf_array = confusion.numpy()
        out_path = "/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/CSTNET-TLS1.3/re/payload/confusion_matrix.tsv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(header + "\n")
            for i, cf_a in enumerate(cf_array):
                row_str = "".join(str(val).ljust(col_width) for val in cf_a)
                f.write(f"{label_names[i].ljust(col_width)}{row_str}\n")

        # 打印 precision, recall, f1
        print("\nReport precision, recall, and f1:")
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
            print(f"Label {label_names[i].ljust(col_width)} Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")

    accuracy = correct / len(dataset)
    print(f"Acc. (Correct/Total): {accuracy:.4f} ({correct}/{len(dataset)})")
    return accuracy, confusion


def main():
    # Debug breakpoint if needed
    # breakpoint()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)
    print("开始加载数据集")
    # Prevent argparse from erroring when required args aren't provided by supplying defaults.
    parser.set_defaults(
        train_path="/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/CSTNET-TLS1.3/tsv/flow/train_dataset.tsv",
        dev_path="/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/CSTNET-TLS1.3/tsv/flow/valid_dataset.tsv",
        test_path="/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/CSTNET-TLS1.3/tsv/flow/test_dataset.tsv",
        vocab_path="/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/models/encryptd_vocab.txt"
    )
    parser.add_argument("--ablation_mode", choices=["full", "payload", "stat"], default="full",
                    help="Specify ablation mode: 'full' for both, 'payload' for payload-only, 'stat' for stat-only.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer.")
    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")

    args = parser.parse_args()
    args.output_model_path = "/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/models/CSTNET-TLS/CSTNETFlowPayload.bin"
    model_path = "/3241903007/workstation/AnomalyTrafficDetection/ET-BERT/models/pre-trained_model.bin"

    # Load hyperparams from config and seed.
    args = load_hyperparam(args)
    set_seed(args.seed)

    # Count labels
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build model
    model = Classifier(args)
    print(model)
    model.ablation_mode = args.ablation_mode
    load_or_initialize_parameters_with_path(model, model_path=model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # load vocab dicts for length / time mapping
    with open("/3241903007/workstation/AnomalyTrafficDetection/FlowVocab/dataset/AttributeValueDictionary/len_dict.pkl", "rb") as f:
        length_vocab = pickle.load(f)
    with open("/3241903007/workstation/AnomalyTrafficDetection/FlowVocab/dataset/AttributeValueDictionary/iat_dict.pkl", "rb") as f:
        time_vocab = pickle.load(f)

    len_embedding = np.load("/3241903007/workstation/AnomalyTrafficDetection/FlowVocab/dataset/AttributeValueDictionary/len_embedding.npy")
    iat_embedding = np.load("/3241903007/workstation/AnomalyTrafficDetection/FlowVocab/dataset/AttributeValueDictionary/iat_embedding.npy")
    # ---- 检查 length embedding ----
    if len(length_vocab) > len_embedding.shape[0]:
        diff = len(length_vocab) - len_embedding.shape[0]
        print(f"[WARN] len_dict 比 len_embedding 多 {diff} 个词，自动扩充。")
        new_emb = np.random.normal(0, 0.02, size=(diff, len_embedding.shape[1]))
        len_embedding = np.vstack([len_embedding, new_emb])
    elif len(length_vocab) < len_embedding.shape[0]:
        diff = len_embedding.shape[0] - len(length_vocab)
        print(f"[WARN] len_embedding 比 len_dict 多 {diff} 行，将截断。")
        len_embedding = len_embedding[:len(length_vocab)]

    # ---- 检查 iat embedding ----
    if len(time_vocab) > iat_embedding.shape[0]:
        diff = len(time_vocab) - iat_embedding.shape[0]
        print(f"[WARN] iat_dict 比 iat_embedding 多 {diff} 个词，自动扩充。")
        new_emb = np.random.normal(0, 0.02, size=(diff, iat_embedding.shape[1]))
        iat_embedding = np.vstack([iat_embedding, new_emb])
    elif len(time_vocab) < iat_embedding.shape[0]:
        diff = iat_embedding.shape[0] - len(time_vocab)
        print(f"[WARN] iat_embedding 比 iat_dict 多 {diff} 行，将截断。")
        iat_embedding = iat_embedding[:len(time_vocab)]
    # Read datasets (payload + stat features)
    # breakpoint()
    trainset = read_dataset(args, args.train_path)
    devset = read_dataset(args, args.dev_path)
    testset = read_dataset(args, args.test_path) if args.test_path else None
    print("The number of training instances:", len(trainset))
    print("The number of dev instances:", len(devset))
    if testset is not None:
        print("The number of test instances:", len(testset))
    # Convert to tensors for training (payload tensors + stat features separate)
    # breakpoint()

    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    # payload / labels for model (these are what current classifier uses)
    src = torch.LongTensor([ex[0] for ex in trainset])
    tgt = torch.LongTensor([ex[1] for ex in trainset])
    seg = torch.LongTensor([ex[2] for ex in trainset])

    # stat features -> lists -> tensors
    # length_idx: LongTensor [N, packet_num]
    # stat_features = build_stat_features(trainset, length_vocab, time_vocab, len_embedding, iat_embedding, args.hidden_size, packet_num=40)
    stat_features, len_embedding, iat_embedding = build_stat_features(
        trainset, length_vocab, time_vocab, len_embedding, iat_embedding, args.hidden_size, packet_num=40
)
    # devset stat tensor
    dev_stat_features, _, _ = build_stat_features(
        devset, length_vocab, time_vocab, len_embedding, iat_embedding, args.hidden_size, packet_num=40
    )

    # testset stat tensor（如果有的话）
    if testset is not None:
        test_stat_features, _, _ = build_stat_features(
            testset, length_vocab, time_vocab, len_embedding, iat_embedding, args.hidden_size, packet_num=40
        )
        # 保存扩充后的 embedding 和词典
    with open("/3241903007/workstation/AnomalyTrafficDetection/FlowVocab/dataset/AttributeValueDictionary/len_dict.pkl", "wb") as f:
        pickle.dump(length_vocab, f)
    np.save("/3241903007/workstation/AnomalyTrafficDetection/FlowVocab/dataset/AttributeValueDictionary/len_embedding.npy", len_embedding)

    with open("/3241903007/workstation/AnomalyTrafficDetection/FlowVocab/dataset/AttributeValueDictionary/iat_dict.pkl", "wb") as f:
        pickle.dump(time_vocab, f)
    np.save("/3241903007/workstation/AnomalyTrafficDetection/FlowVocab/dataset/AttributeValueDictionary/iat_embedding.npy", iat_embedding)
    # direction: expand to 768 dims per packet if you want
    print(stat_features.size())  # [N, 3, packet_num, 768]
    # breakpoint()
    # soft targets optional placeholder (if present in your TSV)
    if args.soft_targets:
        # NOTE: read_dataset currently does not parse logits field; adjust read_dataset if using soft targets.
        soft_tgt = None
    else:
        soft_tgt = None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1
    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)
    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))

    args.model = model

    # print("Start training.")
    total_loss, result, best_result = 0.0, 0.0, 0.0

    print("Start training.")


    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
            # loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            stat_batch = stat_features[i*batch_size:(i+1)*batch_size].to(args.device)
            loss = train_model(args, model, optimizer, scheduler,
                   src_batch, tgt_batch, seg_batch,
                   soft_tgt_batch, stat_tensor=stat_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        # evaluate on devset (we pre-read devset as list)
        # result = evaluate(args, devset)
        result = evaluate(args, devset, dev_stat_features)
        if result[0] > best_result:
            best_result = result[0]
            save_model(model, args.output_model_path)

    # final test
    if testset is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        # evaluate(args, testset, True)
        evaluate(args, testset, test_stat_features, True)


if __name__ == "__main__":
    main()

# # 全模态（payload + stat）
# python /3241903007/workstation/AnomalyDetection/concat/classfiy-all.py --ablation_mode full

# # 仅 payload
# python /3241903007/workstation/AnomalyDetection/concat/classfiy-all.py --ablation_mode payload

# # 仅 stat
# python /3241903007/workstation/AnomalyDetection/concat/classfiy-all.py --ablation_mode stat

'''
python main.py --dataset_path "benchmarks/boolq/dev2.jsonl" --log_file_path "results_with_qwen_logprobs.json"

此代码用于加载数据集、通过 OpenAI API 计算 token 序列的 logprobs，并通过统计检验比较原始顺序和打乱顺序的 logprobs。
'''

# 导入所需的库和模块
import os
import math
import random 
import numpy as np  # 数组和数值计算库
from scipy.stats import t as tdist  # 导入t分布用于统计检验
from multiprocessing import Process, Queue  # 用于并行处理
from tqdm import tqdm  # 用于显示进度条
import json  # 处理 JSON 数据
import fire  # 用于命令行接口
from openai import OpenAI  # 导入OpenAI库用于调用GPT模型

# 设置API key和API的基础URL，用于调用 OpenAI 接口
API_KEY = ""  # 替换为你的 API key
BASE_URL = ""  # 替换为API的基本URL

# 创建OpenAI客户端，用于后续调用API
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 定义两个 lambda 函数：用于展平嵌套列表和打乱列表
flatten = lambda l : [x for s in l for x in s]  # 展平嵌套列表
shuffle = lambda l : random.sample(l, k=len(l))  # 打乱列表

def load_dataset(dataset_path):
    # 加载数据集函数
    if dataset_path.endswith(".json"):
        # 如果是JSON文件，读取内容
        print("loading from json...")
        with open(dataset_path, "r") as f:
            data = f.read()
            examples = json.loads(data)  # 将JSON格式数据解析为Python对象
            return examples

    # 如果不是JSON，逐行读取文件
    with open(dataset_path, "r") as f:
        lines = f.readlines()  # 读取所有行
    return lines  # 返回行列表

def compute_logprob_of_token_sequence(tokens, context_len=2048, device=0):
    """
    调用 OpenAI API 计算一系列 token 的对数概率 (logprobs)。
    """
    # 将token列表合并成一个输入字符串
    input_text = " ".join(tokens)

    try:
        # 使用 GPT 模型调用API并请求返回logprobs
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": input_text}],
            model="gpt-3.5-turbo",  # 使用GPT-3.5模型
            logprobs=True  # 请求返回 logprobs
        )
        
        # 从响应中提取 logprobs
        logprobs = [token_logprob.logprob for token_logprob in response.choices[0].logprobs.content]
        
        # 计算并返回所有 token 的 logprobs 的和
        total_logprob = sum(logprobs)

        return total_logprob  # 返回 logprobs 总和

    except Exception as e:
        # 如果发生错误，打印错误信息
        print(f"An error occurred: {e}")
        return None  # 返回 None 以表示失败

def worker(context_len, device, main_queue, worker_queue):
    # 工作进程，用于处理多个并行任务
    while True:
        # 从 worker_queue 获取 token 列表、shard ID 和是否是 canonical（原始顺序）
        tokens, shard_id, is_canonical = worker_queue.get()

        if tokens == None:  # 如果收到 None，表示退出
            break

        # 计算 token 序列的 logprobs
        logprob = compute_logprob_of_token_sequence(tokens, context_len, device=device)

        # 将结果放入主进程的队列
        main_queue.put((logprob, shard_id, is_canonical))

def main(dataset_path, context_len=2048, num_shards=5, permutations_per_shard=25,
         random_seed=0, log_file_path=None, max_examples=5000):

    # 设置随机种子，保证可重复性
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 加载数据集
    examples = load_dataset(dataset_path)
    examples = examples[:max_examples]  # 限制加载的示例数量
    num_examples = len(examples)  # 获取数据集大小
    print(f"Loaded {num_examples} examples from {dataset_path}")

    # 对示例进行简单的基于空格的分词
    tokenized_examples = [ex.split() for ex in examples]

    # 使用多进程处理请求（在本例中仅使用一个工作进程）
    processes = []
    main_queue = Queue()  # 主进程队列，用于收集工作进程的结果
    worker_queues = [Queue() for _ in range(1)]  # 工作进程队列

    # 启动工作进程
    p = Process(target=worker, args=(context_len, 0, main_queue, worker_queues[0]))
    processes.append(p)
    p.start()

    # 计算每个分片的大小（将数据集分为多个分片）
    shard_counts = [(x + 1 if i < num_examples % num_shards else x) 
                    for i, x in enumerate([num_examples // num_shards] * num_shards)]
    shard_counts = np.asarray(shard_counts)

    # 生成每个分片的索引
    shard_example_indices = [0] + np.cumsum(shard_counts).tolist()
    for i, (start, end) in enumerate(zip(shard_example_indices, shard_example_indices[1:])):
        shard = tokenized_examples[start:end]

        # 将原始顺序的logprobs请求提交到worker队列
        worker_queues[0].put((
            flatten(shard),  # 展平后的token列表
            i,               # 分片ID
            True))           # 标识这是canonical（原始顺序）

        # 将打乱顺序的logprobs请求提交到worker队列
        for j in range(permutations_per_shard):
            worker_queues[0].put((
                flatten(shuffle(shard)),  # 打乱后的token列表
                i,                        # 分片ID
                False))                   # 标识这是打乱顺序

    # 等待所有请求完成，并显示进度条
    total_work = num_shards * (1 + permutations_per_shard)
    pbar = tqdm(total=total_work)

    canonical_logprobs = [None for _ in range(num_shards)]  # 存储每个分片的 canonical logprobs
    shuffled_logprobs  = [[] for _ in range(num_shards)]    # 存储每个分片的打乱顺序 logprobs

    # 处理worker进程返回的结果
    completed = 0
    while completed < total_work:
        logprob, shard_id, is_canonical = main_queue.get()

        if is_canonical:
            canonical_logprobs[shard_id] = logprob  # 存储原始顺序的logprobs
        else:
            shuffled_logprobs[shard_id].append(logprob)  # 存储打乱顺序的logprobs

        pbar.update(1)  # 更新进度条
        completed += 1

    # 终止工作进程
    worker_queues[0].put((None, None, None))  # 向worker发送退出信号

    for p in processes:
        p.join()  # 等待所有worker进程结束

    # 计算 p-value（p值，用于统计显著性检验）
    canonical_logprobs = np.asarray(canonical_logprobs)  # 转换为numpy数组
    shuffled_logprobs  = np.asarray(shuffled_logprobs)

    # 进行 t 检验，计算 canonical 和 shuffled 之间的差异
    diffs = canonical_logprobs - shuffled_logprobs.mean(axis=1)
    z = np.mean(diffs) / np.std(diffs) * np.sqrt(len(diffs))
    pval = 1 - tdist.cdf(z, df=len(diffs)-1)  # 计算 p 值
    print(f"{pval=}")

    # 将结果写入日志文件（如果指定了log_file_path）
    if log_file_path is not None:
        print(f"Writing logprobs to: {log_file_path}")
        with open(f"{log_file_path}", 'w') as f:
            f.write(json.dumps({
                'pval': pval,
                'permutations_per_shard': permutations_per_shard,
                'num_shards': num_shards,
                'canonical_logprobs': canonical_logprobs.tolist(),
                'shuffled_logprobs': shuffled_logprobs.tolist(),
            }))

if __name__ == '__main__':
  # 使用Fire库，将命令行参数解析并传递给main函数
  fire.Fire(main)

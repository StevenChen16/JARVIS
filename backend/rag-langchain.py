# %%
!pip install langchain-community --quiet
!pip install langchain-huggingface --quiet
!pip install einops --quiet
!pip install faiss-cpu --quiet

# %%
import os
import torch
from langchain_community.document_loaders import *
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import joblib
import time

# %%
# CUDA设备
CUDA_device = 'cuda:0,1'

# 文件路径和模型路径
Embedding_Model = 'intfloat/multilingual-e5-large-instruct'
LLM_Model = '/kaggle/input/llama-3.1/transformers/8b-instruct/2'
file_paths = ['/kaggle/input/20230916test/me.txt', "/kaggle/input/20230916test/2024-Wealth-Outlook-MidYear-Edition.pdf", "/kaggle/input/20230916test/elon-musk-tesla-spacex-and-the-quest-for-a-fantastic-future-0062469673-9780062469670_compress.pdf"]
store_path = '/kaggle/working/me.faiss'

# 定义文件扩展名和加载器类的映射关系
LOADER_MAPPING = {
    '.pdf': PyPDFLoader,
    '.txt': TextLoader,
    '.md': UnstructuredMarkdownLoader,
    '.csv': CSVLoader,
    '.jpg': UnstructuredImageLoader,
    '.jpeg': UnstructuredImageLoader,
    '.png': UnstructuredImageLoader,
    '.json': JSONLoader,
    '.html': BSHTMLLoader,
    '.htm': BSHTMLLoader
}

def load_single_file(file_path):
    """
    加载单个文件的内容。
    参数:
        file_path (str): 文件路径
    返回:
        docs (list): 文档内容列表
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    loader_class = LOADER_MAPPING.get(ext)
    if not loader_class:
        print(f"不支持的文件类型: {ext}")
        return None
    loader = loader_class(file_path)
    docs = list(loader.lazy_load())
    return docs

def load_files(file_paths: list):
    """
    批量加载多个文件的内容。
    参数:
        file_paths (list): 文件路径列表
    返回:
        docs (list): 所有加载的文档内容
    """
    if not file_paths:
        return []
    
    docs = []
    for file_path in tqdm(file_paths):
        print("加载文档:", file_path)
        loaded_docs = load_single_file(file_path)
        if loaded_docs:
            docs.extend(loaded_docs)
    return docs

def split_text(txt, chunk_size=200, overlap=20):
    """
    将文本分割为多个小段，以便进行进一步处理。
    参数:
        txt (str): 文本内容
        chunk_size (int): 每个段落的字符数
        overlap (int): 重叠字符数
    返回:
        docs (list): 分割后的文档段落列表
    """
    if not txt:
        return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = splitter.split_documents(txt)
    return docs

def create_embedding_model(model_file):
    """
    创建嵌入模型以生成文本的嵌入表示。
    参数:
        model_file (str): 嵌入模型路径
    返回:
        embedding (HuggingFaceEmbeddings): 嵌入模型实例
    """
    embedding = HuggingFaceEmbeddings(model_name=model_file, model_kwargs={'trust_remote_code': True})
    return embedding

def save_file_paths(store_path, file_paths):
    """
    将文件路径保存到本地。
    参数:
        store_path (str): 存储路径
        file_paths (list): 文件路径列表
    """
    joblib.dump(file_paths, f'{store_path}/file_paths.pkl')

def load_file_paths(store_path):
    """
    加载已保存的文件路径。
    参数:
        store_path (str): 存储路径
    返回:
        list: 加载的文件路径列表
    """
    file_paths_file = f'{store_path}/file_paths.pkl'
    if os.path.exists(file_paths_file):
        return joblib.load(file_paths_file)
    return None

def file_paths_match(store_path, file_paths):
    """
    检查当前文件路径列表是否与存储路径中的文件路径匹配。
    参数:
        store_path (str): 存储路径
        file_paths (list): 当前文件路径列表
    返回:
        bool: 是否匹配
    """
    saved_file_paths = load_file_paths(store_path)
    return saved_file_paths == file_paths

def create_vector_store(docs, store_file, embeddings):
    """
    创建向量存储并保存到本地。
    参数:
        docs (list): 文档列表
        store_file (str): 存储文件路径
        embeddings (HuggingFaceEmbeddings): 嵌入模型
    返回:
        vector_store (FAISS): 向量存储实例
    """
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(store_file)
    return vector_store

def load_vector_store(store_path, embeddings):
    """
    从本地加载向量存储。
    参数:
        store_path (str): 存储路径
        embeddings (HuggingFaceEmbeddings): 嵌入模型
    返回:
        vector_store (FAISS): 向量存储实例
    """
    if os.path.exists(store_path):
        vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    else:
        return None

def load_or_create_store(store_path, file_paths, embeddings):
    """
    加载或创建新的向量存储。
    参数:
        store_path (str): 存储路径
        file_paths (list): 文件路径列表
        embeddings (HuggingFaceEmbeddings): 嵌入模型
    返回:
        vector_store (FAISS): 向量存储实例
    """
    if os.path.exists(store_path) and file_paths_match(store_path, file_paths):
        print("向量数据库与上次使用时一致，无需重新写入")
        vector_store = load_vector_store(store_path, embeddings)
        if vector_store:
            return vector_store
    
    print("重新写入数据库")
    pages = load_files(file_paths)
    docs = split_text(pages)
    vector_store = create_vector_store(docs, store_path, embeddings)
    
    save_file_paths(store_path, file_paths)
    
    return vector_store

def query_vector_store(vector_store: FAISS, query, k=4, relevance_threshold=0.8):
    """
    从向量存储中查询相似文档。
    参数:
        vector_store (FAISS): 向量存储实例
        query (str): 查询内容
        k (int): 返回文档数量
        relevance_threshold (float): 相关性阈值
    返回:
        context (list): 查询到的上下文内容
    """
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": relevance_threshold, "k": k})
    similar_docs = retriever.invoke(query)
    context = [doc.page_content for doc in similar_docs]
    return context

def load_llm(model_path):
    """
    加载本地的大语言模型。
    参数:
        model_path (str): 模型路径
    返回:
        model (AutoModelForCausalLM): 大语言模型实例
        tokenizer (AutoTokenizer): 词汇表实例
    """
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def ask(model, tokenizer: AutoTokenizer, prompt, max_tokens=512):
    """
    生成模型回答。
    参数:
        model (AutoModelForCausalLM): 大语言模型实例
        tokenizer (AutoTokenizer): 词汇表实例
        prompt (str): 问题提示
        max_tokens (int): 最大生成的token数
    返回:
        str: 生成的回答
    """
    background_prompt = """
    You are J.A.R.V.I.S., a highly capable and intelligent private assistant...
    """
    
    messages = [
        {"role": "system", "content": background_prompt},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    terminators = [token for token in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")] if token]

    outputs = model.generate(input_ids, max_new_tokens=max_tokens, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)

    response = outputs[0][input_ids.shape[-1]:]
    for token_id in response:
        token = tokenizer.decode(token_id, skip_special_tokens=True)
        print(token, end="", flush=True)
        time.sleep(0.05)
    print()
    return ""

# %%
# 主函数
def main():
    embedding_model = create_embedding_model(Embedding_Model)
    vector_store = load_or_create_store(store_path, file_paths, embedding_model)
    model, tokenizer = load_llm(LLM_Model)

    while True:
        qiz = input("请输入您的问题：\n")
        if qiz in ['quit', 'exit']:
            print('程序关闭')
            break

        context = query_vector_store(vector_store, qiz, 4, 0.7)
        if not context:
            print('找不到匹配的上下文，直接向LLM询问。')
            prompt = f'请回答问题：\n{qiz}\n'
        else:
            context = '\n'.join(context)
            prompt = f'根据以下内容：\n{context}\n请回答问题：\n{qiz}\n'

        ans = ask(model, tokenizer, prompt)
        print("\n")

if __name__ == '__main__':
    main()

import json
import os.path
import tempfile
import sys
import re
import uuid
import requests
from argparse import ArgumentParser

import torchaudio
from transformers import WhisperFeatureExtractor, AutoTokenizer, AutoModel
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
import whisper


sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")

from speech_tokenizer.utils import extract_speech_token

import gradio as gr
import torch

audio_token_pattern = re.compile(r"<\|audio_(\d+)\|>")

from flow_inference import AudioDecoder


import os
import torch
from langchain_community.document_loaders import *
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import joblib
import time
import datetime
from threading import Thread
import json

Embedding_Model = '/root/autodl-tmp/rag/multilingual-e5-large-instruct'
file_paths = ['/root/autodl-tmp/rag/me.txt', "/root/autodl-tmp/rag/2024-Wealth-Outlook-MidYear-Edition.pdf"]
store_path = '/root/autodl-tmp/rag/me.faiss'

# load file.
# 定义文件扩展名与加载器类的映射
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
    # 获取文件扩展名并转换为小写
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # 根据扩展名获取对应的加载器类
    loader_class = LOADER_MAPPING.get(ext)
    if not loader_class:
        print(f"不支持的文件类型: {ext}")
        return None

    # 实例化加载器并加载文档
    loader = loader_class(file_path)
    docs = list(loader.lazy_load())
    return docs

def load_files(file_paths: list):
    if not file_paths:
        return []
    
    docs = []
    for file_path in tqdm(file_paths):
        print("Loading docs:", file_path)
        loaded_docs = load_single_file(file_path)
        if loaded_docs:
            docs.extend(loaded_docs)  # 使用 extend 而不是 append
    return docs


# split the text into docs.
def split_text(txt, chunk_size=200, overlap=20):
    if not txt:
        return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = splitter.split_documents(txt)
    return docs

# create embedding model from local source.
def create_embedding_model(model_file):
    # embedding = HuggingFaceEmbeddings(model_name=model_file,)
    embedding = HuggingFaceEmbeddings(model_name=model_file, model_kwargs={'trust_remote_code': True})
    # embedding = SentenceTransformer(model_file, trust_remote_code=True)
    return embedding

# 保存 file_paths
def save_file_paths(store_path, file_paths):
    joblib.dump(file_paths, f'{store_path}/file_paths.pkl')

# 加载 file_paths
def load_file_paths(store_path):
    file_paths_file = f'{store_path}/file_paths.pkl'
    if os.path.exists(file_paths_file):
        return joblib.load(file_paths_file)
    return None

# 比较文件路径列表
def file_paths_match(store_path, file_paths):
    saved_file_paths = load_file_paths(store_path)
    return saved_file_paths == file_paths

# 创建并保存向量存储
def create_vector_store(docs, store_file, embeddings):
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(store_file)
    return vector_store

# 从本地加载向量存储
def load_vector_store(store_path, embeddings):
    if os.path.exists(store_path):
        vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    else:
        return None

# 加载或创建新的向量存储
def load_or_create_store(store_path, file_paths, embeddings):
    # 检查文件路径是否匹配
    if os.path.exists(store_path) and file_paths_match(store_path, file_paths):
        print("向量数据库与上次使用时一致，无需重新写入")
        vector_store = load_vector_store(store_path, embeddings)
        if vector_store:
            return vector_store
    
    # 如果文件路径不匹配或存储不存在，重新创建存储
    print("重新写入数据库")
    pages = load_files(file_paths)
    docs = split_text(pages)
    vector_store = create_vector_store(docs, store_path, embeddings)
    
    # 保存文件路径列表
    save_file_paths(store_path, file_paths)
    
    return vector_store


# query content from store (retrieval).
def query_vector_store(vector_store: FAISS, query, k=4, relevance_threshold=0.8):
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": relevance_threshold, "k": k})
    similar_docs = retriever.invoke(query)
    context = [doc.page_content for doc in similar_docs]
    return context

# load llm from local.
def load_llm(model_path):
    # quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    # model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def initialize_embeding_model_and_vector_store(Embedding_Model, store_path, file_paths):
    embedding_model = create_embedding_model(Embedding_Model)
    vector_store = load_or_create_store(store_path, file_paths, embedding_model)

    return vector_store, embedding_model


if __name__ == "__main__":
    # 解析命令行参数
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")  # 服务器运行的主机
    parser.add_argument("--port", type=int, default="7860")  # 端口号
    parser.add_argument("--flow-path", type=str, default="./glm-4-voice-decoder")  # 流模型的文件路径
    parser.add_argument("--model-path", type=str, default="./weights")  # GLM 模型的权重路径
    parser.add_argument("--tokenizer-path", type=str, default="./glm-4-voice-tokenizer")  # 分词器文件的路径
    parser.add_argument("--whisper_model", type=str, default="/root/autodl-tmp/whisper/base")
    parser.add_argument("--share", action='store_true')
    args = parser.parse_args()

    # 定义模型配置和检验点路径
    flow_config = os.path.join(args.flow_path, "config.yaml")
    flow_checkpoint = os.path.join(args.flow_path, 'flow.pt')
    hift_checkpoint = os.path.join(args.flow_path, 'hift.pt')
    glm_tokenizer = None
    device = "cuda"  # 使用 GPU 进行推理
    audio_decoder: AudioDecoder = None  # 音频解码器实例，后置初始化
    whisper_model, feature_extractor = None, None  # Whisper 模型和特征提取器的位置缺省处


    # 准备函数，用于初始化必要的模型
    def initialize_fn():
        """
        初始化所有需要的模型和组件
        """
        global audio_decoder, feature_extractor, whisper_model, glm_model, glm_tokenizer
        global vector_store, embedding_model, whisper_transcribe_model  # 添加为全局变量
    
        # 如果音频解码器已经初始化，则返回
        if audio_decoder is not None:
            return
    
        # 载入 GLM 模型的分词器
        glm_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
        # 载入音频流模型和 hift 模型
        audio_decoder = AudioDecoder(config_path=flow_config, flow_ckpt_path=flow_checkpoint,
                                     hift_ckpt_path=hift_checkpoint,
                                     device=device)
    
        # 载入 Whisper 模型用于音频分词
        whisper_model = WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(device)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)
    
        # 初始化向量存储和嵌入模型
        embedding_model = create_embedding_model(Embedding_Model)
        vector_store = load_or_create_store(store_path, file_paths, embedding_model)
    
        # 载入 Whisper 转录模型
        whisper_transcribe_model = whisper.load_model("/root/autodl-tmp/whisper/base/base.pt")

    # 清除函数，用于清空 UI 状态
    def clear_fn():
        """
        清除 UI 的状态值
        """
        return [], [], '', '', '', None, None
    
    # 进行模型推理的函数
    def inference_fn(
            temperature: float,
            top_p: float,
            max_new_token: int,
            input_mode,
            audio_path: str | None,
            input_text: str | None,
            history: list[dict],
            previous_input_tokens: str,
            previous_completion_tokens: str,
    ):
        global whisper_transcribe_model, vector_store  # 使用全局的模型和向量存储
        using_context = False
    
        print(vector_store)
    
        # 根据音频或文本输入进行推理
        if input_mode == "audio":
            assert audio_path is not None
            history.append({"role": "user", "content": {"path": audio_path}})
            audio_tokens = extract_speech_token(
                whisper_model, feature_extractor, [audio_path]
            )[0]
            if len(audio_tokens) == 0:
                raise gr.Error("No audio tokens extracted")
            audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
            user_input = audio_tokens
            system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in an interleaved manner, with 13 text tokens followed by 26 audio tokens."
    
        # 如果输入模式是文本，使用输入的文本
        else:
            assert input_text is not None
            history.append({"role": "user", "content": input_text})
            user_input = input_text
            system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in an interleaved manner, with 13 text tokens followed by 26 audio tokens."
    
        # 将音频输入使用whisper转换为文本后使用langchain查找数据库
        if input_mode == "audio":
            
            whisper_result = whisper_transcribe_model.transcribe(audio_path)
            # 获取转录的文本
            transcribed_text = whisper_result['text']
            context = query_vector_store(vector_store, transcribed_text, 4, 0.7)
        else:
            context = query_vector_store(vector_store, input_text, 4, 0.7)
        if not context==None:
            using_context = True
    
        # 获取上一次输入的历史
        inputs = previous_input_tokens + previous_completion_tokens
        inputs = inputs.strip()
        if "<|system|>" not in inputs:
            inputs += f"<|system|>\n{system_prompt}"
        if ("<|context|>" not in inputs) and (using_context == True):
            inputs += f"<|context|> According to the following content: {context}, Please answer the question"
        if "<|context|>" not in inputs and context is not None:
            inputs += f"<|context|>\n{context}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

        with torch.no_grad():
            # 向本地服务器发送 POST 请求，生成流式的响应
            response = requests.post(
                "http://127.0.0.1:10000/generate_stream",
                data=json.dumps({
                    "prompt": inputs,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_token,
                }),
                stream=True
            )
            
            # 用于存储返回和提示数据的变量
            text_tokens, audio_tokens = [], []
            audio_offset = glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
            end_token_id = glm_tokenizer.convert_tokens_to_ids('<|user|>')
            complete_tokens = []
            prompt_speech_feat = torch.zeros(1, 0, 80).to(device)
            flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)
            this_uuid = str(uuid.uuid4())
            tts_speechs = []
            tts_mels = []
            prev_mel = None
            is_finalize = False
            block_size = 10
            
            # 处理每个流的响应片段
            for chunk in response.iter_lines():
                token_id = json.loads(chunk)["token_id"]
                if token_id == end_token_id:
                    is_finalize = True
                if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                    block_size = 20
                    tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)

                    # 合并 mel 调调图用于生成音频
                    if prev_mel is not None:
                        prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                    # 从音频代码生成语音
                    tts_speech, tts_mel = audio_decoder.token2wav(tts_token, uuid=this_uuid,
                                                                  prompt_token=flow_prompt_speech_token.to(device),
                                                                  prompt_feat=prompt_speech_feat.to(device),
                                                                  finalize=is_finalize)
                    prev_mel = tts_mel

                    tts_speechs.append(tts_speech.squeeze())
                    tts_mels.append(tts_mel)
                    yield history, inputs, '', '', (22050, tts_speech.squeeze().cpu().numpy()), None
                    flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                    audio_tokens = []
                if not is_finalize:
                    complete_tokens.append(token_id)
                    if token_id >= audio_offset:
                        audio_tokens.append(token_id - audio_offset)
                    else:
                        text_tokens.append(token_id)
        
        # 生成完成的音频并保存
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()
        complete_text = glm_tokenizer.decode(complete_tokens, spaces_between_special_tokens=False)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            torchaudio.save(f, tts_speech.unsqueeze(0), 22050, format="wav")
        history.append({"role": "assistant", "content": {"path": f.name, "type": "audio/wav"}})
        history.append({"role": "assistant", "content": glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)})
        yield history, inputs, complete_text, '', None, (22050, tts_speech.numpy())

    # 更新输入控件的端口选项相应的 UI 可见性
    def update_input_interface(input_mode):
        """
        根据选择的输入模式，更新输入控件的可见性
        """
        if input_mode == "audio":
            return [gr.update(visible=True), gr.update(visible=False)]
        else:
            return [gr.update(visible=False), gr.update(visible=True)]

    # 创建 Gradio 界面
    with gr.Blocks(title="GLM-4-Voice Demo", fill_height=True) as demo:
        with gr.Row():
            # 模型参数的控件
            temperature = gr.Number(label="Temperature", value=0.2)
            top_p = gr.Number(label="Top p", value=0.8)
            max_new_token = gr.Number(label="Max new tokens", value=2000)

        # 聊天机器人组件
        chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages", scale=1)

        with gr.Row():
            with gr.Column():
                # 输入模式的单选按钮
                input_mode = gr.Radio(["audio", "text"], label="Input Mode", value="audio")
                # 音频和文本输入组件
                audio = gr.Audio(label="Input audio", type='filepath', show_download_button=True, visible=True)
                text_input = gr.Textbox(label="Input text", placeholder="Enter your text here...", lines=2, visible=False)

            with gr.Column():
                # 提交和清除按钮
                submit_btn = gr.Button("Submit")
                reset_btn = gr.Button("Clear")
                # 音频输出组件
                output_audio = gr.Audio(label="Play", streaming=True, autoplay=True, show_download_button=False)
                complete_audio = gr.Audio(label="Last Output Audio (If Any)", show_download_button=True)

        # 模型输入和返回的调试信息
        gr.Markdown("""## Debug Info""")
        with gr.Row():
            input_tokens = gr.Textbox(label=f"Input Tokens", interactive=False)
            completion_tokens = gr.Textbox(label=f"Completion Tokens", interactive=False)

        detailed_error = gr.Textbox(label=f"Detailed Error", interactive=False)
        history_state = gr.State([])

        # 设置交互调用函数，用于提交输入和清除状态
        respond = submit_btn.click(
            inference_fn,
            inputs=[temperature, top_p, max_new_token, input_mode, audio, text_input, history_state, input_tokens, completion_tokens, ],
            outputs=[history_state, input_tokens, completion_tokens, detailed_error, output_audio, complete_audio]
        )

        respond.then(lambda s: s, [history_state], chatbot)

        reset_btn.click(clear_fn, outputs=[chatbot, history_state, input_tokens, completion_tokens, detailed_error, output_audio, complete_audio])
        input_mode.input(clear_fn, outputs=[chatbot, history_state, input_tokens, completion_tokens, detailed_error, output_audio, complete_audio]).then(update_input_interface, inputs=[input_mode], outputs=[audio, text_input])

    # 初始化模型
    initialize_fn()

    # 启动 Gradio 界面
    demo.launch(
        server_port=args.port,
        server_name=args.host,
        # share=True,
        share=args.share
    )

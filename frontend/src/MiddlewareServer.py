from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import whisper
import torch
from transformers import WhisperFeatureExtractor, AutoTokenizer
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import json
import uvicorn

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储模型和向量库
whisper_model = None
feature_extractor = None
vq_encoder = None
vector_store = None
embedding_model = None

def initialize_models():
    global whisper_model, feature_extractor, vq_encoder, vector_store, embedding_model
    
    # 初始化Whisper模型
    whisper_model = whisper.load_model("./weights/base.pt")
    
    # 初始化VQ编码器
    vq_encoder = WhisperVQEncoder.from_pretrained("E:\ML\example\nlp\GLM-4-Voice-9B\glm-4-voice-tokenizer").eval().cuda()
    feature_extractor = WhisperFeatureExtractor.from_pretrained("/path/to/tokenizer")
    
    # 初始化embedding模型和向量库
    embedding_model = HuggingFaceEmbeddings(
        model_name='/path/to/embedding/model',
        model_kwargs={'trust_remote_code': True}
    )
    vector_store = FAISS.load_local(
        '/path/to/store.faiss',
        embedding_model,
        allow_dangerous_deserialization=True
    )

@app.on_event("startup")
async def startup_event():
    initialize_models()

@app.post("/process")
async def process(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = None,
    settings: dict = None
):
    try:
        context = None
        user_input = ""
        
        if file:
            # 处理音频文件
            audio_path = save_temp_audio(file)
            
            # Whisper转录
            whisper_result = whisper_model.transcribe(audio_path)
            transcribed_text = whisper_result['text']
            
            # 获取音频tokens
            audio_tokens = extract_speech_token(
                vq_encoder,
                feature_extractor,
                [audio_path]
            )[0]
            
            audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            user_input = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
            
            # RAG检索
            context = query_vector_store(vector_store, transcribed_text)
            
        else:
            # 处理文本输入
            user_input = text
            context = query_vector_store(vector_store, text)
            
        # 构建提示
        system_prompt = "User will provide you with a text instruction..."
        inputs = f"<|system|>\n{system_prompt}"
        
        if context:
            inputs += f"<|context|>\n{context}"
            
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        
        # 调用模型服务器
        response = requests.post(
            "http://127.0.0.1:6006/generate_stream",
            data=json.dumps({
                "prompt": inputs,
                "temperature": settings.get("temperature", 0.2),
                "top_p": settings.get("top_p", 0.8),
                "max_new_tokens": settings.get("max_new_tokens", 2000)
            }),
            stream=True
        )
        
        return StreamingResponse(response.iter_lines())
        
    except Exception as e:
        return {"error": str(e)}

def query_vector_store(store, query, k=4, threshold=0.7):
    retriever = store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": threshold, "k": k}
    )
    similar_docs = retriever.invoke(query)
    return [doc.page_content for doc in similar_docs]

def save_temp_audio(file: UploadFile) -> str:
    # 保存上传的音频文件并返回路径
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(file.file.read())
    return temp_path

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
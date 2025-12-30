import os
import time
from uuid import uuid4
from tqdm.auto import tqdm
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

# ==========================================
# 1. 환경 설정 및 초기화
# ==========================================
load_dotenv(override=True)

# API 키 및 설정값
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
INDEX_NAME = "lowbirth"
NAMESPACE = "lowbirth_1"
BATCH_SIZE = 100

# 서비스 객체 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embedding = OpenAIEmbeddings(model='text-embedding-3-small', api_key=OPENAI_API_KEY)

# ==========================================
# 2. PDF 로드 및 텍스트 분할(Chunking) 설정
# ==========================================
# PDF 로드
loader = PyPDFLoader("asiabrief_3-26.pdf")
data = loader.load()

# 스플리터 설정 (의미 단위로 자르기)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, 
    chunk_overlap=30, 
    length_function=len,
    separators=['\n\n', '\n', ' ', '']
)

# ==========================================
# 3. 데이터 가공 및 Batch 업로드
# ==========================================
texts = []      # 임베딩할 텍스트 리스트
metadatas = []  # 저장할 메타데이터 리스트

print(f"데이터 처리 시작: {len(data)} 페이지 발견")

for doc in tqdm(data, desc="Processing Pages"):
    full_text = doc.page_content
    source = doc.metadata.get("source", "unknown")
    page_num = doc.metadata.get("page", 0)

    # 한 페이지를 여러 청크로 나눔
    chunks = splitter.split_text(full_text)

    for idx, chunk in enumerate(chunks):
        texts.append(chunk)
        metadatas.append({
            "chunk_id": f"{page_num}_{idx}", # 페이지와 순서를 조합한 ID
            "chunk_text": chunk,
            "source": source,
            "page": page_num,
            "title": "asiabrief_3-26"
        })

        # BATCH_SIZE만큼 모이면 업로드
        if len(texts) == BATCH_SIZE:
            ids = [str(uuid4()) for _ in range(BATCH_SIZE)]
            embeddings = embedding.embed_documents(texts)
            
            index.upsert(
                vectors=zip(ids, embeddings, metadatas),
                namespace=NAMESPACE
            )
            
            # 초기화 및 API 부하 방지를 위한 짧은 휴식
            texts, metadatas = [], []
            time.sleep(1)

# ==========================================
# 4. 남은 데이터 처리 (Last Batch)
# ==========================================
if texts:
    print(f"마지막 남은 {len(texts)}개 청크 업로드 중...")
    ids = [str(uuid4()) for _ in range(len(texts))]
    embeddings = embedding.embed_documents(texts)
    index.upsert(
        vectors=zip(ids, embeddings, metadatas),
        namespace=NAMESPACE
    )

print("✅ 모든 데이터가 Pinecone에 성공적으로 업로드되었습니다.")
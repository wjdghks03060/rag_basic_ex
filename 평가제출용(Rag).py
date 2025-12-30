import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 초기 설정 (한 번만 실행되면 되는 것들)
load_dotenv(override=True)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# 객체들을 루프 밖에서 미리 생성해두면 속도가 훨씬 빠릅니다.
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("lowbirth")
embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

chat_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 조수입니다. 제공된 context의 내용만을 바탕으로 답변해주세요."),
    ("human", "질문: {question}\n\n참고 내용(context): {context}"),
])

def search_top_k(question_text):
    # 질문을 벡터로 변환
    embedded_question = embedding.embed_query(question_text)

    # Pinecone 검색
    query_result = index.query(
        namespace="lowbirth_1",
        vector=embedded_question,
        top_k=3,
        include_metadata=True
    )

    context_list = []
    for match in query_result.matches:
        if "chunk_text" in match.metadata:
            context_list.append(match.metadata["chunk_text"])
    
    return "\n\n".join(context_list)

# 2. 메인 실행 루프
if __name__ == "__main__":
    print("=== 저출산 관련 RAG 채팅봇을 시작합니다 (종료하시려면 'exit'를 입력하세요) ===")
    
    chain = chat_template | llm | StrOutputParser()

    # 무한 루프 시작
    while True:
        print("\n" + "="*50)
        user_question = input("질문 (종료: exit): ").strip()

        # [핵심] 종료 조건 확인
        if user_question.lower() in ['exit', 'quit', '종료', '나가기']:
            print("채팅을 종료합니다. 감사합니다!")
            break # 루프를 빠져나가 프로그램 종료

        if not user_question: # 빈 입력 처리
            continue

        # 1단계: 검색
        print("🔍 관련 내용을 찾는 중...")
        top_k_context = search_top_k(user_question)

        # 2단계: 답변 생성
        print("🤖 답변 생성 중...")
        response = chain.invoke({
            "question": user_question,
            "context": top_k_context
        })

        print(f"\n[답변]: {response}")
import sys
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks.base import BaseCallbackHandler

# ChromaDB 최적화를 위한 설정
# Streamlit Cloud와 같은 환경에서 SQLite3 버전 문제를 해결합니다.
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 스트리밍 핸들러 정의 ---
# LLM 토큰을 실시간으로 화면에 출력하기 위한 클래스입니다.
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """LLM이 새 토큰을 생성할 때마다 호출됩니다."""
        self.text += token
        self.container.markdown(self.text) # Markdown 형식으로 실시간 출력

# --- Streamlit UI 설정 ---
st.set_page_config(page_title="CHATPDF", page_icon="📄")
st.title("CHATPDF 📄")
st.write("PDF 파일을 업로드하고 질문을 입력하면 AI가 답변해 드립니다.")
st.write("---")

# 1. 사용자 입력 받기 (사이드바)
with st.sidebar:
    st.header("1. 설정")
    # OpenAI API 키 입력
    openai_api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")
    st.caption("🔑 API 키는 저장되지 않습니다.")
    
    st.header("2. 파일 업로드")
    # PDF 파일 업로드
    uploaded_file = st.file_uploader("PDF 파일을 선택하세요.", type="pdf")

# 메인 화면 구성
st.header("3. 질문하기")
question = st.text_input("업로드한 PDF 내용에 대해 질문을 입력하세요:", placeholder="예: 이 문서의 주요 내용은 무엇인가요?")

# '답변 생성' 버튼
if st.button("답변 생성"):
    # --- 입력값 유효성 검사 ---
    if not uploaded_file:
        st.error("먼저 PDF 파일을 업로드해주세요.")
    elif not openai_api_key:
        st.error("OpenAI API 키를 입력해주세요.")
    elif not question:
        st.error("질문을 입력해주세요.")
    else:
        # --- 모든 로직 실행 ---
        with st.spinner("PDF를 처리하고 답변을 생성하는 중입니다... 잠시만 기다려주세요."):
            try:
                # 2. 업로드된 파일을 임시 파일로 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_filepath = tmp_file.name

                # 3. PDF 로드 및 분할
                loader = PyPDFLoader(temp_filepath)
                pages = loader.load_and_split()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,  # 청크 크기 증가
                    chunk_overlap=50, # 오버랩 증가
                    length_function=len,
                    is_separator_regex=False,
                )
                docs = text_splitter.split_documents(pages)

                # 4. 임베딩 및 벡터 DB 생성 (Chroma)
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=openai_api_key,
                )
                # from_documents를 사용해 문서를 벡터로 변환하고 DB에 저장
                db = Chroma.from_documents(docs, embeddings)

                # 5. RAG 체인 설정 및 실행
                # LLM 초기화 (Retriever용과 Streaming용)
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
                
                # MultiQueryRetriever 설정: 하나의 질문을 여러 관점에서 다시 질문하여 더 정확한 문서를 찾습니다.
                retriever = MultiQueryRetriever.from_llm(
                    retriever=db.as_retriever(),
                    llm=llm
                )
                
                # 프롬프트 가져오기
                prompt = hub.pull("rlm/rag-prompt")

                # 답변이 출력될 영역
                st.write("---")
                st.subheader("🤖 AI 답변")
                chat_box = st.empty()
                stream_handler = StreamHandler(chat_box)
                
                # 스트리밍을 위한 LLM 설정
                streaming_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=openai_api_key,
                    streaming=True,
                    callbacks=[stream_handler] # 스트림 핸들러 연결
                )

                # 문서를 하나의 문자열로 포맷팅하는 함수
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # RAG(검색 증강 생성) 체인 구성
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | streaming_llm  # 스트리밍 LLM 사용
                    | StrOutputParser()
                )

                # 체인 실행 (결과는 스트리밍 핸들러가 chat_box에 바로 출력)
                response = rag_chain.invoke(question)

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
            finally:
                # 임시 파일 삭제
                if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                    os.remove(temp_filepath)


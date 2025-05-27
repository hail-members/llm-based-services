# jargon_db.py
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json
import os

class JargonDatabase:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vector_db = None
        self.load_embeddings()
        self.build_database()
    
    def load_embeddings(self):
        """문장 임베딩 모델 로드"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            print(f"✅ 임베딩 모델 로드 성공: {self.embedding_model}")
        except Exception as e:
            print(f"❌ 임베딩 모델 로드 실패: {e}")
    
    def get_jargon_data(self):
        """Jargon 데이터 정의 (실제로는 JSON 파일에서 로드)"""
        return [
            {
                "term": "딥러닝이 뭐야?",
                "definition": "여러 층을 가진 인공신경망(Deep Neural Network)을 사용하여 기계학습을 수행하는 기술입니다. 인간의 뇌 신경망을 모방하여 복잡한 패턴을 학습할 수 있습니다.",
                "examples": "이미지 인식, 자연어 처리, 음성 인식 등에 활용됩니다."
            },
            {
                "term": "RAG이 뭐야?",
                "definition": "Retrieval-Augmented Generation의 약자로, 외부 지식 소스에서 관련 정보를 검색하여 대규모 언어모델(LLM)의 응답 생성을 보강하는 방법입니다.",
                "examples": "챗봇이 최신 정보나 특정 도메인 지식을 활용해 답변할 때 사용됩니다."
            },
            {
                "term": "임베딩이 뭐야?",
                "definition": "단어, 문장, 이미지 등의 데이터를 컴퓨터가 이해할 수 있는 고정 길이의 벡터(숫자 배열)로 변환하는 기술입니다.",
                "examples": "Word2Vec, BERT, OpenAI Embedding 등이 대표적입니다."
            },
            {
                "term": "파인튜닝이 뭐야?",
                "definition": "사전 훈련된 대규모 모델을 특정 작업이나 도메인에 맞게 추가로 학습시키는 과정입니다.",
                "examples": "GPT를 의료 분야에 특화시키거나, BERT를 감정 분석에 맞게 조정하는 것입니다."
            },
            {
                "term": "트랜스포머이 뭐야?",
                "definition": "Attention 메커니즘을 핵심으로 하는 신경망 아키텍처로, 순차적 데이터 처리에서 뛰어난 성능을 보입니다.",
                "examples": "GPT, BERT, T5 등 대부분의 현대 언어모델이 트랜스포머 구조를 사용합니다."
            },
            {
                "term": "벡터 데이터베이스이 뭐야?",
                "definition": "벡터 형태로 변환된 데이터를 효율적으로 저장하고 유사도 기반 검색을 수행할 수 있는 특수한 데이터베이스입니다.",
                "examples": "Pinecone, Weaviate, Chroma, FAISS 등이 있습니다."
            }
        ]
    
    def build_database(self):
        """벡터 데이터베이스 구축"""
        if not self.embeddings:
            print("❌ 임베딩 모델이 로드되지 않아 벡터 DB를 생성할 수 없습니다.")
            return
        
        try:
            jargon_data = self.get_jargon_data()
            documents = []
            
            for item in jargon_data:
                # 용어와 정의, 예시를 결합한 텍스트 생성
                content = f"{item['definition']} {item.get('examples', '')}"
                doc = Document(
                    page_content=content,
                    metadata={
                        "term": item["term"],
                        "definition": item["definition"],
                        "examples": item.get("examples", "")
                    }
                )
                documents.append(doc)
            
            self.vector_db = FAISS.from_documents(documents, self.embeddings)
            print(f"✅ Jargon 벡터 DB 생성 성공: {len(documents)}개 용어")
            
        except Exception as e:
            print(f"❌ 벡터 DB 생성 실패: {e}")
    
    def search_jargon(self, query, k=2):
        """Jargon 검색"""
        if not self.vector_db:
            return "벡터 데이터베이스가 준비되지 않았습니다."
        
        try:
            docs = self.vector_db.similarity_search(query, k=k)
            
            if not docs:
                return "관련된 용어 정의를 찾지 못했습니다."
            
            results = []
            for doc in docs:
                term = doc.metadata.get("term", "")
                definition = doc.metadata.get("definition", "")
                examples = doc.metadata.get("examples", "")
                
                result = f"**{term}**: {definition}"
                if examples:
                    result += f"\n*예시: {examples}*"
                results.append(result)
            
            return "\n\n".join(results)
            
        except Exception as e:
            return f"검색 중 오류가 발생했습니다: {e}"

# 전역 Jargon DB 인스턴스
jargon_db = JargonDatabase()
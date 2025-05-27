
# llm_handler.py
from gpt4all import GPT4All
import os

class LLMHandler:
    def __init__(self, model_path):
        self.model_path = model_path
        self.llm = None
        self.load_model()
        
    def load_model(self):
        """GPT4All 모델 로드"""
        try:
            # if not os.path.exists(self.model_path):
            #     raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                
            self.llm = GPT4All(
                self.model_path
            )
            print(f"✅ 모델 로드 성공: {self.model_path}")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            self.llm = None
    
    def generate(self, prompt, max_tokens=1024, temperature=0.7, top_p=0.9):
        """텍스트 생성"""
        if not self.llm:
            return "❌ LLM 모델이 로드되지 않았습니다."
            
        try:
            with self.llm.chat_session():
                response = self.llm.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                    top_p=top_p
                )
                return response.strip()
                
        except Exception as e:
            return f"❌ 응답 생성 오류: {str(e)}"
    
    def is_ready(self):
        """모델 준비 상태 확인"""
        return self.llm is not None

# 전역 LLM 핸들러 인스턴스
llm_handler = LLMHandler("Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf")
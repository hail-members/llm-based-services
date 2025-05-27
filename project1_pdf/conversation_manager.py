# conversation_manager.py
from langchain.memory import ConversationBufferMemory
from datetime import datetime

class ConversationManager:
    def __init__(self, max_history=10):
        self.memory = ConversationBufferMemory(
            max_token_limit=2000,  # 메모리 크기 제한
            return_messages=True
        )
        self.max_history = max_history
        self.conversation_log = []
    
    def add_exchange(self, user_input, assistant_response):
        """대화 교환 추가"""
        try:
            # LangChain 메모리에 추가
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(assistant_response)
            
            # 로그에 추가 (타임스탬프 포함)
            exchange = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "user": user_input,
                "assistant": assistant_response
            }
            self.conversation_log.append(exchange)
            
            # 최대 히스토리 개수 제한
            if len(self.conversation_log) > self.max_history:
                self.conversation_log.pop(0)
                
        except Exception as e:
            print(f"대화 기록 추가 오류: {e}")
    
    def get_conversation_history(self, last_n=5):
        """최근 n개 대화 히스토리 반환"""
        recent_log = self.conversation_log[-last_n:] if len(self.conversation_log) > last_n else self.conversation_log
        
        history_text = ""
        for exchange in recent_log:
            history_text += f"사용자: {exchange['user']}\n"
            history_text += f"중재자: {exchange['assistant']}\n\n"
        
        return history_text.strip()
    
    def get_memory_variables(self):
        """LangChain 메모리 변수 반환"""
        try:
            return self.memory.load_memory_variables({})
        except Exception as e:
            print(f"메모리 변수 로드 오류: {e}")
            return {}
    
    def clear_memory(self):
        """대화 기록 초기화"""
        self.memory.clear()
        self.conversation_log = []
        print("대화 기록이 초기화되었습니다.")
    
    def get_formatted_chat_display(self):
        """GUI 출력용 포맷된 대화 내용"""
        display_text = ""
        for exchange in self.conversation_log:
            display_text += f"🕒 {exchange['timestamp']}\n"
            display_text += f"🙋 **사용자**: {exchange['user']}\n"
            display_text += f"🤖 **중재자**: {exchange['assistant']}\n"
            display_text += "-" * 50 + "\n\n"
        
        return display_text

# 전역 대화 관리자 인스턴스
conversation_manager = ConversationManager()
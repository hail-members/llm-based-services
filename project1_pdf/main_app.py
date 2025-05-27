# main_app.py
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QTextEdit, QFileDialog, 
    QLabel, QScrollArea, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from text_extractor import extract_text_from_url, extract_text_from_pdf
from llm_handler import llm_handler
from prompt_templates import (
    get_summary_prompt, get_debate_prompt, 
    get_mediator_prompt, get_jargon_qa_prompt
)
from jargon_db import jargon_db
from conversation_manager import conversation_manager


class PaperAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.conversation_history = []
        self.mediator_active = False
        
    def initUI(self):
        self.setWindowTitle('논문 분석 멀티페르소나 챗봇')
        self.setGeometry(100, 100, 1200, 900)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout()
        
        # 좌측: 입력 및 제어 패널
        left_panel = QVBoxLayout()
        
        # 우측: 대화 패널
        right_panel = QVBoxLayout()
        
        # URL 입력
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel('논문 URL:'))
        self.url_input = QLineEdit()
        url_layout.addWidget(self.url_input)
        self.url_btn = QPushButton('URL 분석')
        url_layout.addWidget(self.url_btn)
        left_panel.addLayout(url_layout)
        
        # PDF 업로드
        self.pdf_btn = QPushButton('PDF 업로드 및 분석')
        left_panel.addWidget(self.pdf_btn)
        
        # 분석 결과 출력
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        left_panel.addWidget(self.output_display)
        
        # 대화 히스토리
        right_panel.addWidget(QLabel('💬 중재자와의 대화'))
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        right_panel.addWidget(self.chat_display)
        
        # 질문 입력
        chat_input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("전문용어나 논문에 대해 질문하세요...")
        chat_input_layout.addWidget(self.chat_input)
        self.chat_btn = QPushButton('전송')
        chat_input_layout.addWidget(self.chat_btn)
        right_panel.addLayout(chat_input_layout)
        
        # 스플리터로 좌우 패널 분할
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 500])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
        # 이벤트 연결
        self.connect_events()
        
    def connect_events(self):
        self.url_btn.clicked.connect(self.analyze_url)
        self.pdf_btn.clicked.connect(self.analyze_pdf)
        self.chat_btn.clicked.connect(self.handle_chat)
        self.chat_input.returnPressed.connect(self.handle_chat)
        
        
    def analyze_url(self):
        """URL 분석 실행"""
        url = self.url_input.text().strip()
        if not url:
            self.show_error("URL을 입력해주세요.")
            return
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        self.set_status("URL에서 텍스트 추출 중...")
        text = extract_text_from_url(url)
        
        if text.startswith("❌"):
            self.show_error(text)
        else:
            self.run_full_analysis(text)
    
    def analyze_pdf(self):
        """PDF 분석 실행"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "PDF 파일 선택", "", "PDF Files (*.pdf)"
        )
        
        if file_path:
            self.set_status("PDF에서 텍스트 추출 중...")
            text = extract_text_from_pdf(file_path)
            
            if text.startswith("❌"):
                self.show_error(text)
            else:
                self.run_full_analysis(text)
    
    def run_full_analysis(self, paper_text):
        """논문 전체 분석 실행"""
        if not llm_handler.is_ready():
            self.show_error("LLM 모델이 준비되지 않았습니다.")
            return
        
        try:
            # 1. 논문 요약
            self.set_status("📋 논문 요약 생성 중...")
            summary_prompt = get_summary_prompt(paper_text)
            summary = llm_handler.generate(summary_prompt, max_tokens=1500)
            
            result = f"## 📋 논문 요약\n\n{summary}\n\n"
            self.output_display.setText(result)
            QApplication.processEvents()
            
            # 2. 페르소나 토론
            self.set_status("💬 페르소나 토론 생성 중...")
            debate_prompt = get_debate_prompt(summary)
            debate = llm_handler.generate(debate_prompt, max_tokens=2000)
            
            result += f"## 💬 페르소나 토론\n\n{debate}\n\n"
            self.output_display.setText(result)
            QApplication.processEvents()
            
            # 3. 중재자 요약
            self.set_status("⚖️ 중재자 종합 평가 생성 중...")
            mediator_prompt = get_mediator_prompt(debate)
            mediator_summary = llm_handler.generate(mediator_prompt, max_tokens=1000)
            
            result += f"## ⚖️ 중재자 종합 평가\n\n{mediator_summary}\n\n"
            result += "=" * 60 + "\n"
            result += "✅ 분석 완료! 우측 대화창에서 중재자와 대화할 수 있습니다."
            
            self.output_display.setText(result)
            self.mediator_active = True
            self.set_status("분석 완료 - 대화 모드 활성화됨")
            
            # 대화 기록 초기화 (새로운 논문 분석 시작)
            conversation_manager.clear_memory()
            self.update_chat_display()
            
        except Exception as e:
            self.show_error(f"분석 중 오류 발생: {str(e)}")
    
    def handle_chat(self):
        """멀티턴 대화 처리"""
        if not self.mediator_active:
            self.show_chat_message("❌ 먼저 논문 분석을 실행해주세요.")
            return
        
        user_input = self.chat_input.text().strip()
        if not user_input:
            return
        
        try:
            # 1. Jargon DB에서 관련 정보 검색
            retrieved_context = jargon_db.search_jargon(user_input, k=2)
            
            # 2. 이전 대화 히스토리 가져오기
            conversation_history = conversation_manager.get_conversation_history(last_n=3)
            
            # 3. 프롬프트 생성 및 LLM 응답
            qa_prompt = get_jargon_qa_prompt(user_input, retrieved_context, conversation_history)
            assistant_response = llm_handler.generate(qa_prompt, max_tokens=800, temperature=0.8)
            
            # 4. 대화 기록에 추가
            conversation_manager.add_exchange(user_input, assistant_response)
            
            # 5. 화면 업데이트
            self.update_chat_display()
            self.chat_input.clear()
            
        except Exception as e:
            self.show_chat_message(f"❌ 대화 처리 중 오류: {str(e)}")
    
    def update_chat_display(self):
        """대화 화면 업데이트"""
        chat_content = conversation_manager.get_formatted_chat_display()
        if not chat_content:
            chat_content = "💡 논문의 전문용어나 내용에 대해 자유롭게 질문해보세요!\n\n"
        
        self.chat_display.setText(chat_content)
        
        # 스크롤을 맨 아래로
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def show_chat_message(self, message):
        """단일 채팅 메시지 표시"""
        current_content = self.chat_display.toPlainText()
        self.chat_display.setText(current_content + f"\n{message}\n")
    
    def set_status(self, message):
        """상태 메시지 설정"""
        self.setWindowTitle(f'논문 분석 멀티페르소나 챗봇 - {message}')
        QApplication.processEvents()
    
    def show_error(self, error_message):
        """오류 메시지 표시"""
        self.output_display.setText(f"❌ 오류\n\n{error_message}")
        self.set_status("오류 발생")

# 메인 실행
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PaperAnalyzerApp()
    ex.show()
    sys.exit(app.exec_())
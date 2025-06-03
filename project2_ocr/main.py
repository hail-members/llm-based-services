# main.py
import sys
import os
import time
import threading
import traceback # 에러 로깅을 위해 추가

# OMP: Error #15 해결을 위한 환경 변수 설정 (다른 임포트보다 먼저)
# 이 설정은 여러 OpenMP 런타임이 로드될 때 발생하는 오류를 회피하기 위함입니다.
# Intel의 경고에 따르면 안전하지 않을 수 있으나, 많은 경우 문제를 해결합니다.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QProgressBar, QFileDialog,
    QDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QRunnable, QThreadPool, pyqtSlot
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QFont

# 실제 OCR 및 LLM 라이브러리 임포트 시도 (런타임에)
try:
    import easyocr
except ImportError:
    easyocr = None
    print("EasyOCR 라이브러리를 찾을 수 없습니다. 'pip install easyocr'로 설치해주세요.")

try:
    from gpt4all import GPT4All
except ImportError:
    GPT4All = None
    print("GPT4All 라이브러리를 찾을 수 없습니다. 'pip install gpt4all'로 설치해주세요.")

try:
    import cv2 # 이미지 로딩 확인을 위해 OpenCV 임포트
except ImportError:
    cv2 = None
    print("OpenCV (cv2) 라이브러리를 찾을 수 없습니다. 'pip install opencv-python'으로 설치해주세요.")


class WorkerSignals(QObject):
    """
    워커 스레드에서 GUI 스레드로 시그널을 보내기 위한 클래스.
    - finished: 작업 완료 시 발생
    - error: 에러 발생 시 발생 (에러 메시지 전달)
    - result: 작업 결과 발생 시 발생 (작업 유형, 데이터 전달)
    - progress: 진행률 업데이트 시 발생 (작업 유형, 퍼센티지 전달)
    """
    finished = pyqtSignal()
    error = pyqtSignal(str) 
    result = pyqtSignal(str, str)  # task_type, data
    progress = pyqtSignal(str, int) # task_type, percentage

class OCRLLMWorker(QRunnable):
    """
    OCR 및 LLM 작업을 백그라운드에서 수행하는 QRunnable 클래스.
    """
    def __init__(self, task_type, image_path=None, text_input=None,
                 model_name="Meta-Llama-3-8B-Instruct.Q4_0.gguf", 
                 stop_event=None, ocr_reader_instance=None, llm_model_instance=None):
        super().__init__()
        self.signals = WorkerSignals()
        self.task_type = task_type
        self.image_path = image_path
        self.text_input = text_input
        self.model_name = model_name
        self.stop_event = stop_event or threading.Event() # 중지 이벤트가 없으면 새로 생성

        # 미리 로드된 모델 인스턴스 사용 또는 여기서 로드
        self.ocr_reader = ocr_reader_instance
        self.llm_model = llm_model_instance
        
        self.ocr_text_internal = "" # OCR 결과를 내부적으로 저장하여 교정 단계에서 사용

    @pyqtSlot()
    def run(self):
        try:
            if self.task_type == "ocr_correct":
                if not easyocr:
                    raise ImportError("EasyOCR이 설치되지 않았습니다. 'pip install easyocr'로 설치 후 다시 시도해주세요.")
                if not GPT4All:
                    raise ImportError("GPT4All이 설치되지 않았습니다. 'pip install gpt4all'로 설치 후 다시 시도해주세요.")
                if not cv2:
                    raise ImportError("OpenCV (cv2)가 설치되지 않았습니다. 'pip install opencv-python'으로 설치 후 다시 시도해주세요.")


                self.signals.progress.emit(self.task_type, 5) # 시작
                print("image_path:", self.image_path)
                if not self.image_path or not os.path.exists(self.image_path):
                    raise ValueError("OCR을 위한 이미지 경로가 유효하지 않습니다.")

                # OpenCV를 사용하여 이미지 로딩 테스트
                img_cv = cv2.imread(self.image_path)
                if img_cv is None:
                    raise ValueError(f"OpenCV가 이미지를 로드할 수 없습니다: {self.image_path}. 파일 경로 또는 파일 손상 여부를 확인하세요.")
                
                # 1. OCR 수행
                self.signals.progress.emit(self.task_type, 10) # OCR 시작 알림
                if not self.ocr_reader: # 미리 로드된 인스턴스가 없다면
                    print("EasyOCR 리더를 이 작업에서 로드합니다...")
                    self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False) # 한국어, 영어 지원
                    print("EasyOCR 리더 로드 완료.")
                
                if self.stop_event.is_set(): 
                    self.signals.finished.emit()
                    return
                
                # EasyOCR은 이미지 경로 또는 NumPy 배열을 입력으로 받을 수 있음
                # 이미 cv2로 로드했으므로, 해당 객체를 사용할 수도 있으나,
                # EasyOCR 내부에서도 경로로 로드하므로 일단 경로를 그대로 사용.
                # 만약 문제가 지속되면 img_cv (BGR) -> RGB 변환 후 전달 시도 가능
                raw_ocr_results = self.ocr_reader.readtext(self.image_path, detail=0, paragraph=True)
                self.ocr_text_internal = "\n".join(raw_ocr_results)
                
                self.signals.result.emit("ocr_result", self.ocr_text_internal)
                self.signals.progress.emit(self.task_type, 50) # OCR 완료
                
                if self.stop_event.is_set(): 
                    self.signals.finished.emit()
                    return

                # 2. LLM 텍스트 교정
                self.signals.progress.emit(self.task_type, 55) # LLM 교정 시작 알림
                if not self.llm_model: # 미리 로드된 인스턴스가 없다면
                    print(f"GPT4All 모델 ({self.model_name})을 이 작업에서 로드합니다...")
                    self.llm_model = GPT4All(self.model_name, device='cpu', allow_download=True)
                    print("GPT4All 모델 로드 완료.")

                prompt_template_correction = """다음은 이미지에서 광학 문자 인식(OCR)을 통해 추출된 텍스트입니다. 이 텍스트에는 오류가 포함되어 있을 수 있습니다. 원본의 의미를 유지하면서 문법, 철자, 구두점 오류를 수정하고, 필요한 경우 가독성을 높이기 위해 문장을 자연스럽게 다듬어주십시오. 추가적인 설명이나 주석 없이 교정된 텍스트만 제공해주세요. 한국어로 답변해주세요.

OCR 텍스트:
---
{TEXT_TO_CORRECT}
---

교정된 텍스트:"""
                prompt = prompt_template_correction.format(TEXT_TO_CORRECT=self.ocr_text_internal)
                
                if self.stop_event.is_set(): 
                    self.signals.finished.emit()
                    return
                
                if not self.llm_model:
                     raise RuntimeError("LLM 모델이 초기화되지 않았습니다 (교정 단계).")


                with self.llm_model.chat_session():
                    corrected_text = self.llm_model.generate(prompt, max_tokens=len(self.ocr_text_internal) * 2 + 300, temp=0.6, top_k=40, top_p=0.9, streaming=False)
                
                self.signals.result.emit("corrected_text_result", corrected_text.strip())
                self.signals.progress.emit(self.task_type, 100) # 교정 완료

            elif self.task_type == "explain":
                if not GPT4All:
                    raise ImportError("GPT4All이 설치되지 않았습니다. 'pip install gpt4all'로 설치 후 다시 시도해주세요.")
                
                self.signals.progress.emit(self.task_type, 10) # 해설 시작 알림
                if not self.text_input:
                    raise ValueError("해설을 위한 텍스트 입력이 없습니다.")

                if not self.llm_model: # 미리 로드된 인스턴스가 없다면
                    print(f"GPT4All 모델 ({self.model_name})을 이 작업에서 로드합니다...")
                    self.llm_model = GPT4All(self.model_name, device='cpu', allow_download=True)
                    print("GPT4All 모델 로드 완료.")
                
                if self.stop_event.is_set(): 
                    self.signals.finished.emit()
                    return

                prompt_template_explanation = """다음은 문서의 내용입니다. 이 문서가 어떤 종류의 문서인지 (예: 이메일, 보고서, 기사, 메모 등), 주요 내용은 무엇인지, 그리고 이 문서의 전반적인 목적은 무엇으로 보이는지 간략하게 설명해주세요. 한국어로 답변해주세요.

문서 내용:
---
{DOCUMENT_TEXT}
---

설명:"""
                prompt = prompt_template_explanation.format(DOCUMENT_TEXT=self.text_input)

                if not self.llm_model:
                     raise RuntimeError("LLM 모델이 초기화되지 않았습니다 (해설 단계).")
                
                with self.llm_model.chat_session():
                    explanation = self.llm_model.generate(prompt, max_tokens=len(self.text_input) + 500, temp=0.7, top_k=50, top_p=0.95, streaming=False)
                
                self.signals.result.emit("explanation_result", explanation.strip())
                self.signals.progress.emit(self.task_type, 100) # 해설 완료

        except Exception as e:
            error_msg = f"작업 유형 '{self.task_type}' 처리 중 오류 발생:\n{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
        finally:
            self.signals.finished.emit()

class ComparisonDialog(QDialog):
    """
    OCR 결과와 LLM 정제 결과를 비교하고 사용자 결정을 받는 대화상자.
    """
    def __init__(self, image_path, ocr_text, refined_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("결과 비교 및 다음 단계 결정")
        self.setMinimumSize(900, 700) # 창 크기 조정

        self.image_path = image_path
        self.ocr_text_content = ocr_text if ocr_text else "OCR 결과가 없습니다."
        self.refined_text_content = refined_text if refined_text else "LLM 정제 결과가 없습니다."
        
        self.final_refined_text = self.refined_text_content # 사용자가 편집할 수 있는 텍스트

        main_layout = QVBoxLayout(self)

        # 상단 콘텐츠 (이미지, OCR 텍스트, 정제 텍스트)
        content_layout = QHBoxLayout()

        # 1. 이미지 표시 영역
        image_display_layout = QVBoxLayout()
        image_title = QLabel("원본 이미지")
        image_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label = QLabel()
        if self.image_path and os.path.exists(self.image_path):
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap.scaled(300, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.image_label.setText("이미지를 표시할 수 없습니다.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(280,380) # 최소 크기 설정
        image_display_layout.addWidget(image_title)
        image_display_layout.addWidget(self.image_label, 1) # stretch factor
        content_layout.addLayout(image_display_layout, 1) # 비율

        # 2. OCR 텍스트 표시 영역
        ocr_display_layout = QVBoxLayout()
        ocr_title = QLabel("OCR 추출 텍스트 (읽기 전용)")
        ocr_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ocr_text_edit = QTextEdit()
        self.ocr_text_edit.setPlainText(self.ocr_text_content)
        self.ocr_text_edit.setReadOnly(True)
        ocr_display_layout.addWidget(ocr_title)
        ocr_display_layout.addWidget(self.ocr_text_edit, 1)
        content_layout.addLayout(ocr_display_layout, 2)

        # 3. LLM 정제 텍스트 표시 영역 (편집 가능)
        refined_display_layout = QVBoxLayout()
        refined_title = QLabel("LLM 정제 텍스트 (편집 가능)")
        refined_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.refined_text_edit = QTextEdit()
        self.refined_text_edit.setPlainText(self.final_refined_text)
        # self.refined_text_edit.setReadOnly(False) # 기본적으로 편집 가능
        refined_display_layout.addWidget(refined_title)
        refined_display_layout.addWidget(self.refined_text_edit, 1)
        content_layout.addLayout(refined_display_layout, 2)
        
        main_layout.addLayout(content_layout)

        # 하단 버튼 영역
        button_layout = QHBoxLayout()
        self.redo_button = QPushButton("🔄 재시도 (Redo)")
        self.accept_button = QPushButton("👍 수락 및 해설 요청 (Accept)")
        self.cancel_button = QPushButton("❌ 취소 (Cancel)")

        button_layout.addStretch()
        button_layout.addWidget(self.redo_button)
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # 버튼 시그널 연결
        self.redo_button.clicked.connect(self.on_redo)
        self.accept_button.clicked.connect(self.on_accept)
        self.cancel_button.clicked.connect(self.reject) # QDialog의 기본 reject 슬롯

    def on_redo(self):
        self.done(1) # 사용자 정의 반환 코드: 재시도

    def on_accept(self):
        self.final_refined_text = self.refined_text_edit.toPlainText() # 수락 시 편집된 텍스트 가져오기
        self.done(2) # 사용자 정의 반환 코드: 수락

    def get_accepted_text(self):
        return self.final_refined_text

class ExplanationDialog(QDialog):
    """
    LLM의 문서 해설 결과를 보여주는 대화상자.
    """
    def __init__(self, explanation_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("문서 해설 결과")
        self.setMinimumSize(700, 500)

        layout = QVBoxLayout(self)
        self.explanation_text_edit = QTextEdit()
        self.explanation_text_edit.setPlainText(explanation_text if explanation_text else "해설 결과가 없습니다.")
        self.explanation_text_edit.setReadOnly(True)
        self.explanation_text_edit.setFont(QFont("Arial", 12)) # 폰트 크기 조절

        layout.addWidget(self.explanation_text_edit)

        close_button = QPushButton("닫기")
        close_button.clicked.connect(self.accept) # QDialog의 기본 accept 슬롯
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("이미지 분석 및 문서 해석기")
        self.setGeometry(100, 100, 1000, 800) # 창 크기 조정

        # 상태 변수
        self.current_image_path = None
        self.raw_ocr_text = ""
        self.current_corrected_text = "" # ComparisonDialog에 전달될 텍스트
        self.final_accepted_text_for_explanation = "" # ComparisonDialog에서 최종 수락된 텍스트

        # LLM 모델 설정
        self.llm_model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf" # 또는 다른 원하는 모델
        self.ocr_reader_instance = None 
        self.llm_model_instance = None  
        
        # --- 모델 미리 로드 (애플리케이션 시작 시) ---
        if easyocr:
            print("EasyOCR 리더 미리 로딩 중...")
            try:
                self.ocr_reader_instance = easyocr.Reader(['ko', 'en'], gpu=False) # GPU 사용 여부 설정
                print("EasyOCR 리더 미리 로드 완료.")
            except Exception as e:
                print(f"EasyOCR 미리 로드 실패: {e}\n{traceback.format_exc()}")
                self.ocr_reader_instance = None # 실패 시 None으로 설정
        else:
            print("EasyOCR 라이브러리가 없어 미리 로드할 수 없습니다.")

        if GPT4All:
            print(f"GPT4All 모델 ({self.llm_model_name}) 미리 로딩 중...")
            try:
                self.llm_model_instance = GPT4All(self.llm_model_name, device='cpu', allow_download=True)
                print("GPT4All 모델 미리 로드 완료.")
            except Exception as e:
                print(f"GPT4All 미리 로드 실패: {e}\n{traceback.format_exc()}")
                self.llm_model_instance = None # 실패 시 None으로 설정
        else:
            print("GPT4All 라이브러리가 없어 미리 로드할 수 없습니다.")
        # --- 모델 미리 로드 끝 ---


        self.threadpool = QThreadPool()
        self.current_worker_runnable = None # 현재 실행 중인 QRunnable 객체
        self.stop_current_worker_event = None # 현재 워커를 중지하기 위한 이벤트

        self._init_ui()
        self._connect_signals_slots()
        self.update_status_message("이미지 파일을 드래그 앤 드롭하거나 '이미지 선택' 버튼을 클릭하세요.")

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 상단: 이미지 표시 및 파일 선택 버튼
        top_controls_layout = QHBoxLayout()
        self.select_image_button = QPushButton("📂 이미지 파일 선택...")
        top_controls_layout.addWidget(self.select_image_button)
        top_controls_layout.addStretch()
        main_layout.addLayout(top_controls_layout)

        self.image_display_label = QLabel("이미지를 여기에 드래그 앤 드롭 하거나 위 버튼으로 선택하세요.")
        self.image_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display_label.setFrameShape(QLabel.Shape.StyledPanel)
        self.image_display_label.setMinimumHeight(300)
        self.image_display_label.setAcceptDrops(True) # 드래그 앤 드롭 활성화
        main_layout.addWidget(self.image_display_label, 1) # stretch factor

        # 중간: 진행률 표시줄 및 상태 메시지
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("대기 중...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        # 하단: 주요 동작 버튼들
        action_buttons_layout = QHBoxLayout()
        self.analyze_button = QPushButton("1. 이미지 분석 (OCR & 정제)")
        self.analyze_button.setEnabled(False)
        self.cancel_task_button = QPushButton("현재 작업 취소 / 초기화")
        
        action_buttons_layout.addWidget(self.analyze_button)
        action_buttons_layout.addWidget(self.cancel_task_button)
        main_layout.addLayout(action_buttons_layout)

    def _connect_signals_slots(self):
        self.select_image_button.clicked.connect(self.open_image_file_dialog)
        self.analyze_button.clicked.connect(self.run_ocr_correction_task)
        self.cancel_task_button.clicked.connect(self.cancel_or_reset_all_processes)

    def update_status_message(self, message):
        self.status_label.setText(message)
        print(f"Status: {message}") # 콘솔에도 출력

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            first_url = mime_data.urls()[0]
            if first_url.isLocalFile():
                file_path = first_url.toLocalFile()
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.load_image_and_prepare_analysis(file_path)
            event.accept()
        else:
            event.ignore()

    def open_image_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "이미지 파일 선택", "", "이미지 파일 (*.png *.jpg *.jpeg)")
        if file_path:
            self.load_image_and_prepare_analysis(file_path)

    def load_image_and_prepare_analysis(self, image_path):
        self.cancel_current_worker_if_running() # 새 이미지 로드 시 이전 작업 중지
        
        self.current_image_path = image_path
        pixmap = QPixmap(image_path)
        self.image_display_label.setPixmap(pixmap.scaled(self.image_display_label.size(),
                                                 Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation))
        self.raw_ocr_text = ""
        self.current_corrected_text = ""
        self.final_accepted_text_for_explanation = ""
        self.progress_bar.setValue(0)
        
        self.analyze_button.setEnabled(True)
        self.update_status_message(f"이미지 로드: {os.path.basename(image_path)}. '이미지 분석' 버튼을 클릭하세요.")

    def _start_background_worker(self, task_type, **kwargs):
        if self.current_worker_runnable is not None:
            self.update_status_message("오류: 다른 작업이 이미 실행 중입니다. 현재 작업을 취소 후 다시 시도해주세요.")
            return

        self.progress_bar.setValue(0)
        self.stop_current_worker_event = threading.Event()

        self.current_worker_runnable = OCRLLMWorker(
            task_type=task_type,
            model_name=self.llm_model_name,
            stop_event=self.stop_current_worker_event,
            ocr_reader_instance=self.ocr_reader_instance, # 미리 로드된 인스턴스 전달
            llm_model_instance=self.llm_model_instance,  # 미리 로드된 인스턴스 전달
            **kwargs
        )
        
        self.current_worker_runnable.signals.result.connect(self.handle_worker_result)
        self.current_worker_runnable.signals.progress.connect(self.handle_worker_progress)
        self.current_worker_runnable.signals.finished.connect(self.handle_worker_finished)
        self.current_worker_runnable.signals.error.connect(self.handle_worker_error)
        
        self.threadpool.start(self.current_worker_runnable)
        self.set_buttons_for_task_running_state(True)
        self.update_status_message(f"{task_type.replace('_',' ').title()} 작업 시작...")

    def run_ocr_correction_task(self):
        if not self.current_image_path:
            self.update_status_message("오류: 분석할 이미지가 선택되지 않았습니다.")
            return
        if not self.ocr_reader_instance and not easyocr: 
             QMessageBox.critical(self, "EasyOCR 오류", "EasyOCR 라이브러리가 없거나 로드에 실패하여 분석을 진행할 수 없습니다.")
             return
        if not self.llm_model_instance and not GPT4All: 
             QMessageBox.critical(self, "GPT4All 오류", "GPT4All 라이브러리가 없거나 로드에 실패하여 분석을 진행할 수 없습니다.")
             return
        if not cv2:
            QMessageBox.critical(self, "OpenCV 오류", "OpenCV(cv2) 라이브러리가 없어 이미지 분석을 진행할 수 없습니다.")
            return

        self.raw_ocr_text = "" 
        self.current_corrected_text = ""
        self._start_background_worker("ocr_correct", image_path=self.current_image_path)

    def run_explanation_task(self, text_for_explanation):
        if not text_for_explanation:
            self.update_status_message("오류: 해설할 텍스트가 없습니다.")
            return
        if not self.llm_model_instance and not GPT4All:
             QMessageBox.critical(self, "GPT4All 오류", "GPT4All 라이브러리가 없거나 로드에 실패하여 해설을 진행할 수 없습니다.")
             return
        self._start_background_worker("explain", text_input=text_for_explanation)

    def handle_worker_result(self, task_type, data):
        if task_type == "ocr_result":
            self.raw_ocr_text = data
            self.update_status_message("OCR 추출 완료. LLM으로 정제 중...")
        elif task_type == "corrected_text_result":
            self.current_corrected_text = data
            self.update_status_message("LLM 정제 완료. 결과 확인 및 다음 단계 진행 가능.")
            self.show_comparison_dialog()
        elif task_type == "explanation_result":
            self.show_explanation_dialog(data)
            self.update_status_message("문서 해설 완료.")
        
    def handle_worker_progress(self, task_type, value):
        self.progress_bar.setValue(value)

    def handle_worker_finished(self):
        if self.current_worker_runnable and self.current_worker_runnable.task_type != "corrected_text_result":
             self.progress_bar.setValue(100) 
        
        self.set_buttons_for_task_running_state(False)
        self.current_worker_runnable = None 
        self.stop_current_worker_event = None

    def handle_worker_error(self, error_message):
        self.update_status_message(f"오류 발생: {error_message.splitlines()[0]}") 
        QMessageBox.critical(self, "작업 오류", f"오류가 발생했습니다:\n{error_message}")
        self.progress_bar.setValue(0)
        self.set_buttons_for_task_running_state(False)
        self.current_worker_runnable = None
        self.stop_current_worker_event = None
        self.reset_ui_after_task(full_reset=True)


    def set_buttons_for_task_running_state(self, is_running):
        self.analyze_button.setEnabled(not is_running and bool(self.current_image_path))
        self.cancel_task_button.setEnabled(True) 
        self.select_image_button.setEnabled(not is_running)


    def show_comparison_dialog(self):
        if not self.current_image_path: 
            self.update_status_message("오류: 비교할 이미지가 없습니다.")
            self.reset_ui_after_task(full_reset=True)
            return 

        dialog = ComparisonDialog(self.current_image_path, self.raw_ocr_text, self.current_corrected_text, self)
        dialog_result = dialog.exec()

        if dialog_result == 1: # Redo
            self.update_status_message("재시도 요청됨. 이미지 분석을 다시 시작합니다.")
            self.run_ocr_correction_task()
        elif dialog_result == 2: # Accept
            self.final_accepted_text_for_explanation = dialog.get_accepted_text()
            if self.final_accepted_text_for_explanation and self.final_accepted_text_for_explanation.strip():
                self.update_status_message("정제된 텍스트 수락됨. 문서 해설을 시작합니다.")
                self.run_explanation_task(self.final_accepted_text_for_explanation)
            else:
                QMessageBox.warning(self, "텍스트 없음", "해설을 위한 텍스트가 없습니다. 다시 시도해주세요.")
                self.reset_ui_after_task() 
        else: # Cancel or closed
            self.update_status_message("비교 작업 취소됨. 초기 상태로 돌아갑니다.")
            self.reset_ui_after_task(full_reset=True) 
            
    def show_explanation_dialog(self, explanation_text):
        dialog = ExplanationDialog(explanation_text, self)
        dialog.exec()
        self.reset_ui_after_task(full_reset=True) 

    def cancel_current_worker_if_running(self):
        if self.current_worker_runnable and self.stop_current_worker_event:
            self.update_status_message("현재 작업 중지 시도 중...")
            self.stop_current_worker_event.set()
            self.current_worker_runnable = None 
            self.stop_current_worker_event = None 
            self.set_buttons_for_task_running_state(False)
            self.progress_bar.setValue(0)
            return True
        return False

    def cancel_or_reset_all_processes(self):
        if self.cancel_current_worker_if_running():
             self.update_status_message("진행 중인 작업이 중지되었습니다. 초기화합니다.")
        else:
            self.update_status_message("초기화 완료.")
        self.reset_ui_after_task(full_reset=True)

    def reset_ui_after_task(self, full_reset=False):
        self.progress_bar.setValue(0)
        self.raw_ocr_text = ""
        self.current_corrected_text = ""
        self.final_accepted_text_for_explanation = ""
        
        if full_reset:
            self.current_image_path = None
            self.image_display_label.setText("이미지를 여기에 드래그 앤 드롭 하거나 위 버튼으로 선택하세요.")
            self.image_display_label.setPixmap(QPixmap()) 
            self.analyze_button.setEnabled(False)
            self.update_status_message("초기화 완료. 새 이미지를 선택하세요.")
        else: 
            self.analyze_button.setEnabled(bool(self.current_image_path))
            self.update_status_message("작업 완료. 다음 분석을 진행하거나 새 이미지를 선택하세요.")

        self.set_buttons_for_task_running_state(False)


    def closeEvent(self, event):
        self.update_status_message("애플리케이션 종료 중...")
        self.cancel_current_worker_if_running()
        print("스레드 풀 정리 시도...")
        self.threadpool.waitForDone(2000) 
        print("애플리케이션 종료됨.")
        super().closeEvent(event)

if __name__ == '__main__':
    if not easyocr or not GPT4All or not cv2:
        app_temp = QApplication.instance() 
        if app_temp is None:
            app_temp = QApplication(sys.argv) 
        
        missing_libs = []
        if not easyocr: missing_libs.append("EasyOCR")
        if not GPT4All: missing_libs.append("GPT4All")
        if not cv2: missing_libs.append("OpenCV (cv2)")

        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setWindowTitle("라이브러리 오류")
        error_dialog.setText(f"필수 라이브러리 ({', '.join(missing_libs)})가 설치되지 않았거나 로드에 실패했습니다.")
        error_dialog.setInformativeText("터미널에서 해당 라이브러리를 설치 후 다시 실행해주세요.\n"
                                       "(예: pip install easyocr gpt4all opencv-python)\n"
                                       "이미 설치했다면, 라이브러리 경로 또는 환경 설정을 확인해주세요.")
        error_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        error_dialog.exec()
        sys.exit(1)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

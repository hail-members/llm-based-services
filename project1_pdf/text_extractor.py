
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

def extract_text_from_url(url):
    """URL에서 텍스트 추출"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 논문 사이트별 최적화 필요
        # arXiv, IEEE, ACM 등 주요 논문 사이트 고려
        for script in soup(["script", "style"]):
            script.decompose()
            
        # 본문 추출 시도
        main_content = (soup.find('article') or 
                       soup.find('main') or 
                       soup.find('div', class_='ltx_page_main') or  # arXiv
                       soup.find('div', {'id': 'BodyWrapper'}) or  # IEEE
                       soup.body)
        
        if main_content:
            paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5'])
            text = '\n'.join([para.get_text().strip() for para in paragraphs if para.get_text().strip()])
            return text
        
        return soup.get_text(separator='\n', strip=True)
        
    except requests.exceptions.RequestException as e:
        return f"❌ URL 접근 오류: {str(e)}"
    except Exception as e:
        return f"❌ 텍스트 추출 오류: {str(e)}"

def extract_text_from_pdf(pdf_path):
    """PDF에서 텍스트 추출"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text + "\n"
            except Exception as e:
                print(f"페이지 {page_num + 1} 추출 오류: {e}")
                continue
                
        return text if text.strip() else "❌ PDF에서 텍스트를 추출할 수 없습니다."
        
    except Exception as e:
        return f"❌ PDF 처리 오류: {str(e)}"
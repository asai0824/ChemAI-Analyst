import streamlit as st
import os
import json
import base64
from google import genai
from google.genai import types
import fitz  # PyMuPDF
from PIL import Image
import io

# --- Page Config ---
st.set_page_config(
    page_title="ChemAI Paper Analyst",
    page_icon="🧪",
    layout="wide"
)

# --- Custom CSS for Styling (Matching React App look) ---
st.markdown("""
<style>
    .report-header { background-color: #f0fdfa; padding: 20px; border-radius: 10px; border-bottom: 2px solid #e5e7eb; margin-bottom: 20px; }
    .report-title { color: #111827; font-family: 'Noto Serif JP', serif; font-weight: bold; font-size: 2em; }
    .report-meta { color: #6b7280; font-size: 0.9em; }
    .section-header { color: #0f766e; border-bottom: 2px solid #ccfbf1; padding-bottom: 5px; margin-top: 30px; margin-bottom: 15px; font-weight: bold; font-size: 1.2em; }
    .summary-box { background-color: #f9fafb; padding: 15px; border-left: 5px solid #2dd4bf; margin-bottom: 20px; }
    .figure-box { border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 20px; background-color: white; }
    .novelty-box { background-color: #eff6ff; padding: 15px; border-left: 5px solid #3b82f6; }
    .author-box { background-color: #eef2ff; padding: 15px; border-radius: 8px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# --- Types & Schema (Mirroring types.ts) ---
# Gemini SDK for Python uses dictionaries for schema
ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title_en": {"type": "STRING", "description": "The original English title."},
        "title_jp": {"type": "STRING", "description": "Japanese translation of the title."},
        "journal_authors": {"type": "STRING", "description": "Journal name and author list."},
        "publication_year": {"type": "STRING", "description": "Year of publication."},
        "background_objective": {"type": "STRING", "description": "Research background and objective in Japanese."},
        "results_summary": {"type": "STRING", "description": "Comprehensive summary of results/discussion in Japanese."},
        "results_figures": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING", "description": "e.g., Figure 1"},
                    "explanation": {"type": "STRING", "description": "Detailed explanation in Japanese."},
                    "page_number": {"type": "INTEGER", "description": "1-based page number."},
                    "bbox": {
                        "type": "ARRAY",
                        "items": {"type": "INTEGER"},
                        "description": "[ymin, xmin, ymax, xmax] 0-1000 scale"
                    }
                },
                "required": ["label", "explanation", "page_number", "bbox"]
            }
        },
        "novelty": {"type": "STRING", "description": "Novelty and significance in Japanese."},
        "conclusion_tasks": {"type": "STRING", "description": "Conclusion and future tasks in Japanese."}
    },
    "required": ["title_en", "title_jp", "journal_authors", "publication_year", "background_objective", "results_summary", "results_figures", "novelty", "conclusion_tasks"]
}

# --- Helper Functions ---

def init_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    # Streamlit Cloud uses st.secrets, usually mapped to env vars, but let's check st.secrets too
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def analyze_pdf_with_gemini(client, file_bytes):
    system_instruction = """
    あなたは優秀な化学者です。英語の化学論文(PDF)を深く読み込み、日本の研究者が理解しやすいように高度な要約を作成してください。
    情報は省略せず、詳細に記述してください。特に発行年(Publication Year)を特定し、
    Figure/Table/Schemeの位置情報(page_number, bbox)を正確に抽出してください。
    bboxは[ymin, xmin, ymax, xmax] (0-1000スケール)です。
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'),
                        types.Part.from_text(text="この論文を解析し、JSON形式で出力してください。")
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ANALYSIS_SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_budget=10240)
            )
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Analysis Error: {str(e)}")
        return None

def extract_images_from_pdf(file_bytes, analysis_data):
    """PyMuPDFを使ってbboxに基づき画像を切り出す"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    enriched_figures = []
    
    for fig in analysis_data.get("results_figures", []):
        try:
            page_num = fig.get("page_number", 1) - 1
            if page_num < 0 or page_num >= len(doc):
                enriched_figures.append(fig)
                continue
                
            page = doc[page_num]
            rect = page.rect  # Page size
            bbox = fig.get("bbox", [])
            
            if len(bbox) == 4:
                # Convert 0-1000 scale to actual PDF coordinates
                # bbox from Gemini is [ymin, xmin, ymax, xmax]
                ymin, xmin, ymax, xmax = bbox
                
                # Add padding
                padding = 20
                h = rect.height
                w = rect.width
                
                y1 = max(0, (ymin - padding) / 1000 * h)
                x1 = max(0, (xmin - padding) / 1000 * w)
                y2 = min(h, (ymax + padding) / 1000 * h)
                x2 = min(w, (xmax + padding) / 1000 * w)
                
                clip_rect = fitz.Rect(x1, y1, x2, y2)
                pix = page.get_pixmap(clip=clip_rect, dpi=200)
                
                # Convert to PIL Image for Streamlit
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # Store object for display (cannot JSON serialize PIL image easily)
                fig["pil_image"] = image
                
        except Exception as e:
            print(f"Error extracting image: {e}")
        
        enriched_figures.append(fig)
        
    analysis_data["results_figures"] = enriched_figures
    return analysis_data

def search_authors(client, authors, title):
    prompt = f"""
    以下の論文の著者、あるいは研究グループについてWeb検索を行い、彼らの過去の研究背景や、今回の論文との関連性を日本語で簡潔にまとめてください。
    論文タイトル: {title}
    著者情報: {authors}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        
        source_urls = []
        if response.candidates[0].grounding_metadata.grounding_chunks:
            for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                if chunk.web and chunk.web.uri:
                    source_urls.append(chunk.web.uri)
                    
        return {
            "summary": response.text,
            "source_urls": list(set(source_urls))
        }
    except Exception as e:
        return {"summary": f"Error: {str(e)}", "source_urls": []}

# --- Auth Logic ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def check_password():
    password = st.session_state.password_input
    # Set your password here or use env var
    correct_password = os.environ.get("ACCESS_PASSWORD", "chem2025")
    if password == correct_password:
        st.session_state.authenticated = True
    else:
        st.error("パスワードが間違っています")

# --- UI: Login Screen ---
if not st.session_state.authenticated:
    st.markdown("<div style='text-align: center; margin-top: 50px;'>", unsafe_allow_html=True)
    st.title("🔒 ChemAI Analyst Login")
    st.text_input("アクセスパスワードを入力してください", type="password", key="password_input", on_change=check_password)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- UI: Main App ---
st.title("🧪 ChemAI Paper Analyst")
st.caption("Powered by Gemini 3.0 Flash")

client = init_gemini_client()

if not client:
    st.warning("⚠️ API Keyが設定されていません。GitHubのSecretsまたは環境変数 `GEMINI_API_KEY` を設定してください。")
    st.stop()

uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

if uploaded_file is not None:
    # Button to start analysis
    if st.button("論文を解析する", type="primary"):
        with st.spinner("Gemini 3.0 Flash が論文を読んでいます... (これには数分かかる場合があります)"):
            file_bytes = uploaded_file.read()
            raw_analysis = analyze_pdf_with_gemini(client, file_bytes)
            
            if raw_analysis:
                with st.spinner("図表を切り出しています..."):
                    final_analysis = extract_images_from_pdf(file_bytes, raw_analysis)
                    st.session_state.analysis_result = final_analysis
                st.rerun()

# --- Display Results ---
result = st.session_state.analysis_result

if result:
    # Header
    st.markdown(f"""
    <div class="report-header">
        <div class="report-meta">Chemistry Research Summary | {result.get('publication_year', 'N/A')}</div>
        <div class="report-title">{result['title_jp']}</div>
        <div style="font-size: 1.1em; color: #4b5563; margin-top: 5px;">{result['title_en']}</div>
        <div style="margin-top: 15px; font-size: 0.9em;">📖 {result['journal_authors']}</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Background
    st.markdown('<div class="section-header">1. 目的・動機・研究背景</div>', unsafe_allow_html=True)
    st.write(result['background_objective'])

    # 2. Results
    st.markdown('<div class="section-header">2. 実験結果・考察</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="summary-box">
        <strong>全体要約:</strong><br>
        {result['results_summary']}
    </div>
    """, unsafe_allow_html=True)

    for fig in result['results_figures']:
        st.markdown(f"**{fig['label']}** (Page {fig['page_number']})")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if "pil_image" in fig:
                st.image(fig["pil_image"], use_container_width=True)
            else:
                st.info("画像なし")
        with col2:
            st.write(fig['explanation'])
        st.divider()

    # 3. Novelty
    st.markdown('<div class="section-header">3. 新規性・学術的意義</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="novelty-box">
        {result['novelty']}
    </div>
    """, unsafe_allow_html=True)

    # 4. Conclusion
    st.markdown('<div class="section-header">4. 結論・今後の課題</div>', unsafe_allow_html=True)
    st.write(result['conclusion_tasks'])

    # 5. Author Search (On Demand)
    st.markdown('<div class="section-header">5. 著者・研究室の背景情報</div>', unsafe_allow_html=True)
    
    if 'author_info' not in st.session_state:
        st.session_state.author_info = None

    if st.button("著者情報をWeb検索する (Gemini 2.5)"):
        with st.spinner("著者を調査中..."):
            author_info = search_authors(client, result['journal_authors'], result['title_en'])
            st.session_state.author_info = author_info

    if st.session_state.author_info:
        info = st.session_state.author_info
        st.markdown(f"""
        <div class="author-box">
            {info['summary']}
        </div>
        """, unsafe_allow_html=True)
        if info['source_urls']:
            st.caption("参照ソース:")
            for url in info['source_urls']:
                st.markdown(f"- [{url}]({url})")

    # Reset Button
    if st.button("別の論文を解析する"):
        st.session_state.analysis_result = None
        st.session_state.author_info = None
        st.rerun()

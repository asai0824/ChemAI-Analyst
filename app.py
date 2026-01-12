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

# --- Custom CSS for Styling ---
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
    "type": types.Type.OBJECT,
    "properties": {
        "title_en": {"type": types.Type.STRING, "description": "The original English title of the paper."},
        "title_jp": {"type": types.Type.STRING, "description": "The Japanese translation of the title."},
        "journal_authors": {"type": types.Type.STRING, "description": "Journal name and list of authors."},
        "publication_year": {"type": types.Type.STRING, "description": "The year of publication."},
        "background_objective": {"type": types.Type.STRING, "description": "Summary of objective and background in Japanese."},
        "results_summary": {"type": types.Type.STRING, "description": "Comprehensive summary of results in Japanese."},
        "results_figures": {
            "type": types.Type.ARRAY,
            "description": "Explanations for figures/tables with bounding boxes.",
            "items": {
                "type": types.Type.OBJECT,
                "properties": {
                    "label": {"type": types.Type.STRING},
                    "explanation": {"type": types.Type.STRING},
                    "page_number": {"type": types.Type.INTEGER},
                    "bbox": {
                        "type": types.Type.ARRAY,
                        "items": {"type": types.Type.INTEGER}
                    }
                },
                "required": ["label", "explanation", "page_number", "bbox"]
            }
        },
        "novelty": {"type": types.Type.STRING, "description": "Novelty in Japanese."},
        "conclusion_tasks": {"type": types.Type.STRING, "description": "Conclusion and tasks in Japanese."}
    },
    "required": ["title_en", "title_jp", "journal_authors", "publication_year", "background_objective", "results_summary", "results_figures", "novelty", "conclusion_tasks"]
}

# --- Helper Functions ---

def get_api_key():
    """Retrieve API Key from sidebar or secrets"""
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return st.sidebar.text_input("Gemini API Key", type="password")

def analyze_pdf_with_gemini(api_key, pdf_bytes):
    """Calls Gemini API to analyze the PDF"""
    client = genai.Client(api_key=api_key)
    
    # PDF bytes to base64 for inline data
    # (Note: Python SDK handles bytes directly usually, but explicit inline data is safe)
    # The SDK's generate_content can take 'bytes' directly if mimetype is specified
    
    system_instruction = """
    あなたは優秀な化学者です。英語の化学論文(PDF)を深く読み込み、日本の研究者が理解しやすいように高度な要約を作成してください。
    (以下、React版と同じプロンプトの要件を記述...)
    1. タイトル(日英)、著者、雑誌名を特定。
    2. 【重要】発行年(Publication Year)を特定。
    3. 目的・動機・背景。
    4. 実験結果・考察 (全体要約 + 図表ごとの詳細解説)。
    5. 図表の位置情報(page_number, bbox [ymin, xmin, ymax, xmax] 0-1000 scale)を正確に特定。
    6. 新規性と学術的意義。
    7. 結論と課題。
    フォーマット: JSON出力。数式はLaTeX ($...$)。
    """

    try:
        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
                "この化学論文を解析し、指定されたJSONスキーマに従って詳細な要約を作成してください。"
            ],
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
                "response_schema": ANALYSIS_SCHEMA,
                # Python SDK uses snake_case for config
                "thinking_config": {"thinking_budget": 16384} 
            }
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Gemini Analysis Error: {e}")
        return None

def extract_images_from_pdf(pdf_bytes, analysis_data):
    """Extracts images using PyMuPDF based on Gemini's bbox"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_figures = []

    for fig in analysis_data.get("results_figures", []):
        page_num = fig.get("page_number", 1) - 1 # 1-based to 0-based
        bbox = fig.get("bbox", [])
        
        image_data = None
        
        if 0 <= page_num < len(doc) and len(bbox) == 4:
            page = doc[page_num]
            # bbox is [ymin, xmin, ymax, xmax] in 0-1000 scale
            # PyMuPDF expects [xmin, ymin, xmax, ymax] in points
            
            ymin, xmin, ymax, xmax = bbox
            h, w = page.rect.height, page.rect.width
            
            # Convert 0-1000 scale to page points
            # Add padding (20/1000 = 2%)
            padding = 20
            r_xmin = max(0, (xmin - padding) / 1000 * w)
            r_ymin = max(0, (ymin - padding) / 1000 * h)
            r_xmax = min(w, (xmax + padding) / 1000 * w)
            r_ymax = min(h, (ymax + padding) / 1000 * h)
            
            # Define crop rectangle
            rect = fitz.Rect(r_xmin, r_ymin, r_xmax, r_ymax)
            
            # Get pixmap (image) of the cropped area
            # matrix=fitz.Matrix(3, 3) for high resolution (similar to scale: 3.0 in React)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect)
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(pix.tobytes()))
            image_data = img
            
        extracted_figures.append({
            "data": fig,
            "image": image_data
        })
    
    return extracted_figures

def search_authors_web(api_key, authors, title):
    """Performs Google Search for author background"""
    client = genai.Client(api_key=api_key)
    prompt = f"以下の論文の著者、研究グループについてWeb検索を行い、研究背景や関連性を日本語でまとめてください。\nタイトル: {title}\n著者: {authors}"
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                "tools": [{"google_search": {}}]
            }
        )
        # Extract URLs from grounding metadata (Python SDK structure might vary slightly, generic approach)
        urls = []
        if response.candidates[0].grounding_metadata and response.candidates[0].grounding_metadata.grounding_chunks:
            for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                if chunk.web and chunk.web.uri:
                    urls.append(chunk.web.uri)
        
        return {"summary": response.text, "urls": list(set(urls))}
    except Exception as e:
        st.error(f"Search Error: {e}")
        return None

# --- Main App Interface ---

def main():
    st.sidebar.title("🧪 ChemAI Analyst")
    api_key = get_api_key()
    
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar to proceed.")
        return

    # File Upload
    uploaded_file = st.file_uploader("Upload Chemistry Paper (PDF)", type="pdf")

    if uploaded_file is not None:
        # Check if we need to re-analyze (new file uploaded)
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.analysis_result = None
            st.session_state.extracted_images = None
            st.session_state.author_info = None
            st.session_state.last_uploaded_file = uploaded_file.name

        # Analysis Button
        if st.session_state.analysis_result is None:
            if st.button("Start Analysis", type="primary"):
                with st.spinner("Analyzing PDF with Gemini 3.0 Pro... (This may take a minute)"):
                    pdf_bytes = uploaded_file.getvalue()
                    
                    # 1. Analyze Text
                    result = analyze_pdf_with_gemini(api_key, pdf_bytes)
                    
                    if result:
                        st.session_state.analysis_result = result
                        
                        # 2. Extract Images
                        with st.spinner("Extracting Figures & Tables..."):
                            images = extract_images_from_pdf(pdf_bytes, result)
                            st.session_state.extracted_images = images
                        
                        st.rerun() # Refresh to show results
        
        # Display Results
        if st.session_state.analysis_result:
            res = st.session_state.analysis_result
            imgs = st.session_state.extracted_images
            
            # -- HEADER --
            st.markdown(f"""
            <div class="report-header">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="background:#ccfbf1; color:#0f766e; padding:2px 8px; border-radius:10px; font-size:0.8em; font-weight:bold;">Chemistry Research Summary</span>
                    <span style="background:#e5e7eb; color:#374151; padding:2px 8px; border-radius:10px; font-size:0.8em;">📅 {res.get('publication_year', 'N/A')}</span>
                </div>
                <div class="report-title">{res.get('title_jp')}</div>
                <div style="font-size: 1.1em; color: #4b5563; margin-bottom: 10px;">{res.get('title_en')}</div>
                <div class="report-meta">📖 {res.get('journal_authors')}</div>
            </div>
            """, unsafe_allow_html=True)

            # -- 1. Background --
            st.markdown('<div class="section-header">1. 目的・動機・研究背景</div>', unsafe_allow_html=True)
            st.write(res.get('background_objective'))

            # -- 2. Results --
            st.markdown('<div class="section-header">2. 実験結果・考察</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="summary-box"><strong>全体要約:</strong><br>' + res.get('results_summary').replace('\n', '<br>') + '</div>', unsafe_allow_html=True)

            # Figures Loop
            if imgs:
                for item in imgs:
                    data = item['data']
                    image = item['image']
                    
                    with st.container():
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            if image:
                                st.image(image, caption=f"Page {data.get('page_number')}", use_container_width=True)
                            else:
                                st.markdown("*No Image Extracted*")
                        
                        with col2:
                            st.markdown(f"**{data.get('label')}**")
                            st.write(data.get('explanation'))
                        
                        st.divider()

            # -- 3. Novelty --
            st.markdown('<div class="section-header">3. 新規性・学術的意義</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="novelty-box">{res.get("novelty")}</div>', unsafe_allow_html=True)

            # -- 4. Conclusion --
            st.markdown('<div class="section-header">4. 結論・今後の課題</div>', unsafe_allow_html=True)
            st.write(res.get('conclusion_tasks'))

            # -- Author Search Section --
            st.markdown('<div class="section-header">5. 著者・研究室情報</div>', unsafe_allow_html=True)
            
            if st.session_state.author_info is None:
                if st.button("著者情報をWeb検索する"):
                    with st.spinner("Searching..."):
                        search_res = search_authors_web(api_key, res.get('journal_authors'), res.get('title_en'))
                        st.session_state.author_info = search_res
                        st.rerun()
            else:
                info = st.session_state.author_info
                st.markdown(f"""
                <div class="author-box">
                    {info.get('summary')}
                    <hr style="margin: 10px 0; border-top: 1px solid #c7d2fe;">
                    <small>Sources:</small>
                    <ul>
                        {''.join([f'<li><a href="{url}">{url}</a></li>' for url in info.get('urls', [])])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # -- HTML Export (for OneNote) --
            # PythonでHTML文字列を構築してダウンロードさせる
            html_content = f"""
            <html><body>
            <h1>{res.get('title_jp')}</h1>
            <h2>{res.get('title_en')}</h2>
            <p><strong>{res.get('journal_authors')}</strong> ({res.get('publication_year')})</p>
            <h3>1. 背景</h3><p>{res.get('background_objective')}</p>
            <h3>2. 結果</h3><p>{res.get('results_summary')}</p>
            {''.join([f"<h4>{fig['data']['label']}</h4><p>{fig['data']['explanation']}</p>" for fig in imgs])}
            <h3>3. 新規性</h3><p>{res.get('novelty')}</p>
            <h3>4. 結論</h3><p>{res.get('conclusion_tasks')}</p>
            </body></html>
            """
            st.download_button(
                label="Download Report (HTML)",
                data=html_content,
                file_name="report.html",
                mime="text/html"
            )

if __name__ == "__main__":
    main()
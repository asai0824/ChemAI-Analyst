import streamlit as st
import os
import json
import base64
import random
import html as html_lib
import re
from google import genai
from google.genai import types
import fitz  # PyMuPDF
from PIL import Image
import io
import streamlit.components.v1 as components

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
</style>
""", unsafe_allow_html=True)

# --- Types & Schema ---
ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title_en": {"type": "STRING", "description": "The original English title."},
        "title_jp": {"type": "STRING", "description": "Japanese translation of the title."},
        "journal_authors": {"type": "STRING", "description": "Journal name and author list."},
        "publication_year": {"type": "STRING", "description": "Year of publication."},
        "background_objective": {"type": "STRING", "description": "Research background and objective in Japanese."},
        "results_summary": {"type": "STRING", "description": "Comprehensive summary of results/discussion in Japanese. Must logically connect the experimental data."},
        "results_figures": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING", "description": "e.g., Figure 1"},
                    "explanation": {"type": "STRING", "description": "Exhaustive Japanese explanation covering ALL discussions of this Figure/Table/Scheme in the paper. Include specific numerical values, comparison results, reaction conditions, and the authors' interpretations. Do NOT summarize briefly - translate the relevant paper text almost verbatim into Japanese."},
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

def get_api_key():
    """
    Retrieves a random API key from a pool to distribute load.
    Supports 'GEMINI_API_KEYS' (comma-separated list) or single 'GEMINI_API_KEY'.
    """
    keys = []
    
    # 1. Check Streamlit Secrets for list or comma-separated string
    if "GEMINI_API_KEYS" in st.secrets:
        secret_keys = st.secrets["GEMINI_API_KEYS"]
        if isinstance(secret_keys, list):
            keys.extend(secret_keys)
        elif isinstance(secret_keys, str):
            keys.extend([k.strip() for k in secret_keys.split(",") if k.strip()])

    # 2. Check Environment Variable
    env_keys = os.environ.get("GEMINI_API_KEYS")
    if env_keys:
        keys.extend([k.strip() for k in env_keys.split(",") if k.strip()])

    # 3. Fallback to single key if no list found
    if not keys:
        single_key = os.environ.get("GEMINI_API_KEY")
        if not single_key and "GEMINI_API_KEY" in st.secrets:
            single_key = st.secrets["GEMINI_API_KEY"]
        if single_key:
            keys.append(single_key)

    if not keys:
        return None
    
    # Return a random key from the pool
    return random.choice(keys)

def analyze_pdf_with_gemini(api_key, file_bytes):
    if not api_key:
        raise ValueError("API Key not found.")
        
    client = genai.Client(api_key=api_key)

    system_instruction = """
    あなたは優秀な化学者です。英語の化学論文(PDF)を深く読み込み、日本の研究者が理解しやすいように高度な要約を作成してください。
    
    以下の点を重視し、情報は省略せず、論理的なつながりを意識して詳細に記述してください:
    1. タイトルは英語と日本語の両方を出力。
    2. 雑誌名と著者を特定。
    3. 【重要】論文の発行年(Publication Year)を必ず特定してください。
    4. 目的・動機・背景を明確に。
    5. 「実験結果・考察」は特に深く分析してください:
         - まず、実験の流れ、条件、主要な発見を含む包括的な要約記述 (results_summary)。ここで図表(Figure, Table, Scheme等)の番号を参照しながら、なぜその実験を行ったのか、結果から何が言えるのかを論理的に説明してください。
         - その後、個々の図・表・スキーム (Figure, Table, Scheme) についての詳細な解説と、PDF内での位置情報 (results_figures)。
    6. 【最重要】results_figuresの各explanationは、論文中でその図表について言及・議論されている内容を**省略せず網羅的に**記述してください:
         - 論文本文中でその図表を参照している箇所の説明を漏れなく含めること。
         - 具体的な数値（反応収率、選択性、温度、時間、濃度など）は必ず記載すること。
         - 比較実験の結果（エントリー間の違い、条件変更による効果など）を詳細に記述すること。
         - Tableの場合は、主要なエントリーの結果を具体的に言及すること。
         - Schemeの場合は、反応の各ステップ・条件・試薬を記述すること。
         - 著者の考察・解釈（なぜその結果になったか、何を示唆するか）も含めること。
         - 短い要約ではなく、論文の該当箇所をほぼそのまま日本語に翻訳する水準の詳細さを目指してください。
    7. 図表やスキームの位置情報(page_number, bbox)は、画像を切り出すために非常に重要ですので、正確に指定してください。bboxは[ymin, xmin, ymax, xmax] (0-1000スケール)です。
    8. 新規性と学術的な面白さを化学者の視点で深く評価。
    9. 結論と残された課題。
    
    出力はJSON形式で行ってください。
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
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
                # Thinking budget increased to 16000 for deeper analysis
                thinking_config=types.ThinkingConfig(thinking_budget=16000)
            )
        )
        return json.loads(response.text)
    except Exception as e:
        # Rethrow to be caught by the caller
        raise e

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

def format_text(text):
    """Simple text formatter for HTML output"""
    if not text:
        return ""
    # Escape HTML special characters
    safe = html_lib.escape(text)
    # Convert newlines to breaks
    safe = safe.replace("\n", "<br>")
    # Convert simple bold **text** to <strong>text</strong>
    safe = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', safe)
    return safe

def generate_html_for_clipboard(result):
    """
    Generates a complete HTML string with inline styles and base64 images
    suitable for pasting into OneNote/Word.
    """
    html = f"""
    <div style="color: #1f2937; max-width: 800px;">
        <h1 style="font-size: 24px; font-weight: bold; color: #111827; margin-bottom: 8px;">{format_text(result['title_jp'])}</h1>
        <h2 style="font-size: 18px; color: #4b5563; margin-bottom: 8px;">{format_text(result['title_en'])}</h2>
        <div style="margin-bottom: 24px; color: #6b7280; font-size: 14px; border-bottom: 1px solid #e5e7eb; padding-bottom: 12px;">
            <span style="font-weight: bold;">{format_text(result['journal_authors'])}</span> | <span>{format_text(result.get('publication_year', 'N/A'))}</span>
        </div>

        <h3 style="font-size: 18px; font-weight: bold; color: #0f766e; border-bottom: 2px solid #ccfbf1; padding-bottom: 6px; margin-top: 24px; margin-bottom: 12px;">1. 目的・動機・研究背景</h3>
        <p style="line-height: 1.6; margin-bottom: 16px;">{format_text(result['background_objective'])}</p>

        <h3 style="font-size: 18px; font-weight: bold; color: #0f766e; border-bottom: 2px solid #ccfbf1; padding-bottom: 6px; margin-top: 24px; margin-bottom: 12px;">2. 実験結果・考察</h3>
        <div style="background-color: #f9fafb; padding: 16px; border-left: 4px solid #2dd4bf; margin-bottom: 24px;">
            <strong style="display: block; margin-bottom: 8px; color: #374151;">全体要約:</strong>
            <p style="line-height: 1.6; margin: 0;">{format_text(result['results_summary'])}</p>
        </div>
    """
    
    for fig in result['results_figures']:
        img_html = ""
        if "pil_image" in fig:
            # Convert PIL image to base64 for embedding in HTML
            buffered = io.BytesIO()
            fig["pil_image"].save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            img_html = f'<div style="text-align: center; margin-bottom: 16px;"><img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto; display: block; margin: 0 auto; max-height: 500px;" /></div>'
        
        html += f"""
        <div style="margin-bottom: 32px; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; background-color: #fff;">
            <p style="font-weight: bold; color: #334155; margin-bottom: 12px; font-size: 16px;">{format_text(fig['label'])} (Page {fig.get('page_number', '?')})</p>
            {img_html}
            <p style="line-height: 1.6; color: #374151;">{format_text(fig['explanation'])}</p>
        </div>
        """
        
    html += f"""
        <h3 style="font-size: 18px; font-weight: bold; color: #0f766e; border-bottom: 2px solid #ccfbf1; padding-bottom: 6px; margin-top: 24px; margin-bottom: 12px;">3. 新規性・学術的意義</h3>
        <div style="line-height: 1.6; margin-bottom: 16px; background-color: #eff6ff; padding: 12px; border-left: 4px solid #3b82f6;">{format_text(result['novelty'])}</div>

        <h3 style="font-size: 18px; font-weight: bold; color: #0f766e; border-bottom: 2px solid #ccfbf1; padding-bottom: 6px; margin-top: 24px; margin-bottom: 12px;">4. 結論・今後の課題</h3>
        <p style="line-height: 1.6; margin-bottom: 16px;">{format_text(result['conclusion_tasks'])}</p>
    </div>
    """
    return html

# --- Auth Logic ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def get_access_password():
    """Streamlit SecretsまたはENVからパスワードを取得（ハードコード禁止）"""
    if "ACCESS_PASSWORD" in st.secrets:
        return st.secrets["ACCESS_PASSWORD"]
    return os.environ.get("ACCESS_PASSWORD")

def check_password():
    password = st.session_state.password_input
    correct_password = get_access_password()
    if not correct_password:
        st.error("ACCESS_PASSWORD が設定されていません。管理者に連絡してください。")
        return
    if password == correct_password:
        st.session_state.authenticated = True
    else:
        st.error("パスワードが間違っています")

# --- UI: Login Screen ---
if not st.session_state.authenticated:
    if not get_access_password():
        st.error("ACCESS_PASSWORD が Secrets または環境変数に設定されていません。デプロイ設定を確認してください。")
        st.stop()
    st.markdown("<div style='text-align: center; margin-top: 50px;'>", unsafe_allow_html=True)
    st.title("🔒 ChemAI Analyst Login")
    st.text_input("アクセスパスワードを入力してください", type="password", key="password_input", on_change=check_password)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- UI: Main App ---
st.title("🧪 ChemAI Paper Analyst")
st.caption("Powered by Gemini 3.0 Pro (Multi-Key Load Balancing)")

# Check if at least one key exists
test_key = get_api_key()
if not test_key:
    st.warning("⚠️ API Keyが設定されていません。`GEMINI_API_KEYS` (カンマ区切り) または `GEMINI_API_KEY` を設定してください。")
    st.stop()

uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

if uploaded_file is not None:
    # Button to start analysis
    if st.button("論文を解析する (Deep Analysis)", type="primary"):
        # Select a key specifically for this request
        current_api_key = get_api_key()
        
        with st.spinner("Gemini 3.0 Pro が論文を深く読み込んでいます... (思考中...)"):
            file_bytes = uploaded_file.read()
            try:
                raw_analysis = analyze_pdf_with_gemini(current_api_key, file_bytes)
                
                if raw_analysis:
                    with st.spinner("図表を切り出しています..."):
                        final_analysis = extract_images_from_pdf(file_bytes, raw_analysis)
                        st.session_state.analysis_result = final_analysis
                    st.rerun()
            except Exception as e:
                st.error(f"Analysis Failed: {str(e)}")

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

    # --- Copy Section for OneNote ---
    st.divider()
    st.subheader("📋 OneNote用コピー")
    st.info("以下のボタンを押すと、画像を含むレポート全体をクリップボードにコピーします。OneNoteやWordに貼り付けてください。")
    
    # Generate HTML content
    html_content = generate_html_for_clipboard(result)
    # Serialize to JSON to safely embed in JS string
    html_json = json.dumps(html_content)
    
    # Render Custom JS Button
    components.html(f"""
    <html>
    <head>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    </head>
    <body style="margin: 0; padding: 0;">
        <div style="display: flex; align-items: center;">
            <button id="copyBtn" onclick="copyContent()" style="
                background-color: #0f766e; 
                color: white; 
                border: none; 
                padding: 12px 20px; 
                border-radius: 8px; 
                cursor: pointer; 
                font-family: sans-serif; 
                font-weight: bold;
                font-size: 14px;
                display: flex;
                align-items: center;
                gap: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: background-color 0.2s;
            ">
                <i class="fa-regular fa-clipboard"></i> レポートをコピー
            </button>
            <span id="status" style="margin-left: 15px; color: #0f766e; font-family: sans-serif; font-size: 14px; font-weight: bold;"></span>
        </div>
        <script>
            async function copyContent() {{
                const content = {html_json};
                const btn = document.getElementById('copyBtn');
                const status = document.getElementById('status');
                
                try {{
                    const blob = new Blob([content], {{ type: 'text/html' }});
                    const item = new ClipboardItem({{ 'text/html': blob }});
                    await navigator.clipboard.write([item]);
                    
                    status.innerText = 'コピー成功！OneNoteに貼り付けてください';
                    btn.style.backgroundColor = '#059669';
                    btn.innerHTML = '<i class="fa-solid fa-check"></i> コピー完了';
                    
                    setTimeout(() => {{
                        status.innerText = '';
                        btn.style.backgroundColor = '#0f766e';
                        btn.innerHTML = '<i class="fa-regular fa-clipboard"></i> レポートをコピー';
                    }}, 3000);
                }} catch (err) {{
                    console.error('Failed to copy: ', err);
                    status.innerText = 'エラー: 手動でコピーしてください';
                    status.style.color = '#dc2626';
                }}
            }}
        </script>
    </body>
    </html>
    """, height=70)
    # --------------------

    # Reset Button
    if st.button("別の論文を解析する"):
        st.session_state.analysis_result = None
        st.rerun()

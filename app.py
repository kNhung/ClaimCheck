import os
import tempfile
from datetime import date

import streamlit as st

# Configuration
model_name = os.getenv("FACTCHECKER_MODEL_NAME", "qwen2.5:0.5b")
max_actions = int(os.getenv("FACTCHECKER_MAX_ACTIONS", "2"))

try:
    # Local import from this repo
    from factchecker.factchecker import factcheck
except Exception as e:
    factcheck = None
    _import_error = e


st.set_page_config(page_title="ClaimCheck - Kiểm chứng tin tức", page_icon="🕵️‍♂️", layout="wide")

st.markdown("""
<style>
.stButton > button {
    background-color: #FF894F;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("🕵️‍♂️ ClaimCheck – Kiểm chứng tin tức")
st.markdown(
    """
    Nhập câu cần kiểm chứng và chọn mốc thời gian. Hệ thống sẽ lập kế hoạch tìm kiếm, thu thập bằng chứng, suy luận và đưa ra kết luận.
    """
)


# --- Inputs ---
with st.sidebar:
    st.markdown("### Kiểm chứng")
    claim = st.text_area("Câu cần kiểm chứng", placeholder="Ví dụ: Ông Putin nói Nga sẽ phản ứng mạnh nếu bị Tomahawk tấn công")
    cutoff = st.date_input("Mốc thời gian (ngày)", value=date.today(), format="DD/MM/YYYY")
    run_btn = st.button("Chạy kiểm chứng")
    st.markdown("---")
    st.markdown("### Lịch sử kiểm chứng")


def _format_date(d: date) -> str:
    return d.strftime("%d-%m-%Y")


if run_btn:
    if not claim or len(claim.strip()) == 0:
        st.error("Vui lòng nhập câu claim.")
        st.stop()

    if factcheck is None:
        st.error(f"Không thể import pipeline: {_import_error}")
        st.stop()

    # Handle image upload
    image_path = None
    multimodal = False
    with st.status("Đang lập kế hoạch, thu thập bằng chứng và suy luận...", expanded=True) as status:
        try:
            status.write("Bắt đầu chạy pipeline...")
            selected_model = model_name.strip() if model_name and model_name.strip() else None
            verdict, report_path = factcheck(claim.strip(), _format_date(cutoff), max_actions=max_actions, model_name=selected_model)
            status.update(label="Hoàn tất", state="complete")
        except Exception as e:
            status.update(label="Lỗi khi chạy pipeline", state="error")
            st.exception(e)
            st.stop()
        finally:
            # Clean up temporary image file
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception:
                    pass  # Ignore cleanup errors

    report_dir = os.path.dirname(report_path)
    report_md_path = os.path.join(report_dir, "report.md")
    report_json_path = os.path.join(report_dir, "report.json")
    

    # --- Load artifacts ---
    report_md = None
    report_json = None
    try:
        if os.path.exists(report_md_path):
            with open(report_md_path, "r") as f:
                report_md = f.read()
        if os.path.exists(report_json_path):
            import json
            with open(report_json_path, "r") as f:
                report_json = json.load(f)
    except Exception as e:
        st.warning(f"Không thể đọc file báo cáo: {e}")

    # --- Display Verdict ---
    st.subheader("Kết luận")
    st.metric("Phán quyết", verdict)
    if report_json and report_json.get("judged_verdict"):
        with st.expander("Giải thích chi tiết"):
            st.markdown(report_json["judged_verdict"])
    st.markdown(f"📁 Báo cáo: {report_dir}")

    st.divider()

    # --- Display Evidence ---
    st.subheader("Bằng chứng")
    with st.expander("Xem bằng chứng", expanded=False):
        if report_json and report_json.get("actions"):
            for action_id, info in report_json["actions"].items():
                st.markdown(f"*{action_id}*")
                results = info.get("results") or {}
                for url, item in results.items():
                    summary = item.get("summary")
                    st.markdown(url)
                    st.write(summary)
        else:
            st.info("Chưa có bằng chứng.")
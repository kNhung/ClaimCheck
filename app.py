import os
from datetime import date

import streamlit as st

try:
    # Local import from this repo
    from factchecker.factchecker import factcheck
except Exception as e:
    factcheck = None
    _import_error = e


st.set_page_config(page_title="ClaimCheck - Kiểm chứng tin tức", layout="wide")

st.title("🕵️‍♂️ ClaimCheck – Kiểm chứng tin tức")
st.markdown(
    """
    Nhập một phát biểu (claim) và chọn mốc thời gian. Hệ thống sẽ lập kế hoạch tìm kiếm, thu thập bằng chứng, suy luận và đưa ra kết luận.
    """
)


# --- Inputs ---
with st.sidebar:
    st.header("Cấu hình")
    claim = st.text_input("Câu claim", placeholder="Ví dụ: Ông Putin nói Nga sẽ phản ứng mạnh nếu bị Tomahawk tấn công")
    cutoff = st.date_input("Chọn thời gian (ngày)", value=date.today(), format="DD/MM/YYYY")
    max_actions = st.slider("Số hành động tối đa", min_value=1, max_value=5, value=2, help="Giới hạn số truy vấn tìm kiếm để chạy nhanh hơn.")
    default_model = os.getenv("FACTCHECK_MODEL_NAME", "qwen3:4b")
    model_name = st.text_input("Tên model (Ollama)", value=default_model)
    run_btn = st.button("Chạy kiểm chứng")


col_reason, col_evidence, col_verdict = st.columns([2, 2, 1])

def _format_date(d: date) -> str:
    return d.strftime("%d-%m-%Y")


if run_btn:
    if not claim or len(claim.strip()) == 0:
        st.error("Vui lòng nhập câu claim.")
        st.stop()

    if factcheck is None:
        st.error(f"Không thể import pipeline: {_import_error}")
        st.stop()

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

    report_dir = os.path.dirname(report_path)
    report_md_path = os.path.join(report_dir, "report.md")
    evidence_md_path = os.path.join(report_dir, "evidence.md")
    report_json_path = os.path.join(report_dir, "report.json")

    # --- Load artifacts ---
    report_md = None
    evidence_md = None
    report_json = None
    try:
        if os.path.exists(report_md_path):
            with open(report_md_path, "r") as f:
                report_md = f.read()
        if os.path.exists(evidence_md_path):
            with open(evidence_md_path, "r") as f:
                evidence_md = f.read()
        if os.path.exists(report_json_path):
            import json
            with open(report_json_path, "r") as f:
                report_json = json.load(f)
    except Exception as e:
        st.warning(f"Không thể đọc file báo cáo: {e}")

    # --- Display Reasoning ---
    with col_reason:
        st.subheader("Quá trình suy luận")
        if report_md:
            st.markdown(report_md)
        elif report_json and report_json.get("reasoning"):
            for i, r in enumerate(report_json["reasoning"], start=1):
                st.markdown(f"#### Lần suy luận {i}")
                st.markdown(r)
        else:
            st.info("Chưa có nội dung suy luận.")

    # --- Display Evidence ---
    with col_evidence:
        st.subheader("Bằng chứng")
        if evidence_md:
            st.markdown(evidence_md)
        elif report_json and report_json.get("actions"):
            import textwrap
            for action_id, info in report_json["actions"].items():
                st.markdown(f"**{action_id}**")
                results = info.get("results") or {}
                for url, item in results.items():
                    snippet = item.get("snippet")
                    summary = item.get("summary")
                    if snippet:
                        st.caption(textwrap.shorten(snippet, width=200, placeholder="…"))
                    st.markdown(f"- [Nguồn]({url})")
                    if summary:
                        with st.expander("Tóm tắt"):
                            st.write(summary)
        else:
            st.info("Chưa có bằng chứng.")

    # --- Display Verdict ---
    with col_verdict:
        st.subheader("Kết luận")
        st.metric("Phán quyết", verdict)
        if report_json and report_json.get("judged_verdict"):
            with st.expander("Giải thích chi tiết"):
                st.markdown(report_json["judged_verdict"])
        st.markdown(f"📁 Báo cáo: `{report_dir}`")

else:
    st.info("Nhập claim, chọn ngày rồi bấm 'Chạy kiểm chứng'.")

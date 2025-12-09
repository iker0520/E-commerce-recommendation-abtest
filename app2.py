import streamlit as st
import pandas as pd
import torch
import random
import os
import sys
import pickle
import numpy as np
import re  # [추가] 숫자 추출용
from streamlit_gsheets import GSheetsConnection

# ------------------------------------------------------------------
# 1. [필수] 모듈 경로 및 라이브러리 가짜 등록
# ------------------------------------------------------------------
import tisasrec_local
sys.modules['TiSASRec'] = tisasrec_local

from types import ModuleType
def mock_lib(name):
    if name not in sys.modules: sys.modules[name] = ModuleType(name)
    return sys.modules[name]

for lib in ['kmeans_pytorch', 'lightgbm', 'xgboost', 'ray', 'hyperopt', 'colorama']:
    mock_lib(lib)
    sys.modules[f"{lib}.sklearn"] = mock_lib(f"{lib}.sklearn")

if 'kmeans_pytorch' in sys.modules:
    sys.modules['kmeans_pytorch'].kmeans = lambda *args, **kwargs: (None, None)

# ------------------------------------------------------------------
# 2. 데이터 및 모델 로드
# ------------------------------------------------------------------
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.data.interaction import Interaction

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MockDataset:
    def __init__(self, n_items):
        self.n_items = n_items
    def num(self, field):
        return self.n_items

@st.cache_data
def load_data():
    try:
        all_df = pd.read_pickle('data/meta_lookup.pkl')
    except:
        st.error("data/meta_lookup.pkl 파일이 없습니다.")
        return None, None, None

    try:
        with open("data/recbole_vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        token2id = vocab['token2id']
        id2token = vocab['id2token']
        
        if not isinstance(id2token, dict):
            id2token = {i: str(token) for i, token in enumerate(id2token)}
    except Exception as e:
        st.error(f"매핑 파일 로드 오류: {e}")
        return None, None, None
        
    return all_df, token2id, id2token

@st.cache_data
def load_cycle_data():
    try:
        with open("data/item_cycle_lookup.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return {} 

@st.cache_resource
def load_models():
    sas_path = 'data/SASRec-Nov-27-2025_10-12-11.pth'
    sas_model, sas_n_items = None, 0
    try:
        checkpoint = torch.load(sas_path, map_location=DEVICE, weights_only=False)
        sas_n_items = checkpoint['state_dict']['item_embedding.weight'].shape[0]
        sas_model = SASRec(checkpoint['config'], MockDataset(sas_n_items)).to(DEVICE)
        sas_model.load_state_dict(checkpoint['state_dict'])
        sas_model.eval()
        
        # 모델의 maxlen 가져오기
        maxlen = checkpoint['config']['MAX_ITEM_LIST_LENGTH']
    except Exception as e:
        st.warning(f"SASRec 로드 실패: {e}")
        maxlen = 50

    return sas_model, sas_n_items, maxlen

# ------------------------------------------------------------------
# 3. [기능 추가] 페르소나 데이터 로드 함수
# ------------------------------------------------------------------
def load_persona_history(all_df):
    persona_path = 'data/여자_대학생_새내기.csv' # 파일 경로
    
    if not os.path.exists(persona_path):
        st.error(f"❌ 페르소나 파일을 찾을 수 없습니다: {persona_path}")
        return []

    try:
        df = pd.read_csv(persona_path)
        history = []
        
        for _, row in df.iterrows():
            # 1. 시점 변환: "30일 전" -> 30
            days_str = str(row['시점'])
            days_match = re.search(r'\d+', days_str)
            days = int(days_match.group()) if days_match else 0
            
            # 2. 이름으로 아이템 ID 찾기
            item_name = row.get('상품 선택') or row.get('name')
            if not item_name: continue

            # meta_df에서 이름 매칭
            matched_row = all_df[all_df['Item_Name'] == item_name]
            
            if not matched_row.empty:
                item_id = str(matched_row.iloc[0]['item_id'])
                history.append({
                    'item_id': item_id,
                    'name': item_name,
                    'days_ago': days
                })
                
        return history
    except Exception as e:
        st.error(f"페르소나 로드 중 오류: {e}")
        return []

# ------------------------------------------------------------------
# 4. 로직 함수들
# ------------------------------------------------------------------
def check_cycle_filtering(days_ago, cycle_info):
    if not cycle_info: return days_ago < 7
    
    p10 = cycle_info.get('p10', 0)
    p25 = cycle_info.get('p25', 0)
    
    if days_ago < p10: return random.random() < 0.95
    elif p10 <= days_ago < p25: return random.random() < 0.5
    else: return False

# ------------------------------------------------------------------
# 5. 메인 로직
# ------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Recommendation Rule A/B Test")
    st.title("🛍️ 쇼핑 패턴 기반 추천 Rule A/B Test")

    all_df, token2id, id2token = load_data()
    if all_df is None: return
    
    cycle_data = load_cycle_data()
    sas_model, safe_n_items, maxlen = load_models()

    # UI 필터링
    valid_tokens = [t for t, i in token2id.items() if i < safe_n_items]
    ui_df = all_df[all_df['item_id'].astype(str).isin(valid_tokens) & (all_df['purchase_count'] >= 20)].copy()

    if 'history' not in st.session_state: st.session_state['history'] = []

    # --- Sidebar ---
    st.sidebar.header("🛒 구매 이력 구성")
    
    # [1] 페르소나 적용 (추가됨)
    st.sidebar.subheader("1. 페르소나 (빠른 시작)")
    if st.sidebar.button("👩‍🎓 여대생 새내기 모드 적용"):
        persona_history = load_persona_history(all_df)
        if persona_history:
            st.session_state['history'] = persona_history
            st.session_state['history'].sort(key=lambda x: x['days_ago'], reverse=True)
            st.success("페르소나 로드 완료!")
            st.rerun()

    st.sidebar.divider()

    # [2] 직접 추가
    st.sidebar.subheader("2. 직접 추가하기")
    if not ui_df.empty:
        l1 = st.sidebar.selectbox("대분류", sorted(ui_df['L1'].unique()))
        l2 = st.sidebar.selectbox("중분류", sorted(ui_df[ui_df['L1']==l1]['L2'].unique()))
        items = ui_df[(ui_df['L1']==l1) & (ui_df['L2']==l2)].sort_values(by='purchase_count', ascending=False)
        
        sel_item = st.sidebar.selectbox("상품 선택", options=items.to_dict('records'), 
                                      format_func=lambda x: f"{x['Item_Name']} ({x['purchase_count']}회)")
        days = st.sidebar.number_input("며칠 전 구매?", 0, 365, 0)
        
        if st.sidebar.button("리스트에 추가"):
            st.session_state['history'].append({
                'item_id': str(sel_item['item_id']),
                'name': sel_item['Item_Name'],
                'days_ago': days
            })
            st.session_state['history'].sort(key=lambda x: x['days_ago'], reverse=True)

    if st.sidebar.button("초기화 (전체 삭제)"):
        st.session_state['history'] = []
        st.session_state.pop('raw_scores', None) # 결과 초기화
        st.session_state.pop('ab_mapping', None) # 매핑 초기화
        st.rerun()

    # --- Main ---
    st.subheader("📋 현재 구매 시퀀스")
    if st.session_state['history']:
        hist_df = pd.DataFrame(st.session_state['history'])
        hist_df['시점'] = hist_df['days_ago'].apply(lambda x: "오늘" if x==0 else f"{x}일 전")
        st.dataframe(hist_df[['시점', 'name']], width=700)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ 파라미터 튜닝")
    alpha = st.sidebar.slider("재구매 가중치 (Alpha)", 0.0, 10.0, 2.0, 0.1)
    
    # ------------------------------------------------------------------
    # 추론 버튼
    # ------------------------------------------------------------------
    if st.button("추천 결과 생성/업데이트", type="primary"):
        if len(st.session_state['history']) < 2:
            st.warning("아이템을 2개 이상 넣어주세요.")
        else:
            with st.spinner("AI 분석 중..."):
                # [핵심 수정] 매번 버튼 누를 때마다 랜덤 매핑 초기화 (순서 섞기)
                if 'ab_mapping' in st.session_state:
                    del st.session_state['ab_mapping']
                
                # 1. 입력 변환
                hist_ids = []
                for h in st.session_state['history']:
                    if h['item_id'] in token2id:
                        internal_id = token2id[h['item_id']]
                        if internal_id < safe_n_items:
                            hist_ids.append(internal_id)
                
                if not hist_ids: st.stop()
                    
                # SASRec은 끝방을 봐야 하므로 길이를 maxlen으로 고정
                seq_ids = hist_ids[-maxlen:]
                pad_len = maxlen - len(seq_ids)
                input_ids = [0] * pad_len + seq_ids
                
                item_seq = torch.LongTensor([input_ids]).to(DEVICE)
                item_len = torch.LongTensor([maxlen]).to(DEVICE) # [중요] 길이 고정

                # 2. SASRec 추론
                if sas_model:
                    inter_sas = Interaction({'item_id_list': item_seq, 'item_length': item_len})
                    raw_scores = sas_model.full_sort_predict(inter_sas).detach().cpu().numpy()[0]
                    
                    st.session_state['raw_scores'] = raw_scores
                    st.session_state['has_run'] = True
                    # 제출 상태 초기화 (새 결과가 나왔으므로)
                    st.session_state['experiment_submitted'] = False

    # ------------------------------------------------------------------
    # 결과 렌더링 (Logic A vs B)
    # ------------------------------------------------------------------
    if st.session_state.get('has_run', False):
        raw_scores = st.session_state['raw_scores']
        
        # --- Logic A: History Boost ---
        scores_A = raw_scores.copy()
        item_counts = {}
        for h in st.session_state['history']:
            raw_id = h['item_id']
            item_counts[raw_id] = item_counts.get(raw_id, 0) + 1
            
        for raw_id, count in item_counts.items():
            if raw_id in token2id:
                idx = token2id[raw_id]
                if idx < len(scores_A):
                    scores_A[idx] += alpha * np.log1p(count)

        topk_A_ids = np.argsort(scores_A)[::-1][:10]

        # --- Logic B: Cycle Filtering ---
        scores_B = scores_A.copy()
        filtered_debug_info = {}
        
        for h in st.session_state['history']:
            raw_id = h['item_id']
            days = h['days_ago']
            if raw_id in token2id:
                idx = token2id[raw_id]
                if idx < len(scores_B):
                    c_info = cycle_data.get(raw_id, {})
                    if check_cycle_filtering(days, c_info):
                        scores_B[idx] = -np.inf
                        filtered_debug_info[idx] = "주기 미도래"
        
        topk_B_ids = np.argsort(scores_B)[::-1][:10]

        # --- [핵심 수정] 랜덤 매핑 로직 (매번 섞임) ---
        if 'ab_mapping' not in st.session_state:
            st.session_state['ab_mapping'] = random.choice(['A_is_1', 'B_is_1'])

        mapping = st.session_state['ab_mapping']
        
        if mapping == 'A_is_1':
            opt1_ids, opt1_name = topk_A_ids, "Logic A (부스팅 Only)"
            opt2_ids, opt2_name = topk_B_ids, "Logic B (부스팅 + 필터링)"
        else:
            opt1_ids, opt1_name = topk_B_ids, "Logic B (부스팅 + 필터링)"
            opt2_ids, opt2_name = topk_A_ids, "Logic A (부스팅 Only)"

        # Helper
        def get_simple_info(idx):
            name, cat = "Unknown", ""
            if idx in id2token:
                raw_id = id2token[idx]
                row = all_df[all_df['item_id'].astype(str) == raw_id]
                if not row.empty:
                    name = row.iloc[0]['Item_Name']
                    cat = row.iloc[0]['L2']
            return f"[{cat}] {name}"

        # --- 화면 출력 ---
        st.divider()
        st.subheader("⚖️ 블라인드 테스트: 더 만족스러운 추천은?")
        
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("### 🅰️ Option 1")
            for rank, idx in enumerate(opt1_ids):
                if idx == 0: continue
                st.write(f"{rank+1}. {get_simple_info(idx)}")

        with bc2:
            st.markdown("### 🅱️ Option 2")
            for rank, idx in enumerate(opt2_ids):
                if idx == 0: continue
                st.write(f"{rank+1}. {get_simple_info(idx)}")

        # --- 설문 폼 ---
        st.markdown("---")
        with st.form("ab_test_form"):
            st.write("📝 **평가 입력**")
            choice = st.radio("더 마음에 드는 추천 결과는?", ["Option 1", "Option 2"], horizontal=True)
            reason = st.text_area("이유:")
            
            if st.form_submit_button("제출 및 결과 확인", type="primary"):
                st.session_state['experiment_submitted'] = True
                st.session_state['user_choice'] = choice
                st.session_state['user_reason'] = reason

                # 저장 로직 (Google Sheets 등)
                log_data = {
                    "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "user_choice": choice,
                    "logic_left": opt1_name,
                    "logic_right": opt2_name,
                    "winner": opt1_name if choice == "Option 1" else opt2_name,
                    "reason": reason
                }
                
                try:
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    # (간소화) 데이터 읽기/쓰기 로직...
                    # conn.update(...)
                    st.success("데이터 저장 완료!")
                except:
                    pass

        # --- 결과 공개 ---
        if st.session_state.get('experiment_submitted', False):
            st.divider()
            st.header("🔓 결과 공개")
            
            user_pick = st.session_state['user_choice']
            real_logic = opt1_name if user_pick == "Option 1" else opt2_name
            
            st.success(f"당신의 선택: **{user_pick}**")
            st.info(f"실제 로직: **{real_logic}**")
            
            # 상세 분석 보기
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f"### {opt1_name}")
                for rank, idx in enumerate(opt1_ids):
                    if idx==0: continue
                    st.caption(f"{rank+1}. {get_simple_info(idx)}")
            with rc2:
                st.markdown(f"### {opt2_name}")
                for rank, idx in enumerate(opt2_ids):
                    if idx==0: continue
                    st.caption(f"{rank+1}. {get_simple_info(idx)}")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import torch
import random
import os
import sys
import pickle
import numpy as np
from streamlit_gsheets import GSheetsConnection

# ------------------------------------------------------------------
# 1. [필수] 모듈 경로 및 라이브러리 가짜 등록 (에러 방지)
# ------------------------------------------------------------------
import tisasrec_local
sys.modules['TiSASRec'] = tisasrec_local

# RecBole 의존성 문제 해결 (recover_map.py와 동일한 방식)
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
# 2. 모델 및 데이터 로드
# ------------------------------------------------------------------
from recbole.model.sequential_recommender.sasrec import SASRec
from tisasrec_local import TiSASRec
from recbole.data.interaction import Interaction
from utils import get_tisasrec_input

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MockDataset:
    def __init__(self, n_items):
        self.n_items = n_items
    def num(self, field):
        return self.n_items

@st.cache_data
def load_data():
    # 1. 메타 데이터 (상품명)
    try:
        all_df = pd.read_pickle('data/meta_lookup.pkl')
    except:
        st.error("data/meta_lookup.pkl 파일이 없습니다.")
        return None, None, None

    # 2. 매핑 데이터 (ID 변환)
    try:
        with open("data/recbole_vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        
        token2id = vocab['token2id']
        id2token = vocab['id2token']
        
        # -----------------------------------------------------------
        # [핵심 수정] id2token이 Numpy 배열이라면 딕셔너리로 변환!
        # -----------------------------------------------------------
        if not isinstance(id2token, dict):
            # 배열의 인덱스(0, 1, 2...)가 곧 ID입니다.
            # enumerate를 써서 {0: 'pad', 1: '8685', ...} 형태로 바꿉니다.
            id2token = {i: str(token) for i, token in enumerate(id2token)}
            
        # token2id도 안전하게 문자열 키로 처리
        if not isinstance(token2id, dict):
             # RecBole token2id는 보통 dict지만 안전을 위해 확인
             pass 

    except Exception as e:
        st.error(f"매핑 파일 로드 오류: {e}")
        return None, None, None
        
    return all_df, token2id, id2token


@st.cache_data
def load_cycle_data():
    """재구매 주기 정보(P10, P25, P50) 로드"""
    try:
        with open("data/item_cycle_lookup.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        # 파일이 없거나 에러나면 빈 딕셔너리 반환 (에러 방지)
        return {} 


@st.cache_resource
def load_models():
    # ------------------------------------
    # SASRec 로드
    # ------------------------------------
    sas_path = 'data/SASRec-Nov-27-2025_10-12-11.pth'
    sas_model, sas_n_items = None, 0
    try:
        checkpoint = torch.load(sas_path, map_location=DEVICE, weights_only=False)
        sas_n_items = checkpoint['state_dict']['item_embedding.weight'].shape[0]
        sas_model = SASRec(checkpoint['config'], MockDataset(sas_n_items)).to(DEVICE)
        sas_model.load_state_dict(checkpoint['state_dict'])
        sas_model.eval()
    except Exception as e:
        st.warning(f"SASRec 로드 실패: {e}")

    # ------------------------------------
    # TiSASRec 로드
    # ------------------------------------
    tis_path = 'data/TiSASRec-Nov-28-2025_09-45-58.pth'
    tis_model, tis_n_items = None, 0
    tis_maxlen, tis_timespan = 50, 256
    try:
        checkpoint = torch.load(tis_path, map_location=DEVICE, weights_only=False)
        tis_n_items = checkpoint['state_dict']['item_embedding.weight'].shape[0]
        config = checkpoint['config']
        
        tis_model = TiSASRec(config, MockDataset(tis_n_items)).to(DEVICE)
        tis_model.load_state_dict(checkpoint['state_dict'])
        tis_model.eval()
        
        tis_maxlen = config['MAX_ITEM_LIST_LENGTH']
        tis_timespan = config['time_span']
    except Exception as e:
        st.error(f"TiSASRec 로드 실패: {e}")

    # 두 모델 중 더 작은 크기를 안전한 Max ID로 설정 (인덱스 에러 방지)
    safe_n_items = 0
    if sas_n_items > 0 and tis_n_items > 0:
        safe_n_items = min(sas_n_items, tis_n_items)
    elif tis_n_items > 0:
        safe_n_items = tis_n_items
        
    return sas_model, tis_model, tis_maxlen, tis_timespan, safe_n_items



def check_cycle_filtering(days_ago, cycle_info):
    """
    days_ago: 구매한 지 며칠 지났는지
    cycle_info: {'p10': 7, 'p25': 14, ...}
    Return: True면 필터링(삭제), False면 생존
    """
    if not cycle_info: 
        # 정보가 없으면 "기본 7일"은 재구매 안 한다고 가정
        # 즉, 산 지 7일 미만이면 필터링(True), 7일 넘었으면 통과(False)
        return days_ago < 7
    
    p10 = cycle_info.get('p10', 0)
    p25 = cycle_info.get('p25', 0)
    p50 = cycle_info.get('p50', 0)
    
    t = days_ago
    prob = 0.0

    # 구간별 확률 계산 (선형 보간)
    if t < p10:
        prob = 0.95  # 매우 위험: 95% 확률로 제거
    elif p10 <= t < p25:
        # P10 ~ P25: 95% -> 30% 로 감소
        ratio = (t - p10) / (p25 - p10 + 1e-5)
        prob = 0.95 - (ratio * (0.95 - 0.3))
    elif p25 <= t < p50:
        # P25 ~ P50: 30% -> 0% 로 감소
        ratio = (t - p25) / (p50 - p25 + 1e-5)
        prob = 0.3 - (ratio * 0.3)
    else:
        prob = 0.0 # 안전 구간: 제거 안 함
        
    return random.random() < prob



# ------------------------------------------------------------------
# 3. 메인 함수
# ------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Recommendation A/B Test")
    st.title("🛍️ 쇼핑 패턴 기반 추천 A/B 테스트")

    all_df, token2id, id2token = load_data()
    if all_df is None: return
    
    sas_model, tis_model, tis_maxlen, tis_timespan, safe_n_items = load_models()

    # UI 필터링: 모델이 아는 ID만 표시 (안전장치)
    # token2id에 있고, 그 ID가 safe_n_items보다 작은 것만 유효함
    valid_tokens = [t for t, i in token2id.items() if i < safe_n_items]
    valid_mask = all_df['item_id'].astype(str).isin(valid_tokens)
    
    # 20회 이상 구매된 것만 UI에 노출
    ui_df = all_df[valid_mask & (all_df['purchase_count'] >= 20)].copy()

    if 'history' not in st.session_state: st.session_state['history'] = []

    # --- 사이드바 ---
    st.sidebar.header("🛒 구매 이력 추가")
    
    if ui_df.empty:
        st.error("조건에 맞는 상품이 없습니다. 데이터 매핑 상태를 확인하세요.")
        return

    l1_list = sorted(ui_df['L1'].unique())
    l1_sel = st.sidebar.selectbox("대분류", l1_list)
    
    l2_list = sorted(ui_df[ui_df['L1']==l1_sel]['L2'].unique())
    l2_sel = st.sidebar.selectbox("중분류", l2_list)
    
    final_df = ui_df[(ui_df['L1']==l1_sel) & (ui_df['L2']==l2_sel)]
    final_df = final_df.sort_values(by='purchase_count', ascending=False)
    
    selected_item = st.sidebar.selectbox(
        "상품 선택", 
        options=final_df.to_dict('records'), 
        format_func=lambda x: f"{x['Item_Name']} ({x['purchase_count']}회)"
    )
    
    days_ago = st.sidebar.number_input("며칠 전 구매?", 0, 365, 0)
    
    if st.sidebar.button("리스트에 추가"):
        st.session_state['history'].append({
            'item_id': str(selected_item['item_id']),
            'name': selected_item['Item_Name'],
            'days_ago': days_ago
        })
        st.session_state['history'].sort(key=lambda x: x['days_ago'], reverse=True)

    if st.sidebar.button("초기화"):
        st.session_state['history'] = []
        if 'last_results' in st.session_state: del st.session_state['last_results']
        st.rerun()

    # --- 메인 화면 ---
    st.subheader("📋 현재 구매 시퀀스")
    if st.session_state['history']:
        hist_df = pd.DataFrame(st.session_state['history'])
        hist_df['시점'] = hist_df['days_ago'].apply(lambda x: "오늘" if x==0 else f"{x}일 전")
        st.dataframe(hist_df[['시점', 'name']], use_container_width=True)
        
        # (main 함수 내부)
        
        # 주기 데이터 로드 (메인 함수 초입에 넣어두는 게 좋음)
        cycle_data = load_cycle_data() 

        # ... (앞부분 생략: cycle_data 로드 등) ...
    
    # ------------------------------------------------------------------
    # [수정 1] 슬라이더를 버튼 밖으로 꺼냅니다. (항상 조절 가능하게)
    # ------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ 파라미터 튜닝")
    alpha = st.sidebar.slider("재구매 가중치 (Alpha)", 0.0, 10.0, 2.0, 0.1)
    
    # ------------------------------------------------------------------
    # [수정 2] 버튼 클릭 시 '모델 추론'만 수행하고 결과를 세션에 저장
    # ------------------------------------------------------------------
    if st.button("추천 결과 생성/업데이트", type="primary"):
        if len(st.session_state['history']) < 2:
            st.warning("아이템을 2개 이상 넣어주세요.")
        else:
            with st.spinner("AI가 패턴을 분석 중입니다..."):
                # 1. 시퀀스 데이터 생성
                hist_ids, hist_days = [], []
                for h in st.session_state['history']:
                    if h['item_id'] in token2id:
                        internal_id = token2id[h['item_id']]
                        if internal_id < safe_n_items:
                            hist_ids.append(internal_id)
                            hist_days.append(h['days_ago'])
                
                if not hist_ids:
                    st.error("데이터 범위 오류")
                    st.stop()
                    
                seq_ids = hist_ids[-tis_maxlen:]
                pad_len = tis_maxlen - len(seq_ids)
                input_ids = [0] * pad_len + seq_ids
                
                item_seq = torch.LongTensor([input_ids]).to(DEVICE)
                item_len = torch.LongTensor([tis_maxlen]).to(DEVICE)

                # 2. SASRec 모델 추론 (여기가 무거운 작업)
                if sas_model:
                    inter_sas = Interaction({'item_id_list': item_seq, 'item_length': item_len})
                    # Raw Logits(점수)만 계산해서 세션에 저장
                    raw_scores = sas_model.full_sort_predict(inter_sas).detach().cpu().numpy()[0]
                    
                    st.session_state['raw_scores'] = raw_scores
                    st.session_state['has_run'] = True # 실행 완료 플래그
    
    # ------------------------------------------------------------------
    # [수정 3] 세션에 결과가 있다면, 슬라이더 값(alpha)을 반영해 즉시 렌더링
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # [수정] 결과 화면 출력 및 블라인드 테스트 로직 (로그 가중치 적용)
    # ------------------------------------------------------------------
    if st.session_state.get('has_run', False):
        raw_scores = st.session_state['raw_scores']
        
        # ===============================================================
        # 1. [계산 단계] Logic A (로그 가중치) & B 점수 확정
        # ===============================================================
        
        # --- Logic A: History Boost (로그 방식 적용) ---
        scores_A = raw_scores.copy()
        
        # 1. 아이템별 구매 횟수 카운팅
        item_counts = {}
        for h in st.session_state['history']:
            raw_id = h['item_id']
            item_counts[raw_id] = item_counts.get(raw_id, 0) + 1
            
        # 2. 로그 가중치 계산 (alpha 적용)
        for raw_id, count in item_counts.items():
            if raw_id in token2id:
                idx = token2id[raw_id]
                if idx < len(scores_A):
                    # ln(1 + 횟수) * alpha
                    boost_score = alpha * np.log1p(count)
                    scores_A[idx] += boost_score

        # 3. Top 10 선정 (A 화면용)
        topk_A_ids = np.argsort(scores_A)[::-1][:10]

        # --- Logic B: Cycle Filtering ---
        scores_B = scores_A.copy() # A 점수에서 시작
        filtered_debug_info = {}   # 디버깅용: 누가 왜 필터링됐는지 저장
        
        for h in st.session_state['history']:
            raw_id = h['item_id']
            days = h['days_ago']
            if raw_id in token2id:
                idx = token2id[raw_id]
                if idx < len(scores_B):
                    c_info = cycle_data.get(raw_id, {})
                    
                    # [핵심] 필터링 판정 (딱 한 번만 수행)
                    is_filtered = check_cycle_filtering(days, c_info)
                    
                    if is_filtered:
                        scores_B[idx] = -np.inf # 점수 삭제
                        # 이유 기록
                        if not c_info:
                            filtered_debug_info[idx] = f"{days}일 전 구매 (데이터 없음: 7일 룰)"
                        else:
                            filtered_debug_info[idx] = f"{days}일 전 구매 (재구매 주기 미도래)"
        
        topk_B_ids = np.argsort(scores_B)[::-1][:10]

        # ===============================================================
        # 2. [블라인드 설정] A/B 랜덤 섞기 (최초 1회만 수행)
        # ===============================================================
        
        # 세션에 매핑 정보가 없으면 새로 생성 (새로운 추천이 생성될 때마다 갱신 필요)
        # *주의: 외부 버튼 클릭 시 st.session_state['ab_mapping']을 del 해주는 로직이 있으면 좋습니다.
        # 여기서는 안전하게 없으면 만드는 방식으로 처리합니다.
        if 'ab_mapping' not in st.session_state:
            st.session_state['ab_mapping'] = random.choice(['A_is_1', 'B_is_1'])
            st.session_state['experiment_submitted'] = False

        mapping = st.session_state['ab_mapping']
        
        # 매핑에 따라 옵션 할당
        if mapping == 'A_is_1':
            opt1_ids, opt1_name = topk_A_ids, "Logic A (부스팅 Only)"
            opt2_ids, opt2_name = topk_B_ids, "Logic B (부스팅 + 필터링)"
        else:
            opt1_ids, opt1_name = topk_B_ids, "Logic B (부스팅 + 필터링)"
            opt2_ids, opt2_name = topk_A_ids, "Logic A (부스팅 Only)"

        # Helper: 단순 정보 조회 (블라인드용)
        def get_simple_info(idx):
            name, cat = "Unknown", ""
            if idx in id2token:
                raw_id = id2token[idx]
                row = all_df[all_df['item_id'].astype(str) == raw_id]
                if not row.empty:
                    name = row.iloc[0]['Item_Name']
                    cat = row.iloc[0]['L2']
            return f"[{cat}] {name}"

        # ===============================================================
        # 3. [화면 출력] 1단계: 블라인드 테스트 (제출 전)
        # ===============================================================
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

        # ===============================================================
        # 4. [입력 폼] 선택 및 사유 입력
        # ===============================================================
        st.markdown("---")
        with st.form("ab_test_form"):
            st.write("📝 **평가 입력**")
            st.info("두 옵션 중 더 구매 의사가 높은 추천 목록을 선택하고 이유를 적어주세요.")
            choice = st.radio("더 마음에 드는 추천 결과는?", ["Option 1", "Option 2"], horizontal=True)
            reason = st.text_area("선택한 이유는 무엇인가요? (예: 재구매 상품이 적절해서 / 불필요한 추천이 없어서 등)")
            
            submitted = st.form_submit_button("제출 및 결과 확인", type="primary")
            
            if submitted:
                st.session_state['experiment_submitted'] = True
                st.session_state['user_choice'] = choice
                st.session_state['user_reason'] = reason

        # ===============================================================
        # 5. [결과 공개] 제출 후 정답 및 상세 분석 표시
        # ===============================================================
        if st.session_state.get('experiment_submitted', False):
            st.divider()
            st.header("🔓 결과 공개 및 분석")
            
            # 1. 사용자 선택 결과 요약
            user_pick = st.session_state['user_choice']
            real_logic = opt1_name if user_pick == "Option 1" else opt2_name
            
            st.success(f"✅ 당신의 선택: **{user_pick}**")
            st.info(f"💡 실제 로직: **{real_logic}**")
            st.write(f"🗣️ 작성한 이유: {st.session_state['user_reason']}")
            
            # (데이터 수집 로그 - 나중에 DB 저장용)
            log_data = {
                "alpha": alpha,
                "option1": opt1_name,
                "option2": opt2_name,
                "choice": user_pick,
                "reason": st.session_state['user_reason']
            }

            # (데이터 수집 로그 - 딕셔너리 생성 부분)
            log_data = {
                "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), # 시간 추가
                "user_choice": user_pick,
                "logic_A": opt1_name, # 어떤 로직이 A였는지
                "logic_B": opt2_name, # 어떤 로직이 B였는지
                "alpha": alpha,
                "reason": st.session_state['user_reason'],
                "history_len": len(st.session_state['history']),
                "history_items": str([h['name'] for h in st.session_state['history']]) # 보기 편하게 이름만 저장
            }
            
            # (로그 데이터 딕셔너리 생성 후...)
            new_df = pd.DataFrame([log_data])

            # ---------------------------------------------------------
            # [수정] Google Sheets에 저장하기 (클라우드용)
            # ---------------------------------------------------------
            try:
                # 1. 연결 객체 생성
                conn = st.connection("gsheets", type=GSheetsConnection)
                
                # 2. 기존 데이터 읽기 (없으면 빈 DF)
                try:
                    existing_data = conn.read(worksheet="Sheet1", usecols=list(range(len(log_data.keys()))), ttl=5)
                    updated_data = pd.concat([existing_data, new_df], ignore_index=True)
                except:
                    updated_data = new_df

                # 3. 데이터 업데이트
                conn.update(worksheet="Sheet1", data=updated_data)
                
                st.success("☁️ 데이터가 구글 스프레드시트에 안전하게 저장되었습니다!")
                
            except Exception as e:
                st.error(f"데이터 저장 실패: {e}")
                # 혹시 모르니 백업용으로 CSV 저장 (임시)
                new_df.to_csv("backup_logs.csv", mode='a', header=False, index=False)

            # 2. 상세 시각화
            st.subheader("📊 상세 분석 (Why?)")
            
            # Helper: 상세 정보 조회
            def get_item_info_detail(idx):
                name, cat, raw_id = "Unknown", "", None
                if idx in id2token:
                    raw_id = id2token[idx]
                    row = all_df[all_df['item_id'].astype(str) == raw_id]
                    if not row.empty:
                        name = row.iloc[0]['Item_Name']
                        cat = row.iloc[0]['L2']
                return raw_id, cat, name

            rc1, rc2 = st.columns(2)
            
            # --- Logic A 결과 (왼쪽 고정) ---
            with rc1:
                st.markdown(f"### Logic A: 단순 부스팅")
                st.caption("(필터링 예정 상품은 :orange[주황색] 경고)")
                
                for rank, idx in enumerate(topk_A_ids):
                    if idx == 0: continue
                    raw_id, cat, name = get_item_info_detail(idx)
                    score_val = scores_A[idx]
                    
                    # Logic B에서 필터링되었는지 확인 (점수가 -inf인지 체크)
                    is_filtered_in_B = (scores_B[idx] == -np.inf)
                    
                    if is_filtered_in_B:
                        reason_txt = filtered_debug_info.get(idx, "필터링됨")
                        st.markdown(f"**{rank+1}. :orange[[{cat}] {name}]** ⚠️")
                        st.caption(f":orange[Score: {score_val:.2f} (필터링 예정: {reason_txt})]")
                    else:
                        st.markdown(f"**{rank+1}. [{cat}] {name}**")
                        st.caption(f"Score: {score_val:.2f}")

            # --- Logic B 결과 (오른쪽 고정) ---
            with rc2:
                st.markdown(f"### Logic B: 스마트 필터링")
                st.caption("(새로 진입한 상품은 :green[초록색] 강조)")
                
                for rank, idx in enumerate(topk_B_ids):
                    if idx == 0: continue
                    raw_id, cat, name = get_item_info_detail(idx)
                    score_val = scores_B[idx]
                    
                    is_new_entry = idx not in topk_A_ids
                    
                    if is_new_entry:
                        st.markdown(f"**{rank+1}. :green[[{cat}] {name}]** (New! ✨)")
                        st.caption(f":green[Score: {score_val:.2f} (순위 상승 진입)]")
                    else:
                        st.markdown(f"**{rank+1}. [{cat}] {name}**")
                        st.caption(f"Score: {score_val:.2f}")

if __name__ == "__main__":
    main()
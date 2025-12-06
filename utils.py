import torch
import numpy as np

def get_tisasrec_input(days_list, maxlen, time_span=256):
    """
    days_list: [30, 27, 5] (일 전)
    학습이 '일 단위'로 되었으므로, 초 변환 없이 일(Day) 차이를 그대로 계산합니다.
    """
    
    # 1. 일(Day) 단위 그대로 사용
    # 기준점(10000일)에서 단순히 날짜를 뺍니다. 
    # 예: 30일 전 -> 9970, 27일 전 -> 9973. (차이는 3)
    # (큰 수 10000은 음수가 나오지 않게 하기 위한 임의의 기준점입니다)
    current_day_point = 10000 
    
    # [수정됨] * 86400 삭제!
    timestamps = [current_day_point - d for d in days_list]
    
    # 2. Padding (앞쪽 0 채움)
    seq_len = len(timestamps)
    pad_len = maxlen - seq_len
    
    # 0은 패딩용
    time_seq = [0] * pad_len + timestamps
    
    # 3. Time Matrix 생성
    time_matrix = np.zeros((maxlen, maxlen), dtype=np.int64)
    
    for i in range(maxlen):
        for j in range(maxlen):
            # 패딩이 아닌 실제 값에 대해서만
            if time_seq[i] != 0 and time_seq[j] != 0:
                # 절대값 차이 계산 (이제 '일' 단위로 계산됨)
                diff = abs(time_seq[i] - time_seq[j])
                
                # Clipping (최대 256일까지만 구별)
                if diff >= time_span:
                    time_matrix[i][j] = time_span
                else:
                    time_matrix[i][j] = int(diff)
            else:
                time_matrix[i][j] = 0
    
    # Tensor 변환
    # Tensor 변환 (수정됨)
    # 리스트([])를 np.array()로 한 번 감싸서 넘겨주면 PyTorch가 좋아합니다.
    time_matrix_tensor = torch.LongTensor(np.array([time_matrix]))
    time_seq_tensor = torch.LongTensor(np.array([time_seq]))
    
    return time_seq_tensor, time_matrix_tensor
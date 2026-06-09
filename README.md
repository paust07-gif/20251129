# FRED CPI & PMI Streamlit Dashboard

FRED에서 월간 CPI와 PMI 데이터를 가져와 Streamlit과 Plotly로 시각화하는 Python 프로젝트입니다.

## 구조

```text
src/config.py        # .env 기반 설정
src/fred_client.py   # requests 기반 FRED API 클라이언트
src/transform.py     # 월간 시계열 정제/결합
app/dashboard.py     # Streamlit 대시보드
```

## 설정

1. 의존성을 설치합니다.

   ```bash
   pip install -r requirements.txt
   ```

2. 프로젝트 루트에 `.env` 파일을 만들고 FRED API 키를 설정합니다.

   ```env
   FRED_API_KEY=your_fred_api_key_here
   ```

## 실행

```bash
streamlit run app/dashboard.py
```

CPI 기본 `series_id`는 `CPIAUCSL`입니다. PMI `series_id`는 화면의 사이드바에서 변경할 수 있으며, PMI 조회가 실패하면 오류 메시지와 FRED series search 도구가 표시됩니다.

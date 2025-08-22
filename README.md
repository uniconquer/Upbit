# Upbit Mini (경량 시세 & 패턴/Mean Reversion 뷰)

이 저장소는 업비트 KRW 마켓 상위 종목을 살펴보고 간단한 패턴/평균회귀(Mean Reversion) 조건을 시각화하는 최소 예제입니다.
복잡한 전략/모의매매/리스크 매니지먼트 코드는 제거하여 가볍게 유지합니다.

## 현재 제공 기능
Streamlit 사이드바 탭(버튼) 기반 3개 뷰:

1. Markets
	- 상위 거래대금 KRW-* 마켓 리스트 로드 (24h 거래대금 정렬)
2. Account
	- API 키가 있을 경우 보유 자산 평가액/수익률 표시
3. Backtest (시각화 전용)
	- 캔들/거래량 + Bollinger Bands(20,2), RSI(14)
	- 전환 패턴: 3연속 음봉 + 3번째 하단밴드 종가 이탈 & RSI≤30 이후 불리시 엔걸핑 (옵션: RSI 다이버전스, MACD 히스토그램 전환)
	- Mean Reversion 표시: 하단밴드/과매도(RSI) 진입 → 중심/상단밴드 또는 과매수(RSI) 조건 청산 (옵션 SL/TP)
	- 다중 인터벌(minute1, 15, 60, day, week, month) 및 기간 범위 수집
	- 스캔: 상위 N 종목에 대해 최근 3연속 하락+이탈 패턴 조건 탐색

## 폴더 구조 (간소화)
- src/
  - `app_streamlit.py` Streamlit UI 메인
  - `main.py` 간단 CLI (markets / ticker / bollinger / accounts / order)
  - `upbit_api.py` 업비트 REST 래퍼
- README.md
- requirements.txt

제거됨: 전략 프레임워크, 모의 트레이더, 리스크 매니저, 관련 테스트.

## 설치
```powershell
pip install -r requirements.txt
```

## 환경변수 (.env)
키 없이도 공개 시세/패턴 시각화는 동작합니다. 계정 조회/주문은 키 필요.
```
UPBIT_ACCESS_KEY=...   # 선택
UPBIT_SECRET_KEY=...   # 선택
```

## Streamlit 실행
```powershell
python -m streamlit run .\src\app_streamlit.py --server.headless true
```

## CLI 활용 (선택)
```powershell
python .\src\main.py --help
python .\src\main.py markets --limit 10
python .\src\main.py ticker --market KRW-BTC
python .\src\main.py bollinger --market KRW-BTC --interval day --period 20 --mult 2
python .\src\main.py accounts  # 키 필요
```

`order` 서브커맨드는 기본 구조만 남아 있으며 실제 매매 전 반드시 소액/모의 환경에서 검증하세요.

## 패턴 설명 요약
전환(Engulf Reversal):
1) 3연속 음봉 & 3번째 종가 < 하단밴드 & RSI≤30 -> Breakdown 기준점
2) 이후 강한 불리시 엔걸핑 (필수 여부 선택)
3) (선택) RSI 다이버전스: 두 번째 저점이 더 낮거나 같고 RSI 는 상승
4) (선택) MACD 히스토그램 상승 반전 (직전<0, 증가 전환)

Mean Reversion:
- 진입: 종가 < 하단밴드 OR RSI < 매수레벨
- 청산: 종가 > 상단밴드 OR RSI > 매도레벨 OR (옵션) 중심선 돌파
- SL/TP %: 캔들 고저 범위로 충족 시 즉시 청산으로 가정 (단순화)

## 주의
- 본 코드는 교육/연습용. 실거래 손익 책임은 사용자에게 있습니다.
- 업비트 API 약관 및 레이트리밋을 준수하세요.

## 라이선스
MIT

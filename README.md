# Upbit Trading Starter (Python)

이 저장소는 VS Code에서 바로 실행 가능한 업비트 매매 프로그램 스타터입니다.
기본적으로 공개 시세 조회와 간단한 전략 백테스트/모의매매를 제공하며,
API 키를 설정하면 실제 주문까지 확장할 수 있습니다. 기본값은 모의(PAPER) 모드입니다.

## 폴더 구조
- src/ 공개/비공개 API 래퍼, 전략, 트레이더, 실행 스크립트
- tests/ 간단한 유닛 테스트 예시
- .env.example 환경변수 템플릿 (실제 키는 .env에 저장, 커밋 금지)

## 준비물
- Python 3.9+ 권장
- VS Code

## 설치
1) 패키지 설치

```powershell
pip install -r requirements.txt
```

2) 환경변수 설정
- `.env.example`를 복사해 `.env`를 만들고 값을 채우세요.
- 키가 없으면 공개 기능(시세 조회, 백테스트, 모의매매)만 동작합니다.

## 실행 예시
- 도움말
```powershell
python .\src\main.py --help
```

- 마켓 목록 상위 5개
```powershell
python .\src\main.py markets
```

- 특정 마켓 현재가
```powershell
python .\src\main.py ticker --market KRW-BTC
```

- 간단 SMA 교차 백테스트 (리스크/슬리피지/수수료 옵션 포함 예시)
```powershell
python .\src\main.py backtest --market KRW-BTC --interval day --short 5 --long 20 \
	--slippage 2 --fee 5 --fraction 0.5 \
	--max-pos 60 --max-dd 10 --stop-loss 5 --take-profit 15
```
추가 파라미터 설명:
- --slippage 체결 슬리피지 (bps, 1bp=0.01%)
- --fee 수수료 (bps, 편도)
- --fraction 포지션 비율 (0~1)
- --max-pos 최대 포지션 % (에쿼티 대비)
- --max-dd 최대 인트라데이 드로다운 % (초과 시 신규 매수 차단)
- --stop-loss 손절 % (예: 5 => -5%)
- --take-profit 익절 %

- 모의 매매(짧은 데모 러닝)
```powershell
python .\src\main.py papertrade --market KRW-BTC --cash 1000000 --steps 200 --interval minute5 --short 5 --long 20
```

## 실제 주문 / 시뮬레이션 (선택)
1) 업비트에서 API 키(Access, Secret)를 발급합니다.
2) `.env`에 다음을 채웁니다.
```
UPBIT_ACCESS_KEY=...
UPBIT_SECRET_KEY=...
UPBIT_LIVE=1   # 실제 주문 허용 (주의!) 없거나 0이면 항상 시뮬레이션
```
- 실거래는 UPBIT_LIVE=1 이고 --simulate 를 지정하지 않을 때만 발생
- 실제 매매 전 반드시 소액으로 테스트하세요. 모든 책임은 사용자에게 있습니다.

### 인증 확인 (읽기 전용)
```powershell
python .\src\main.py accounts
```
키가 올바르면 보유 자산 목록이 출력됩니다.

### 주문 (기본 시뮬레이션)
```powershell
# 제한가 매수 시뮬레이션
python .\src\main.py order --market KRW-BTC --side bid --ord-type limit --price 50000000 --volume 0.002 --simulate

# 시장가 매수 (ord-type=price, KRW 총액)
python .\src\main.py order --market KRW-BTC --side bid --ord-type price --price 10000 --simulate

# 시장가 매도 (ord-type=market)
python .\src\main.py order --market KRW-BTC --side ask --ord-type market --volume 0.001 --simulate
```
실거래: --simulate 제거 + .env UPBIT_LIVE=1 설정.

## Streamlit UI 확장
- Markets: 상위 거래대금
- Backtest: SMA 백테스트 및 Buy&Hold 비교
- Account: 잔고
- Live: 실시간(WebSocket/REST) 티커
- Trade: 주문 폼 (시뮬레이션/LIVE 가드)
	- UI 리스크 파라미터는 안내용이며 자동 청산 로직은 추후 구현 예정.

## 주의 사항
- 투자 손실 위험이 있으니 스스로 판단하고 사용하세요.
- API 이용 시 업비트 이용약관/레이트리밋을 준수하세요.

## 라이선스
MIT

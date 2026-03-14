# Work Log

## 2026-03-14

### supervised live pilot 준비

- 실주문 경로에 client `identifier`를 추가했다.
- 주문 POST 응답이 애매하게 실패한 경우에도 `identifier`로 다시 조회해 상태를 복구하도록 보강했다.
- `MRMonitor`에 주기적 거래소 재동기화를 추가했다.
  - 초기 1회 동기화에 멈추지 않고 성공 후 180초, 실패 후 60초 기준으로 다시 동기화한다.
  - 동기화 성공 알림은 최초 성공 시점 위주로만 보내도록 정리했다.
- `MyOrder` private WebSocket 보조 경로를 추가했다.
  - 백그라운드 CLI 워커는 주문 이벤트를 먼저 소비하고, 그다음 REST 재조회로 보강한다.
- 거래소에 미체결 주문이 더 이상 없으면 로컬 stale pending 주문도 지우도록 정리했다.
- 라이브 데스크 UI 경로도 `identifier`와 주기적 재동기화 규칙을 맞췄다.

### 검증

- `python -m pytest -q` -> `59 passed`
- `python -m compileall src tests`
- `python src\\mr_worker.py --cycles 1 --markets 2 --loop-seconds 1 --count 120 --state-name smoke-live-ready`

### 현재 판단

- supervised live pilot 가능
- 권장 조건
  - 백그라운드 CLI 워커 사용
  - 1~2개 종목
  - 아주 작은 주문금액
  - 텔레그램 모니터링 유지
  - 긴급중지 준비
  - 사람이 상태를 보는 운영

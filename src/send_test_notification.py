"""Quick script to test notifier configuration.
Usage (PowerShell):
  python .\src\send_test_notification.py --msg "hello"
If no --msg given, sends a default test message.
"""
import argparse
from notifier import get_notifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--msg', default='Notification test: 안녕하세요 (mean reversion 알림 테스트)')
    args = p.parse_args()
    n = get_notifier()
    resp = n.send_text(args.msg)
    print('available:', n.available())
    print('response:', resp)
    if not resp.get('ok'):
        print('\n[FAIL] 알림 전송 실패. 환경 변수(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) 또는 네트워크 확인.')
    else:
        print('\n[SUCCESS] 알림 전송됨.')

if __name__ == '__main__':
    main()

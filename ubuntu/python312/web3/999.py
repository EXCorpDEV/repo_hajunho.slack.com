from web3 import Web3
import time

# 폴리곤 메인넷 RPC URL
polygon_rpc_url = "https://polygon-rpc.com"

# Web3 인스턴스 생성
w3 = Web3(Web3.HTTPProvider(polygon_rpc_url))

# 연결 확인
if w3.is_connected():
    print("폴리곤 네트워크에 연결되었습니다.")
else:
    print("연결 실패. RPC URL을 확인하세요.")
    exit(1)

# 기본 네트워크 정보 출력
print(f"\n==== 폴리곤 네트워크 정보 ====")
print(f"현재 블록 번호: {w3.eth.block_number}")
print(f"네트워크 ID: {w3.eth.chain_id}")
print(f"네트워크 버전: {w3.net.version}")

# 피어 카운트는 오류가 발생하므로 try-except로 처리
try:
    print(f"피어 수: {w3.net.peer_count}")
except Exception as e:
    print(f"피어 수: 정보를 조회할 수 없습니다 (사유: {e})")

# 가스 정보
try:
    gas_price = w3.eth.gas_price
    print(f"\n==== 가스 정보 ====")
    print(f"현재 가스 가격: {w3.from_wei(gas_price, 'gwei'):.2f} Gwei")
except Exception as e:
    print(f"\n가스 정보를 조회할 수 없습니다: {e}")

# 최신 블록 정보
try:
    latest_block = w3.eth.get_block('latest')
    print(f"\n==== 최신 블록 정보 ====")
    print(f"블록 번호: {latest_block['number']}")
    print(f"블록 해시: {latest_block['hash'].hex()}")
    print(f"타임스탬프: {time.ctime(latest_block['timestamp'])}")
    print(f"난이도: {latest_block['difficulty']}")
    print(f"가스 사용량: {latest_block['gasUsed']}")
    print(f"가스 한도: {latest_block['gasLimit']}")
    print(f"트랜잭션 수: {len(latest_block['transactions'])}")

    # 블록 크기 계산 시도
    try:
        block_hex = w3.to_hex(w3.eth.get_block(latest_block['number'], True))
        print(f"블록 크기: {len(block_hex) // 2} 바이트")
    except Exception as e:
        print(f"블록 크기: 정보를 조회할 수 없습니다 (사유: {e})")

except Exception as e:
    print(f"\n블록 정보를 조회할 수 없습니다: {e}")

# 몇 가지 추가 정보 조회 시도
print(f"\n==== 추가 네트워크 정보 ====")

# 현재 블록에서 진행 중인 트랜잭션 수
try:
    tx_count = len(w3.eth.get_block('latest')['transactions'])
    print(f"최신 블록의 트랜잭션 수: {tx_count}")
except Exception as e:
    print(f"트랜잭션 수: 정보를 조회할 수 없습니다 (사유: {e})")

# 현재 가스 한도
try:
    gas_limit = w3.eth.get_block('latest')['gasLimit']
    print(f"현재 블록 가스 한도: {gas_limit}")
except Exception as e:
    print(f"가스 한도: 정보를 조회할 수 없습니다 (사유: {e})")
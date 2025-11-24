import torch
import time
import threading
import subprocess
import signal
import sys
import math
import os
from datetime import datetime

class GPUPrimeFinder:
    def __init__(self, target_usage=3.0, max_memory_gb=2.0):
        """
        GPU로 소수 찾기 - 사용률과 메모리 제한
        
        Args:
            target_usage: 목표 GPU 사용률 (%)
            max_memory_gb: 최대 GPU 메모리 사용량 (GB)
        """
        self.target_usage = target_usage
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.running = False
        
        # 소수 찾기 상태
        self.current_range_start = 1000000  # 100만부터 시작
        self.batch_size = 50000  # 한 번에 처리할 숫자 개수
        self.found_primes = []
        self.total_checked = 0
        self.primes_found = 0
        self.session_start_time = datetime.now()
        
        # 성능 조절
        self.work_intensity = 15  # 초기값
        self.rest_time = 0.1  # 배치 간 휴식 시간
        
        # 파일 관리
        self.results_file = "gpu_primes_collection.txt"
        self.load_previous_progress()
        
        print(f"🔢 GPU 소수 찾기 시작")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"사용 제한: {max_memory_gb} GB")
        print(f"목표 사용률: {target_usage}%")
        print(f"시작 범위: {self.current_range_start:,}")
        print(f"결과 파일: {self.results_file}")
        print("-" * 50)
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        print(f"\n종료 중...")
        self.save_final_summary()
        self.stop()
        sys.exit(0)
    
    def get_gpu_usage(self):
        """GPU 사용률 확인"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0.0
    
    def get_gpu_memory_usage(self):
        """GPU 메모리 사용량 확인 (bytes)"""
        try:
            return torch.cuda.memory_allocated(self.device)
        except:
            return 0
    
    def load_previous_progress(self):
        """이전 진행 상황 로드"""
        try:
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r') as f:
                    lines = f.readlines()
                
                # 마지막 범위 찾기
                last_range = 0
                total_primes = 0
                
                for line in lines:
                    if line.startswith("# 현재 범위:"):
                        try:
                            last_range = int(line.split(":")[1].strip().replace(",", ""))
                        except:
                            pass
                    elif line.startswith("# 총 소수:"):
                        try:
                            total_primes = int(line.split(":")[1].strip().replace(",", ""))
                        except:
                            pass
                
                if last_range > self.current_range_start:
                    self.current_range_start = last_range
                    print(f"📂 이전 진행 상황 로드: {last_range:,}부터 계속")
                    print(f"📊 이전까지 발견한 소수: {total_primes:,}개")
                
        except Exception as e:
            print(f"이전 진행 상황 로드 실패: {e}")
    
    def save_primes_to_file(self, new_primes):
        """새로 발견한 소수들을 파일에 추가"""
        if not new_primes:
            return
        
        try:
            # 파일이 없으면 헤더 생성
            if not os.path.exists(self.results_file):
                with open(self.results_file, 'w') as f:
                    f.write("# GPU 소수 컬렉션 📊\n")
                    f.write(f"# GPU: {torch.cuda.get_device_name()}\n")
                    f.write(f"# 시작일: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("# ----------------------------------------\n")
                    f.write("# 형식: 소수 (발견시간)\n")
                    f.write("# ----------------------------------------\n\n")
            
            # 새 소수들 추가
            with open(self.results_file, 'a') as f:
                current_time = datetime.now().strftime('%H:%M:%S')
                
                for prime in new_primes:
                    f.write(f"{prime} ({current_time})\n")
                
                # 현재 상태 업데이트
                f.write(f"\n# 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 현재 범위: {self.current_range_start:,}\n")
                f.write(f"# 이번 세션 발견: {self.primes_found:,}개\n")
                f.write(f"# 총 확인한 수: {self.total_checked:,}개\n")
                
                # 성능 통계
                elapsed = (datetime.now() - self.session_start_time).total_seconds()
                if elapsed > 0:
                    rate = self.total_checked / elapsed
                    f.write(f"# 확인 속도: {rate:.0f} 수/초\n")
                
                f.write("# ----------------------------------------\n\n")
                f.flush()
            
            print(f"💾 {len(new_primes)}개 소수 파일에 저장됨")
            
        except Exception as e:
            print(f"파일 저장 오류: {e}")
    
    def save_final_summary(self):
        """최종 요약 저장"""
        try:
            with open(self.results_file, 'a') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"# 세션 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                elapsed = (datetime.now() - self.session_start_time).total_seconds()
                hours = elapsed // 3600
                minutes = (elapsed % 3600) // 60
                seconds = elapsed % 60
                
                f.write(f"# 실행 시간: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초\n")
                f.write(f"# 최종 범위: {self.current_range_start:,}\n")
                f.write(f"# 이번 세션 결과:\n")
                f.write(f"#   - 확인한 수: {self.total_checked:,}개\n")
                f.write(f"#   - 발견한 소수: {self.primes_found:,}개\n")
                
                if self.total_checked > 0:
                    prime_ratio = (self.primes_found / self.total_checked) * 100
                    f.write(f"#   - 소수 비율: {prime_ratio:.4f}%\n")
                
                if elapsed > 0:
                    rate = self.total_checked / elapsed
                    f.write(f"#   - 평균 확인 속도: {rate:.0f} 수/초\n")
                
                f.write(f"{'='*50}\n\n")
                f.flush()
            
            print(f"📋 최종 요약 저장 완료")
            
        except Exception as e:
            print(f"최종 요약 저장 오류: {e}")
    
    def is_prime_gpu_batch(self, numbers):
        """GPU에서 배치로 소수 판별"""
        try:
            # 입력을 GPU 텐서로 변환
            nums = torch.tensor(numbers, device=self.device, dtype=torch.long)
            batch_size = len(numbers)
            
            # 결과 텐서 (True = 소수)
            is_prime = torch.ones(batch_size, device=self.device, dtype=torch.bool)
            
            # 1과 짝수 제거 (2 제외)
            is_prime = is_prime & (nums > 1)
            is_prime = is_prime & ((nums == 2) | (nums % 2 != 0))
            
            # 3부터 sqrt(n)까지의 홀수로 나누어 확인
            max_num = int(math.sqrt(max(numbers))) + 1
            for divisor in range(3, max_num, 2):
                if not self.running:  # 중단 체크
                    break
                
                # GPU에서 나머지 연산
                divisor_tensor = torch.tensor(divisor, device=self.device)
                remainder = nums % divisor_tensor
                is_prime = is_prime & (remainder != 0)
                
                # 메모리 사용량 체크
                if self.get_gpu_memory_usage() > self.max_memory_bytes:
                    print("⚠️ 메모리 한도 초과, 배치 크기 축소")
                    break
            
            # CPU로 결과 복사
            result = is_prime.cpu().numpy()
            
            # GPU 메모리 정리
            del nums, is_prime
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            print(f"GPU 소수 판별 오류: {e}")
            torch.cuda.empty_cache()
            return [False] * len(numbers)
    
    def prime_finding_worker(self):
        """소수 찾기 메인 워커"""
        print("🔍 소수 찾기 시작...")
        
        while self.running:
            try:
                # 현재 범위 설정
                range_end = self.current_range_start + self.batch_size
                
                # 메모리 사용량에 따른 배치 크기 조정
                current_memory = self.get_gpu_memory_usage()
                if current_memory > self.max_memory_bytes * 0.8:  # 80% 초과시
                    self.batch_size = max(1000, self.batch_size // 2)
                    print(f"배치 크기 축소: {self.batch_size}")
                
                # 숫자 배치 생성 (홀수만, 성능 최적화)
                if self.current_range_start % 2 == 0:
                    self.current_range_start += 1
                
                numbers = list(range(self.current_range_start, range_end, 2))
                
                # 강도 조절: 일부만 처리
                actual_batch_size = max(100, len(numbers) * self.work_intensity // 30)
                numbers = numbers[:actual_batch_size]
                
                # GPU에서 소수 판별
                prime_flags = self.is_prime_gpu_batch(numbers)
                
                # 소수 추출 및 즉시 저장
                new_primes = [num for num, is_prime in zip(numbers, prime_flags) if is_prime]
                
                # 파일에 즉시 저장 (큰 소수들은 바로 저장)
                if new_primes:
                    self.save_primes_to_file(new_primes)
                
                # 결과 업데이트
                self.found_primes.extend(new_primes)
                self.total_checked += len(numbers)
                self.primes_found += len(new_primes)
                
                # 최근 소수 출력
                if new_primes:
                    recent_primes = new_primes[-min(3, len(new_primes)):]
                    print(f"✨ 새 소수 발견: {recent_primes}")
                
                # 범위 업데이트
                self.current_range_start = range_end
                
                # 강도에 따른 휴식
                time.sleep(self.rest_time)
                
            except Exception as e:
                print(f"소수 찾기 오류: {e}")
                torch.cuda.empty_cache()
                time.sleep(1)
        
        print("🔍 소수 찾기 종료")
    
    def monitor_and_adjust(self):
        """성능 모니터링 및 조정"""
        print("📊 모니터링 시작")
        
        while self.running:
            # 현재 상태 확인
            gpu_usage = self.get_gpu_usage()
            memory_usage = self.get_gpu_memory_usage()
            memory_gb = memory_usage / (1024**3)
            
            # 통계 출력
            print(f"📈 GPU: {gpu_usage:.1f}% | 메모리: {memory_gb:.2f}GB | 강도: {self.work_intensity}")
            print(f"🔢 이번 세션: {self.total_checked:,}개 확인, {self.primes_found:,}개 소수 발견")
            if self.primes_found > 0:
                print(f"🎯 현재 범위: {self.current_range_start:,} | 최근 소수: {self.found_primes[-1]:,}")
                
            # 실행 시간 및 속도 계산
            elapsed = (datetime.now() - self.session_start_time).total_seconds()
            if elapsed > 0:
                rate = self.total_checked / elapsed
                print(f"⚡ 실행시간: {elapsed/60:.1f}분 | 확인속도: {rate:.0f}수/초")
            
            # 사용률 조정
            if gpu_usage < self.target_usage * 0.8:
                if self.work_intensity < 30:
                    self.work_intensity += 1
                    self.rest_time = max(0.01, self.rest_time * 0.9)
                    print(f"→ 강도 증가: {self.work_intensity}")
            elif gpu_usage > self.target_usage * 1.3:
                if self.work_intensity > 5:
                    self.work_intensity -= 1
                    self.rest_time = min(0.5, self.rest_time * 1.1)
                    print(f"→ 강도 감소: {self.work_intensity}")
            else:
                print("✅ 목표 사용률 달성")
            
            # 메모리 초과 시 강제 정리
            # if memory_gb > self.max_memory_gb * 0.9:
            if memory_gb > (self.max_memory_bytes / (1024**3)) * 0.9:
                print("🧹 메모리 정리 중...")
                torch.cuda.empty_cache()
            
            print("-" * 50)
            time.sleep(10)
    
    def start(self):
        """소수 찾기 시작"""
        if not torch.cuda.is_available():
            print("❌ CUDA를 사용할 수 없습니다!")
            return
        
        print("🚀 GPU 소수 찾기 시작...")
        self.running = True
        
        # 소수 찾기 워커 시작
        prime_worker = threading.Thread(target=self.prime_finding_worker)
        prime_worker.daemon = True
        prime_worker.start()
        
        # 모니터링 워커 시작
        monitor_worker = threading.Thread(target=self.monitor_and_adjust)
        monitor_worker.daemon = True
        monitor_worker.start()
        
        print("💡 Ctrl+C로 종료 및 결과 저장")
        print("=" * 50)
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n사용자 종료 요청")
            self.save_final_summary()
            self.stop()
    
    def stop(self):
        """종료"""
        print("🛑 소수 찾기 종료 중...")
        self.running = False
        torch.cuda.empty_cache()
        print("✅ 완료")

def main():
    # GPU 사용률 3%, 메모리 2GB 제한
    finder = GPUPrimeFinder(target_usage=3.0, max_memory_gb=2.0)
    finder.start()

if __name__ == "__main__":
    main()


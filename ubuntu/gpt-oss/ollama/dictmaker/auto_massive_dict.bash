#!/bin/bash
# 완전 자동화 영어 사전 생성 스크립트
# 2일 출장 중 무인 실행용

LOG_DIR="./logs"
SCRIPT_PATH="./massive_dict.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 로그 디렉토리 생성
mkdir -p $LOG_DIR

echo "=== 완전 영어 사전 생성 시작 ==="
echo "시작 시간: $(date)"
echo "예상 완료: $(date -d '+2 days')"
echo "로그 위치: $LOG_DIR"
echo "=================================="

# 1단계: 대용량 단어 수집
echo "[$(date)] 1단계 시작: 대용량 단어 수집 (2letter 모드)"
echo "y" | python3 $SCRIPT_PATH --stage 1 --mode 2letter --batch 35 > "$LOG_DIR/stage1_${TIMESTAMP}.log" 2>&1

# 1단계 결과 확인
if [ $? -eq 0 ]; then
    echo "[$(date)] 1단계 완료!"
    
    # 생성된 파일 찾기
    WORDLIST_FILE=$(ls -t massive_wordlist_2letter_*.txt 2>/dev/null | head -1)
    
    if [ -n "$WORDLIST_FILE" ]; then
        echo "[$(date)] 생성된 파일: $WORDLIST_FILE"
        
        # 단어 수 확인
        WORD_COUNT=$(python3 -c "import json; data=json.load(open('$WORDLIST_FILE')); print(data['metadata']['total_words'])")
        echo "[$(date)] 수집된 단어 수: $WORD_COUNT"
        
        # 2단계: 상세 정보 보완
        echo "[$(date)] 2단계 시작: 상세 정보 보완"
        echo "y" | python3 $SCRIPT_PATH --stage 2 --file "$WORDLIST_FILE" --chunk 80 > "$LOG_DIR/stage2_${TIMESTAMP}.log" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "[$(date)] 2단계 완료!"
            
            # 최종 결과 확인
            FINAL_FILE=$(ls -t enhanced_massive_dict_*.txt 2>/dev/null | head -1)
            if [ -n "$FINAL_FILE" ]; then
                FINAL_COUNT=$(python3 -c "import json; data=json.load(open('$FINAL_FILE')); print(data['metadata']['statistics']['total_words'])")
                echo "[$(date)] 최종 완성: $FINAL_FILE"
                echo "[$(date)] 최종 단어 수: $FINAL_COUNT"
                
                # 성공 보고서 생성
                cat > "$LOG_DIR/success_report_${TIMESTAMP}.txt" << EOF
=== 완전 영어 사전 생성 완료 ===
완료 시간: $(date)
소요 시간: $(echo "$(date +%s) - $(date -d "$(stat -c %y $LOG_DIR/stage1_${TIMESTAMP}.log | cut -d' ' -f1-2)" +%s)" | bc) 초

1단계 결과:
- 파일: $WORDLIST_FILE  
- 수집 단어 수: $WORD_COUNT

2단계 결과:
- 파일: $FINAL_FILE
- 최종 단어 수: $FINAL_COUNT

로그 파일:
- 1단계: $LOG_DIR/stage1_${TIMESTAMP}.log
- 2단계: $LOG_DIR/stage2_${TIMESTAMP}.log

오래 걸릴듯!
EOF
                
                echo "[$(date)] 성공 보고서 생성: $LOG_DIR/success_report_${TIMESTAMP}.txt"
                
            else
                echo "[$(date)] 오류: 최종 파일을 찾을 수 없습니다"
            fi
        else
            echo "[$(date)] 오류: 2단계 실패"
        fi
        
    else
        echo "[$(date)] 오류: 1단계 출력 파일을 찾을 수 없습니다"
    fi
else
    echo "[$(date)] 오류: 1단계 실패"
fi

echo "[$(date)] 전체 프로세스 종료"

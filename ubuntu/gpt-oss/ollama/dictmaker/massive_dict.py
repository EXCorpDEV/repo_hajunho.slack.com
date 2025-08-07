#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대용량 영어 사전 생성기 (접두사별 분할)
영어 사전 수준의 단어를 수집하기 위해 접두사별로 세분화
예: a -> aa, ab, ac, ad... / b -> ba, bb, bc, bd...
"""

import requests
import json
import time
import os
import glob
from datetime import datetime
import argparse
import string

class MassiveDictionaryGenerator:
    def __init__(self, model="gpt-oss:120b", host="localhost", port=11434):
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/api/generate"
        
        # 연령대 분류
        self.age_groups = {
            "elementary": "초등학생 (6-12세)",
            "middle": "중학생 (13-15세)", 
            "high": "고등학생 (16-18세)",
            "adult": "성인 (19세 이상)",
            "professional": "전문가"
        }
    
    def ask_ollama(self, prompt, timeout=45):
        """ollama에게 질문하고 답변 받기"""
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.base_url, json=data, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"API 오류: {e}")
            return ""
    
    def generate_prefixes(self, mode="2letter"):
        """접두사 목록 생성"""
        prefixes = []
        
        if mode == "2letter":
            # 2글자 접두사: aa, ab, ac, ..., zz
            for first in string.ascii_lowercase:
                for second in string.ascii_lowercase:
                    prefixes.append(first + second)
        
        elif mode == "3letter":
            # 3글자 접두사: aaa, aab, aac, ..., zzz (너무 많으므로 일부만)
            for first in string.ascii_lowercase:
                for second in string.ascii_lowercase:
                    # 자주 사용되는 조합만 선택
                    common_thirds = ['a', 'e', 'i', 'o', 'u', 'r', 's', 't', 'n', 'l']
                    for third in common_thirds:
                        prefixes.append(first + second + third)
        
        elif mode == "adaptive":
            # 적응형: 일반적인 조합 우선
            # 1글자로 시작 (기본 26개)
            prefixes.extend(string.ascii_lowercase)
            
            # 자주 사용되는 2글자 조합 추가
            common_combinations = [
                'th', 'he', 'in', 'er', 'an', 're', 'ed', 'nd', 'ou', 'ea',
                'ti', 'to', 'it', 'st', 'io', 'le', 'is', 'on', 'al', 'ar',
                'at', 'se', 'ng', 'me', 'de', 'of', 'te', 'en', 'ty', 'ch',
                'co', 'di', 'ho', 'li', 'ma', 'ne', 'pe', 'ro', 'so', 'tr',
                'un', 'pr', 'ex', 'in', 'con', 'dis', 'pre', 'pro', 'anti'
            ]
            
            for combo in common_combinations:
                if combo not in prefixes:
                    prefixes.append(combo)
        
        return prefixes
    
    def collect_words_by_prefix(self, prefix, batch_size=50):
        """특정 접두사로 시작하는 단어들 수집"""
        if len(prefix) == 1:
            prompt = f"""
List common English words that start with the letter '{prefix}'.
Provide up to {batch_size} words, one word per line.
Include words of different difficulty levels from basic to advanced.
No explanations, just the words.

Format:
word1
word2
word3
"""
        else:
            prompt = f"""
List English words that start with '{prefix}'.
Provide up to {batch_size} words, one word per line.
Include both common and less common words.
No explanations, just the words.

Format:
word1
word2
word3
"""
        
        print(f"'{prefix}' 접두사 단어 수집 중...", end=" ")
        response = self.ask_ollama(prompt, timeout=30)
        words = self.parse_prefix_words(response, prefix)
        print(f"{len(words)}개")
        return words
    
    def parse_prefix_words(self, response, prefix):
        """접두사별 단어 응답 파싱"""
        words = []
        lines = response.strip().split('\n')
        
        for line in lines:
            word = line.strip().lower()
            # 유효성 검사
            if (word and 
                word.replace('-', '').replace("'", "").isalpha() and  # 하이픈, 아포스트로피 허용
                word.startswith(prefix.lower()) and 
                len(word) > len(prefix)):  # 접두사보다 길어야 함
                
                words.append({
                    "word": word,
                    "prefix": prefix,
                    "collected_at": datetime.now().isoformat()
                })
        
        return words
    
    def collect_massive_wordlist(self, mode="adaptive", batch_size=30, max_prefixes=None):
        """대용량 단어 목록 수집"""
        print("=== 대용량 영어 단어 수집 시작 ===")
        print(f"모드: {mode}")
        print(f"배치 크기: {batch_size}")
        print(f"모델: {self.model}")
        print("-" * 50)
        
        prefixes = self.generate_prefixes(mode)
        
        if max_prefixes:
            prefixes = prefixes[:max_prefixes]
            print(f"제한된 접두사 수: {len(prefixes)}")
        else:
            print(f"총 접두사 수: {len(prefixes)}")
        
        print(f"예상 최대 단어 수: {len(prefixes) * batch_size}")
        print(f"예상 소요시간: {len(prefixes) * 2 / 60:.1f}분")
        
        if input("\n계속하시겠습니까? (y/N): ").strip().lower() != 'y':
            return None
        
        all_words = []
        failed_prefixes = []
        word_count_by_prefix = {}
        
        for i, prefix in enumerate(prefixes, 1):
            try:
                words = self.collect_words_by_prefix(prefix, batch_size)
                
                if words:
                    all_words.extend(words)
                    word_count_by_prefix[prefix] = len(words)
                    
                    # 진행률 표시
                    if i % 20 == 0:
                        avg_words = len(all_words) / i
                        print(f"    진행률: {i}/{len(prefixes)} ({i/len(prefixes)*100:.1f}%) - "
                              f"총 {len(all_words)}개 단어 (평균 {avg_words:.1f}개/접두사)")
                else:
                    failed_prefixes.append(prefix)
                
                time.sleep(0.8)  # API 과부하 방지
                
            except KeyboardInterrupt:
                print("\n중단됨. 지금까지 수집한 데이터를 저장합니다...")
                break
            except Exception as e:
                print(f"오류 ({prefix}): {e}")
                failed_prefixes.append(prefix)
        
        # 중복 제거
        unique_words = {}
        for word_data in all_words:
            word = word_data["word"]
            if word not in unique_words:
                unique_words[word] = word_data
        
        final_words = list(unique_words.values())
        
        # 결과 저장
        if final_words:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"massive_wordlist_{mode}_{timestamp}.txt"
            
            # 통계 생성
            letter_stats = {}
            for word_data in final_words:
                first_letter = word_data["word"][0].upper()
                letter_stats[first_letter] = letter_stats.get(first_letter, 0) + 1
            
            massive_data = {
                "metadata": {
                    "stage": 1,
                    "type": "massive_wordlist",
                    "title": f"Massive English Wordlist ({mode} mode)",
                    "model_used": self.model,
                    "collection_mode": mode,
                    "batch_size": batch_size,
                    "total_prefixes": len(prefixes),
                    "successful_prefixes": len(prefixes) - len(failed_prefixes),
                    "total_words": len(final_words),
                    "duplicates_removed": len(all_words) - len(final_words),
                    "failed_prefixes": failed_prefixes,
                    "word_count_by_prefix": word_count_by_prefix,
                    "letter_distribution": letter_stats,
                    "created_at": datetime.now().isoformat()
                },
                "words": final_words
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(massive_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n대용량 단어 목록 완성!")
            print(f"파일: {filename}")
            print(f"총 단어 수: {len(final_words):,}개")
            print(f"중복 제거: {len(all_words) - len(final_words):,}개")
            print(f"성공률: {(len(prefixes) - len(failed_prefixes))/len(prefixes)*100:.1f}%")
            
            # 알파벳별 분포
            print("\n알파벳별 단어 수:")
            for letter in sorted(letter_stats.keys()):
                print(f"  {letter}: {letter_stats[letter]:,}개")
            
            return filename
        else:
            print("수집된 단어가 없습니다.")
            return None
    
    def enhance_massive_dictionary(self, wordlist_file, chunk_size=100):
        """대용량 단어 목록의 상세 정보 보완 (청크 단위)"""
        print("=== 대용량 사전 상세 정보 보완 ===")
        
        if not os.path.exists(wordlist_file):
            print(f"파일을 찾을 수 없습니다: {wordlist_file}")
            return None
        
        # 단어 목록 로드
        with open(wordlist_file, 'r', encoding='utf-8') as f:
            wordlist_data = json.load(f)
        
        words = wordlist_data["words"]
        total = len(words)
        
        print(f"총 {total:,}개 단어의 상세 정보를 보완합니다.")
        print(f"청크 크기: {chunk_size}개씩")
        print(f"예상 API 호출 수: {(total + chunk_size - 1) // chunk_size}회")
        
        if input("\n계속하시겠습니까? (y/N): ").strip().lower() != 'y':
            return None
        
        enhanced_words = []
        failed_chunks = []
        
        # 청크 단위로 처리
        for i in range(0, total, chunk_size):
            chunk = words[i:i+chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (total + chunk_size - 1) // chunk_size
            
            print(f"\n청크 {chunk_num}/{total_chunks} 처리 중... ({len(chunk)}개 단어)")
            
            enhanced_chunk = self.enhance_word_chunk(chunk)
            
            if enhanced_chunk:
                enhanced_words.extend(enhanced_chunk)
                print(f"  성공: {len(enhanced_chunk)}/{len(chunk)}개")
            else:
                failed_chunks.append(chunk_num)
                print(f"  실패: 청크 {chunk_num}")
            
            # 진행률 표시
            progress = min(i + chunk_size, total)
            print(f"전체 진행률: {progress:,}/{total:,} ({progress/total*100:.1f}%)")
            
            time.sleep(1)  # API 과부하 방지
        
        # 결과 저장
        if enhanced_words:
            return self.save_enhanced_massive_dictionary(enhanced_words, wordlist_file)
        else:
            print("보완된 단어가 없습니다.")
            return None
    
    def enhance_word_chunk(self, word_chunk):
        """단어 청크의 상세 정보 한번에 요청"""
        word_list = [w["word"] for w in word_chunk]
        words_str = ", ".join(word_list)
        
        prompt = f"""
For these English words: {words_str}

Provide information for each word in this format:
WORD: korean_meaning|age_level|part_of_speech|simple_example

Where:
- korean_meaning: Korean translation
- age_level: elementary/middle/high/adult/professional  
- part_of_speech: noun/verb/adjective/adverb/etc
- simple_example: short English sentence

Example:
apple: 사과|elementary|noun|I eat an apple.
beautiful: 아름다운|middle|adjective|She is beautiful.

Just provide the word information, nothing else.
"""
        
        response = self.ask_ollama(prompt, timeout=60)
        return self.parse_chunk_response(response, word_chunk)
    
    def parse_chunk_response(self, response, original_chunk):
        """청크 응답 파싱"""
        enhanced_words = []
        lines = response.strip().split('\n')
        
        # 원본 단어들을 딕셔너리로 변환
        original_dict = {w["word"]: w for w in original_chunk}
        
        for line in lines:
            line = line.strip()
            if ':' not in line or '|' not in line:
                continue
            
            try:
                word_part, info_part = line.split(':', 1)
                word = word_part.strip().lower()
                
                info_parts = info_part.strip().split('|')
                if len(info_parts) >= 4:
                    korean = info_parts[0].strip()
                    age_group = info_parts[1].strip().lower()
                    part_of_speech = info_parts[2].strip()
                    example = info_parts[3].strip()
                    
                    if word in original_dict and age_group in self.age_groups:
                        enhanced_word = {
                            **original_dict[word],  # 기존 정보
                            "korean": korean,
                            "age_group": age_group,
                            "age_description": self.age_groups[age_group],
                            "part_of_speech": part_of_speech,
                            "example": example,
                            "enhanced_at": datetime.now().isoformat()
                        }
                        enhanced_words.append(enhanced_word)
            
            except Exception as e:
                continue  # 파싱 실패한 라인 무시
        
        return enhanced_words
    
    def save_enhanced_massive_dictionary(self, enhanced_words, source_file):
        """향상된 대용량 사전 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_massive_dict_{timestamp}.txt"
        
        # 통계 생성
        stats = {
            "total_words": len(enhanced_words),
            "by_letter": {},
            "by_age_group": {},
            "by_part_of_speech": {}
        }
        
        for word_data in enhanced_words:
            first_letter = word_data["word"][0].upper()
            age_group = word_data.get("age_group", "unknown")
            pos = word_data.get("part_of_speech", "unknown")
            
            stats["by_letter"][first_letter] = stats["by_letter"].get(first_letter, 0) + 1
            stats["by_age_group"][age_group] = stats["by_age_group"].get(age_group, 0) + 1
            stats["by_part_of_speech"][pos] = stats["by_part_of_speech"].get(pos, 0) + 1
        
        enhanced_data = {
            "metadata": {
                "stage": 2,
                "type": "enhanced_massive_dictionary",
                "title": "Enhanced Massive English-Korean Dictionary",
                "description": "대용량 영어-한국어 사전 (연령대별 분류)",
                "model_used": self.model,
                "source_file": source_file,
                "statistics": stats,
                "created_at": datetime.now().isoformat()
            },
            "age_groups": self.age_groups,
            "words": enhanced_words
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n최종 대용량 사전 완성!")
        print(f"파일: {filename}")
        print(f"총 단어 수: {len(enhanced_words):,}개")
        
        # 통계 출력
        print(f"\n연령대별 분포:")
        for age, count in stats["by_age_group"].items():
            if age in self.age_groups:
                print(f"  {self.age_groups[age]}: {count:,}개")
        
        return filename

def main():
    parser = argparse.ArgumentParser(description="대용량 영어 사전 생성기")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1,
                       help="실행할 단계 (1: 대용량 단어 수집, 2: 상세 정보 보완)")
    parser.add_argument("--file", type=str, help="2단계에서 사용할 단어 목록 파일")
    parser.add_argument("--model", type=str, default="gpt-oss:120b", help="사용할 ollama 모델")
    parser.add_argument("--mode", type=str, choices=["adaptive", "2letter", "3letter"], 
                       default="adaptive", help="수집 모드")
    parser.add_argument("--batch", type=int, default=30, help="접두사당 수집할 단어 수")
    parser.add_argument("--limit", type=int, help="처리할 최대 접두사 수 (테스트용)")
    parser.add_argument("--chunk", type=int, default=50, help="2단계에서 한번에 처리할 단어 수")
    
    args = parser.parse_args()
    
    generator = MassiveDictionaryGenerator(model=args.model)
    
    if args.stage == 1:
        result_file = generator.collect_massive_wordlist(
            mode=args.mode, 
            batch_size=args.batch,
            max_prefixes=args.limit
        )
        
        if result_file:
            print(f"\n다음 단계 실행 명령:")
            print(f"python3 {__file__} --stage 2 --file {result_file} --chunk {args.chunk}")
    
    elif args.stage == 2:
        if not args.file:
            # 가장 최근 massive_wordlist 파일 찾기
            wordlist_files = glob.glob("massive_wordlist_*.txt")
            if wordlist_files:
                args.file = max(wordlist_files)
                print(f"자동 선택된 파일: {args.file}")
            else:
                print("대용량 단어 목록 파일을 찾을 수 없습니다.")
                return
        
        generator.enhance_massive_dictionary(args.file, args.chunk)

if __name__ == "__main__":
    main()

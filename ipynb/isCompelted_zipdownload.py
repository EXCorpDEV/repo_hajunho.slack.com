#!/usr/bin/env python3
"""
ZIP 파일 완성도 검사기
현재 폴더의 모든 zip 파일이 완료되었는지 검사합니다.
"""

import os
import zipfile
import glob
from datetime import datetime

def check_zip_file(zip_path):
    """
    ZIP 파일의 완성도를 검사합니다.
    
    Args:
        zip_path (str): ZIP 파일 경로
        
    Returns:
        dict: 검사 결과 정보
    """
    result = {
        'filename': os.path.basename(zip_path),
        'size': 0,
        'is_valid': False,
        'error_message': '',
        'file_count': 0
    }
    
    try:
        # 파일 크기 확인
        result['size'] = os.path.getsize(zip_path)
        
        # ZIP 파일 유효성 검사
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 파일 목록 개수
            result['file_count'] = len(zip_ref.namelist())
            
            # ZIP 파일 무결성 테스트
            bad_file = zip_ref.testzip()
            if bad_file is None:
                result['is_valid'] = True
                result['error_message'] = '완료됨'
            else:
                result['is_valid'] = False
                result['error_message'] = f'손상된 파일: {bad_file}'
                
    except zipfile.BadZipFile:
        result['error_message'] = '잘못된 ZIP 파일 (불완전하거나 손상됨)'
    except FileNotFoundError:
        result['error_message'] = '파일을 찾을 수 없음'
    except PermissionError:
        result['error_message'] = '파일 접근 권한 없음'
    except Exception as e:
        result['error_message'] = f'알 수 없는 오류: {str(e)}'
    
    return result

def format_file_size(size_bytes):
    """파일 크기를 읽기 쉬운 형태로 변환"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def main():
    """메인 함수"""
    print("=" * 70)
    print("ZIP 파일 완성도 검사기")
    print(f"검사 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 현재 폴더에서 모든 zip 파일 찾기
    zip_files = glob.glob("*.zip")
    
    if not zip_files:
        print("현재 폴더에 ZIP 파일이 없습니다.")
        return
    
    print(f"총 {len(zip_files)}개의 ZIP 파일을 발견했습니다.\n")
    
    # 각 ZIP 파일 검사
    valid_count = 0
    invalid_count = 0
    
    for zip_file in sorted(zip_files):
        result = check_zip_file(zip_file)
        
        # 상태 표시
        status_icon = "✅" if result['is_valid'] else "❌"
        status_text = "완료" if result['is_valid'] else "불완전/손상"
        
        print(f"{status_icon} {result['filename']}")
        print(f"   크기: {format_file_size(result['size'])}")
        print(f"   파일 수: {result['file_count']}개")
        print(f"   상태: {status_text}")
        print(f"   메시지: {result['error_message']}")
        print()
        
        if result['is_valid']:
            valid_count += 1
        else:
            invalid_count += 1
    
    # 요약 정보
    print("=" * 70)
    print("검사 결과 요약:")
    print(f"✅ 완료된 파일: {valid_count}개")
    print(f"❌ 불완전/손상된 파일: {invalid_count}개")
    print(f"📊 전체 파일: {len(zip_files)}개")
    
    if invalid_count > 0:
        print("\n⚠️  불완전하거나 손상된 파일이 있습니다. 다시 다운로드하거나 확인이 필요합니다.")
    else:
        print("\n🎉 모든 ZIP 파일이 정상적으로 완료되었습니다!")

if __name__ == "__main__":
    main()


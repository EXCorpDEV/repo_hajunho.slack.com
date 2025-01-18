from huggingface_hub import snapshot_download

# LaSOT 데이터셋 다운로드
snapshot_download(
    repo_id="l-lt/LaSOT",  # LaSOT 데이터셋 ID
    repo_type="dataset",
    local_dir="data/LaSOT"  # 다운로드 경로
)

print("LaSOT 데이터셋 다운로드 완료!")


document.addEventListener('DOMContentLoaded', function() {
    // 모델 데이터 로드
    fetch('models.json')
        .then(response => response.json())
        .then(data => {
            renderModels(data.models);
            setupFilters(data.models);
        })
        .catch(error => {
            console.error('모델 데이터를 로드하는 중 오류가 발생했습니다:', error);
            // 데이터 로드 실패 시 오류 메시지 표시
            document.querySelector('.models-table tbody').innerHTML = 
                '<tr><td colspan="6" style="text-align: center; padding: 30px;">모델 데이터를 로드할 수 없습니다. 나중에 다시 시도해주세요.</td></tr>';
        });
});

// 토스트 메시지 표시 함수
function showToast(message) {
    // 토스트 메시지 요소가 없으면 생성
    let toast = document.querySelector('.toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.className = 'toast';
        document.body.appendChild(toast);
    }
    
    // 메시지 설정 및 표시
    toast.textContent = message;
    toast.style.opacity = 1;
    
    // 3초 후 숨기기
    setTimeout(() => {
        toast.style.opacity = 0;
    }, 3000);
}

// 클립보드에 텍스트 복사 함수
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('마그넷 주소가 복사되었습니다.');
    }).catch(err => {
        console.error('클립보드 복사 실패:', err);
        showToast('클립보드 복사에 실패했습니다.');
    });
}

// 모델 목록 렌더링 함수
function renderModels(models) {
    const tableBody = document.querySelector('.models-table tbody');
    tableBody.innerHTML = '';
    
    models.forEach(model => {
        const row = document.createElement('tr');
        row.dataset.name = model.name.toLowerCase();
        row.dataset.type = model.type;
        row.dataset.license = model.license.replace(/\s+/g, '').toLowerCase();
        
        // 링크 버튼 생성
        let linksHtml = '<div class="link-buttons">';
        
        // Google Drive 링크
        if (model.gdriveUrl) {
            linksHtml += `
                <a href="${model.gdriveUrl}" target="_blank" class="link-btn gdrive-btn">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h9"></path>
                        <line x1="12" y1="10" x2="17" y2="15"></line>
                        <path d="M17 10l5 5-5 5"></path>
                    </svg>
                    탐색
                </a>
            `;
        }
        
        // HuggingFace 링크
        if (model.huggingfaceUrl) {
            linksHtml += `
                <a href="${model.huggingfaceUrl}" target="_blank" class="link-btn hf-btn">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                        <polyline points="15 3 21 3 21 9"></polyline>
                        <line x1="10" y1="14" x2="21" y2="3"></line>
                    </svg>
                    HuggingFace
                </a>
            `;
        }
        
        // GitHub 링크
        if (model.githubUrl) {
            linksHtml += `
                <a href="${model.githubUrl}" target="_blank" class="link-btn github-btn">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
                    </svg>
                    GitHub
                </a>
            `;
        }
        
        // 마그넷 링크
        if (model.magnetUrl) {
            linksHtml += `
                <a href="javascript:void(0)" class="link-btn magnet-btn" onclick="copyToClipboard('${model.magnetUrl}')">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.93 4.93l1.41 1.41m11.32 11.32l1.41 1.41M4.93 19.07l1.41-1.41m11.32-11.32l1.41-1.41"></path>
                        <circle cx="12" cy="12" r="4"></circle>
                    </svg>
                    토렌트
                </a>
            `;
        }
        
        linksHtml += '</div>';
        
        // 모델 타입에 따른 클래스 결정
        let typeClass = '';
        switch(model.type) {
            case 'nlp':
                typeClass = 'type-nlp';
                break;
            case 'cv':
                typeClass = 'type-cv';
                break;
            case 'multimodal':
                typeClass = 'type-multimodal';
                break;
            case 'code':
                typeClass = 'type-code';
                break;
        }
        
        // 모델 타입 표시 이름
        let typeDisplayName = '';
        switch(model.type) {
            case 'nlp':
                typeDisplayName = '자연어 처리';
                break;
            case 'cv':
                typeDisplayName = '이미지 생성';
                break;
            case 'multimodal':
                typeDisplayName = '멀티모달';
                break;
            case 'code':
                typeDisplayName = '코드 생성';
                break;
        }
        
        row.innerHTML = `
            <td>
                <strong>${model.name}</strong>
                <div style="font-size: 13px; color: #666;">${model.description}</div>
            </td>
            <td><span class="model-type ${typeClass}">${typeDisplayName}</span></td>
            <td><span class="license">${model.license}</span></td>
            <td>${model.size}</td>
            <td>${model.updated}</td>
            <td colspan="3">${linksHtml}</td>
        `;
        
        tableBody.appendChild(row);
    });
}

// 필터 기능 설정
function setupFilters(models) {
    const searchInput = document.querySelector('.search-box input');
    const typeFilter = document.querySelector('.filter-select:nth-child(2)');
    const licenseFilter = document.querySelector('.filter-select:nth-child(3)');
    
    // 검색 필터링 함수
    function filterModels() {
        const searchTerm = searchInput.value.toLowerCase();
        const selectedType = typeFilter.value;
        const selectedLicense = licenseFilter.value;
        
        const rows = document.querySelectorAll('.models-table tbody tr');
        
        rows.forEach(row => {
            // 모델명 검색 - 부분 일치도 검색되도록 수정
            const modelName = row.dataset.name;
            const modelType = row.dataset.type;
            const modelLicense = row.dataset.license;
            
            const matchesSearch = searchTerm === '' || modelName.includes(searchTerm);
            const matchesType = selectedType === 'all' || modelType === selectedType;
            const matchesLicense = selectedLicense === 'all' || modelLicense.includes(selectedLicense.toLowerCase());
            
            if (matchesSearch && matchesType && matchesLicense) {
                row.classList.remove('hidden');
            } else {
                row.classList.add('hidden');
            }
        });
    }
    
    // 이벤트 리스너 설정
    searchInput.addEventListener('input', filterModels);
    typeFilter.addEventListener('change', filterModels);
    licenseFilter.addEventListener('change', filterModels);
}

// 전역 함수로 노출 (HTML에서 직접 호출 가능하도록)
window.copyToClipboard = copyToClipboard;

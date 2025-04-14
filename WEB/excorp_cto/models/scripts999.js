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

// 모델 목록 렌더링 함수
function renderModels(models) {
    const tableBody = document.querySelector('.models-table tbody');
    tableBody.innerHTML = '';
    
    models.forEach(model => {
        const row = document.createElement('tr');
        row.dataset.name = model.name.toLowerCase();
        row.dataset.type = model.type;
        row.dataset.license = model.license.replace(/\s+/g, '').toLowerCase();
        
        // 다운로드 링크 타입에 따라 다른 UI 생성
        let downloadLink;
        if (model.downloadType === 'gdrive') {
            downloadLink = `
                <a href="${model.downloadUrl}" target="_blank" class="download-btn">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    다운로드
                </a>
            `;
        } else if (model.downloadType === 'huggingface') {
            downloadLink = `
                <a href="${model.downloadUrl}" target="_blank" class="external-link">
                    HuggingFace
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                        <polyline points="15 3 21 3 21 9"></polyline>
                        <line x1="10" y1="14" x2="21" y2="3"></line>
                    </svg>
                </a>
            `;
        } else {
            downloadLink = `
                <a href="${model.downloadUrl}" target="_blank" class="external-link">
                    GitHub
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                        <polyline points="15 3 21 3 21 9"></polyline>
                        <line x1="10" y1="14" x2="21" y2="3"></line>
                    </svg>
                </a>
            `;
        }
        
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
            <td>${downloadLink}</td>
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

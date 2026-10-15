// ==========================================================================
// Skin Lesion AI Diagnostic Studio • Dynamic Client Application Logic
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Core UI Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const samplesGrid = document.getElementById('samples-grid');
  const emptyState = document.getElementById('empty-state');
  const loadingState = document.getElementById('loading-state');
  const resultsContainer = document.getElementById('results-container');

  // Navigation & Expandable Panels
  const btnToggleAbcde = document.getElementById('btn-toggle-abcde');
  const btnCloseAbcde = document.getElementById('btn-close-abcde');
  const panelAbcde = document.getElementById('panel-abcde');

  const btnToggleTechnical = document.getElementById('btn-toggle-technical');
  const btnCloseTechnical = document.getElementById('btn-close-technical');
  const panelTechnical = document.getElementById('panel-technical');
  const btnPrintReport = document.getElementById('btn-print-report');

  // Camera Modal Elements
  const btnToggleCamera = document.getElementById('btn-toggle-camera');
  const cameraModal = document.getElementById('camera-modal');
  const cameraVideo = document.getElementById('camera-video');
  const cameraCanvas = document.getElementById('camera-canvas');
  const btnCapturePhoto = document.getElementById('btn-capture-photo');
  const btnCancelCamera = document.getElementById('btn-cancel-camera');
  const btnCloseCamera = document.getElementById('btn-close-camera');
  let cameraStream = null;

  // Triage Hero Elements
  const triageCard = document.getElementById('triage-card');
  const triageLevelTag = document.getElementById('triage-level-tag');
  const triageCategoryTag = document.getElementById('triage-category-tag');
  const activeModelTag = document.getElementById('active-model-tag');
  const triageSimpleName = document.getElementById('triage-simple-name');
  const triageMedicalName = document.getElementById('triage-medical-name');
  const triagePlainDesc = document.getElementById('triage-plain-desc');
  const topConfidence = document.getElementById('top-confidence');
  const triageDot = document.getElementById('triage-dot');

  // Visuals & Heatmap Slider Elements
  const originalPreview = document.getElementById('original-preview');
  const gradcamPreview = document.getElementById('gradcam-preview');
  const visualSideBySide = document.getElementById('visual-side-by-side');
  const visualSliderWrap = document.getElementById('visual-slider-wrap');
  const sliderImgOriginal = document.getElementById('slider-img-original');
  const sliderImgCam = document.getElementById('slider-img-cam');
  const sliderAfterCrop = document.getElementById('slider-after-crop');
  const compareSliderInput = document.getElementById('compare-slider-input');
  const sliderDividerLine = document.getElementById('slider-divider-line');

  const viewSplitBtn = document.getElementById('view-split');
  const viewSliderBtn = document.getElementById('view-slider');
  const viewOverlayBtn = document.getElementById('view-overlay');

  // Recommendations Elements
  const urgencyValue = document.getElementById('urgency-value');
  const suggestionsDos = document.getElementById('suggestions-dos');
  const suggestionsDonts = document.getElementById('suggestions-donts');

  // Improper Image Alert Elements
  const alertCard = document.getElementById('alert-card');
  const alertTitle = document.getElementById('alert-title');
  const alertMessage = document.getElementById('alert-message');
  const alertTipsList = document.getElementById('alert-tips-list');
  const btnRetryUpload = document.getElementById('btn-retry-upload');

  // Probabilities & Consensus
  const probList = document.getElementById('prob-list');

  // Application State
  let currentImageSource = null;
  let selectedModel = 'ensemble'; // Default to Optimal Weighted Ensemble (84.4% Benchmark Accuracy)

  // Initialize Modules
  initPanels();
  initTechnicalRadios();
  initUpload();
  initCamera();
  initVisualToggles();
  initClipboardPaste();
  initPrintReport();
  fetchStatusAndSamples();

  // --------------------------------------------------------------------------
  // Navigation & Dropdown Panels
  // --------------------------------------------------------------------------
  function initPanels() {
    if (btnToggleAbcde && panelAbcde) {
      btnToggleAbcde.addEventListener('click', () => {
        panelAbcde.classList.toggle('hidden');
        if (panelTechnical) panelTechnical.classList.add('hidden');
      });
      if (btnCloseAbcde) {
        btnCloseAbcde.addEventListener('click', () => panelAbcde.classList.add('hidden'));
      }
    }

    if (btnToggleTechnical && panelTechnical) {
      btnToggleTechnical.addEventListener('click', () => {
        panelTechnical.classList.toggle('hidden');
        if (panelAbcde) panelAbcde.classList.add('hidden');
      });
      if (btnCloseTechnical) {
        btnCloseTechnical.addEventListener('click', () => panelTechnical.classList.add('hidden'));
      }
    }
  }

  function initPrintReport() {
    if (btnPrintReport) {
      btnPrintReport.addEventListener('click', () => {
        window.print();
      });
    }
  }

  function initTechnicalRadios() {
    const radioLabels = document.querySelectorAll('.tech-radio');
    radioLabels.forEach(label => {
      label.addEventListener('click', () => {
        radioLabels.forEach(l => l.classList.remove('active'));
        label.classList.add('active');
        const radio = label.querySelector('input[type="radio"]');
        if (radio) {
          radio.checked = true;
          selectedModel = radio.value;
          if (currentImageSource) {
            runInference();
          }
        }
      });
    });
  }

  // --------------------------------------------------------------------------
  // Camera Capture Integration
  // --------------------------------------------------------------------------
  function initCamera() {
    if (!btnToggleCamera || !cameraModal) return;

    btnToggleCamera.addEventListener('click', async () => {
      try {
        cameraModal.classList.remove('hidden');
        cameraStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
        });
        cameraVideo.srcObject = cameraStream;
      } catch (err) {
        console.error('Camera access error:', err);
        closeCamera();
        showImageAlert(
          'Camera Access Error',
          'Could not access your device camera. Please check browser permissions or upload a saved photo.',
          ['Grant camera permissions in your browser address bar.', 'Ensure no other application is using the webcam.']
        );
      }
    });

    const closeCamera = () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
      }
      if (cameraVideo) cameraVideo.srcObject = null;
      cameraModal.classList.add('hidden');
    };

    if (btnCancelCamera) btnCancelCamera.addEventListener('click', closeCamera);
    if (btnCloseCamera) btnCloseCamera.addEventListener('click', closeCamera);

    if (btnCapturePhoto) {
      btnCapturePhoto.addEventListener('click', () => {
        if (!cameraVideo.videoWidth) return;
        cameraCanvas.width = cameraVideo.videoWidth;
        cameraCanvas.height = cameraVideo.videoHeight;
        const ctx = cameraCanvas.getContext('2d');
        // Mirror the canvas image to match preview
        ctx.translate(cameraCanvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
        const base64Data = cameraCanvas.toDataURL('image/jpeg', 0.92);
        closeCamera();

        currentImageSource = { type: 'upload', b64: base64Data };
        document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
        runInference();
      });
    }
  }

  // --------------------------------------------------------------------------
  // File Upload, Drag & Drop, and Clipboard Paste
  // --------------------------------------------------------------------------
  function initUpload() {
    if (btnRetryUpload) {
      btnRetryUpload.addEventListener('click', () => {
        alertCard.classList.add('hidden');
        fileInput.click();
      });
    }

    dropzone.addEventListener('click', () => {
      fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) handleFile(file);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const file = dt.files[0];
      if (file) handleFile(file);
    });
  }

  function initClipboardPaste() {
    window.addEventListener('paste', (e) => {
      const items = e.clipboardData && e.clipboardData.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          const file = items[i].getAsFile();
          if (file) {
            handleFile(file);
            break;
          }
        }
      }
    });
  }

  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      showImageAlert(
        'Invalid File Format',
        'Please upload an image file (JPEG, PNG, WebP).',
        ['Files like PDF, video, or documents cannot be analyzed.', 'Select an image ending with .jpg, .jpeg, or .png.']
      );
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const base64Data = e.target.result;
      currentImageSource = { type: 'upload', b64: base64Data };
      document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
      runInference();
    };
    reader.readAsDataURL(file);
  }

  // --------------------------------------------------------------------------
  // Interactive Grad-CAM Heatmap Viewer Modes
  // --------------------------------------------------------------------------
  function initVisualToggles() {
    const updateSliderPosition = (pct) => {
      if (sliderAfterCrop) {
        sliderAfterCrop.style.clipPath = `polygon(0 0, ${pct}% 0, ${pct}% 100%, 0 100%)`;
      }
      if (sliderDividerLine) {
        sliderDividerLine.style.left = `${pct}%`;
      }
    };

    if (compareSliderInput) {
      compareSliderInput.addEventListener('input', (e) => {
        updateSliderPosition(e.target.value);
      });
    }

    if (viewSplitBtn && viewSliderBtn && viewOverlayBtn) {
      viewSplitBtn.addEventListener('click', () => {
        viewSplitBtn.classList.add('active');
        viewSliderBtn.classList.remove('active');
        viewOverlayBtn.classList.remove('active');
        visualSideBySide.classList.remove('hidden');
        visualSliderWrap.classList.add('hidden');
        originalPreview.parentElement.parentElement.classList.remove('hidden');
        gradcamPreview.parentElement.parentElement.classList.remove('hidden');
      });

      viewSliderBtn.addEventListener('click', () => {
        viewSliderBtn.classList.add('active');
        viewSplitBtn.classList.remove('active');
        viewOverlayBtn.classList.remove('active');
        visualSideBySide.classList.add('hidden');
        visualSliderWrap.classList.remove('hidden');
        updateSliderPosition(compareSliderInput.value || 50);
      });

      viewOverlayBtn.addEventListener('click', () => {
        viewOverlayBtn.classList.add('active');
        viewSplitBtn.classList.remove('active');
        viewSliderBtn.classList.remove('active');
        visualSideBySide.classList.remove('hidden');
        visualSliderWrap.classList.add('hidden');
        originalPreview.parentElement.parentElement.classList.add('hidden');
        gradcamPreview.parentElement.parentElement.classList.remove('hidden');
      });
    }
  }

  function showImageAlert(title, message, tips = []) {
    loadingState.classList.add('hidden');
    resultsContainer.classList.add('hidden');
    emptyState.classList.add('hidden');
    if (alertCard) {
      alertCard.classList.remove('hidden');
      alertTitle.textContent = title;
      alertMessage.textContent = message;

      alertTipsList.innerHTML = '';
      (tips || []).forEach(tip => {
        const li = document.createElement('li');
        li.textContent = tip;
        alertTipsList.appendChild(li);
      });

      alertCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // --------------------------------------------------------------------------
  // Benchmark Samples Fetching & Rendering
  // --------------------------------------------------------------------------
  const RISK_FRIENDLY_MAP = {
    nv: { name: 'Normal Mole', risk: 'low' },
    mel: { name: 'Melanoma', risk: 'critical' },
    bcc: { name: 'Basal Cell Carcinoma', risk: 'high' },
    akiec: { name: 'Precancerous Sun Spot', risk: 'moderate' },
    bkl: { name: 'Harmless Age Spot', risk: 'low' },
    df: { name: 'Skin Nodule', risk: 'low' },
    vasc: { name: 'Blood Vessel Mark', risk: 'low' },
  };

  async function fetchStatusAndSamples() {
    try {
      const samplesRes = await fetch('/api/samples');
      const samplesData = await samplesRes.json();
      renderSamples(samplesData.samples || []);
    } catch (err) {
      console.error('Error fetching initial data:', err);
      samplesGrid.innerHTML = '<div class="sample-skeleton">Could not load benchmark samples. Server offline?</div>';
    }
  }

  function renderSamples(samples) {
    if (!samples.length) {
      samplesGrid.innerHTML = '<div class="sample-skeleton">No test samples found.</div>';
      return;
    }

    samplesGrid.innerHTML = '';
    samples.forEach(sample => {
      const friendlyMeta = RISK_FRIENDLY_MAP[sample.dx] || { name: sample.dx_full, risk: 'low' };
      const chip = document.createElement('div');
      chip.className = `sample-chip risk-${friendlyMeta.risk}`;
      chip.dataset.imageId = sample.image_id;

      chip.innerHTML = `
        <img class="chip-thumb" src="/api/image?id=${sample.image_id}" alt="${sample.dx_full}" loading="lazy">
        <div class="chip-text">
          <span class="chip-title">${friendlyMeta.name}</span>
          <span class="chip-code">${sample.dx.toUpperCase()}</span>
        </div>
      `;

      chip.addEventListener('click', () => {
        document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');

        currentImageSource = {
          type: 'sample',
          id: sample.image_id,
          url: `/api/image?id=${sample.image_id}`,
          trueDx: sample.dx,
        };

        runInference();
      });

      samplesGrid.appendChild(chip);
    });
  }

  // --------------------------------------------------------------------------
  // AI Inference & API Call
  // --------------------------------------------------------------------------
  async function runInference() {
    if (!currentImageSource) return;

    // Show loading state
    emptyState.classList.add('hidden');
    resultsContainer.classList.add('hidden');
    if (alertCard) alertCard.classList.add('hidden');
    loadingState.classList.remove('hidden');

    try {
      const requestPayload = { model: selectedModel };
      if (currentImageSource.type === 'sample') {
        requestPayload.image_id = currentImageSource.id;
      } else {
        requestPayload.image_base64 = currentImageSource.b64;
      }

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload),
      });

      const data = await res.json();
      if (!res.ok || data.status === 'improper_image') {
        showImageAlert(
          data.error_title || 'Improper Image Detected',
          data.error_message || data.error || 'The uploaded photo could not be analyzed.',
          data.suggestions || [
            'Ensure the photo is clear, well-lit, and in sharp focus.',
            'Make sure human skin or a skin mole is clearly visible.',
            'Avoid extreme shadows, flash glare, or non-skin objects.',
          ]
        );
        return;
      }

      renderResults(data);
    } catch (err) {
      showImageAlert(
        'Upload / Connection Error',
        err.message || 'Could not communicate with the AI engine.',
        ['Check that the local server is running.', 'Try reloading the page and uploading again.']
      );
    }
  }

  // --------------------------------------------------------------------------
  // Results Dashboard Rendering
  // --------------------------------------------------------------------------
  function renderResults(data) {
    loadingState.classList.add('hidden');
    resultsContainer.classList.remove('hidden');

    const pred = data.prediction;
    const rel = data.reliability;
    const suggestions = pred.suggestions || {
      urgency: 'Consult a dermatologist for routine confirmation.',
      dos: ['Track this mole for changes using the ABCDE criteria.'],
      donts: ['Do not scratch or irritate the spot.'],
    };

    // 1. Triage Hero Card State
    const riskLevel = pred.risk_level || (pred.category.toLowerCase().includes('malignant') ? 'high' : 'low');
    triageCard.className = `result-card triage-card risk-state-${riskLevel}`;

    const riskBadgeText = pred.risk_badge || (riskLevel.toUpperCase() + ' RISK');
    triageLevelTag.textContent = riskBadgeText;

    if (triageDot) {
      const dotColors = {
        healthy: '#10b981',
        low: '#10b981',
        moderate: '#f59e0b',
        high: '#f43f5e',
        critical: '#f43f5e',
      };
      triageDot.style.backgroundColor = dotColors[riskLevel] || '#10b981';
      triageDot.style.color = dotColors[riskLevel] || '#10b981';
    }

    triageCategoryTag.textContent = pred.category.toUpperCase();

    if (activeModelTag) {
      if (data.model_info) {
        const accText = data.model_info.test_acc ? ` (${(data.model_info.test_acc * 100).toFixed(1)}% Acc)` : '';
        activeModelTag.textContent = `${data.model_info.name}${accText}`;
      } else if (data.model_used) {
        activeModelTag.textContent = data.model_used;
      }
    }

    triageSimpleName.textContent = pred.simple_name || pred.class_name;
    triageMedicalName.textContent = `Medical Classification: ${pred.class_name} (${pred.class_code.toUpperCase()})`;
    triagePlainDesc.textContent = pred.plain_explanation || pred.notes;
    topConfidence.textContent = `${(pred.confidence * 100).toFixed(1)}%`;

    // 2. Set Visual Images (Side-by-side & Interactive Slider)
    const imgSrc = (currentImageSource.type === 'sample') ? currentImageSource.url : currentImageSource.b64;
    originalPreview.src = imgSrc;
    gradcamPreview.src = data.gradcam_overlay;

    if (sliderImgOriginal) sliderImgOriginal.src = imgSrc;
    if (sliderImgCam) sliderImgCam.src = data.gradcam_overlay;

    // 3. Set Quantitative ABCDE Metrics
    const abcdeCard = document.getElementById('abcde-metrics-card');
    if (abcdeCard && data.abcde) {
      const abcde = data.abcde;
      const overallBadge = document.getElementById('abcde-overall-badge');
      if (overallBadge) {
        overallBadge.textContent = abcde.composite_risk || 'Standard Profile';
        const isWarning = (abcde.composite_score >= 5.5);
        overallBadge.className = `abcde-overall-badge ${isWarning ? 'badge-warning' : 'badge-safe'}`;
      }

      // Asymmetry
      const asymScore = document.getElementById('asym-score');
      const asymStatus = document.getElementById('asym-status');
      const asymBar = document.getElementById('asym-bar');
      if (asymScore && abcde.asymmetry) {
        asymScore.textContent = `${abcde.asymmetry.score} / 10`;
        asymStatus.textContent = abcde.asymmetry.status;
        asymBar.style.width = `${Math.min(100, abcde.asymmetry.score * 10)}%`;
        asymBar.className = `metric-bar-fill fill-${abcde.asymmetry.level}`;
      }

      // Border
      const borderScore = document.getElementById('border-score');
      const borderStatus = document.getElementById('border-status');
      const borderBar = document.getElementById('border-bar');
      if (borderScore && abcde.border) {
        borderScore.textContent = `${abcde.border.score} / 10`;
        borderStatus.textContent = abcde.border.status;
        borderBar.style.width = `${Math.min(100, abcde.border.score * 10)}%`;
        borderBar.className = `metric-bar-fill fill-${abcde.border.level}`;
      }

      // Color
      const colorScore = document.getElementById('color-score');
      const colorStatus = document.getElementById('color-status');
      const colorBar = document.getElementById('color-bar');
      if (colorScore && abcde.color) {
        colorScore.textContent = `${abcde.color.score} / 10`;
        colorStatus.textContent = abcde.color.status;
        colorBar.style.width = `${Math.min(100, abcde.color.score * 10)}%`;
        colorBar.className = `metric-bar-fill fill-${abcde.color.level}`;
      }

      // Diameter
      const diamScore = document.getElementById('diameter-score');
      const diamStatus = document.getElementById('diameter-status');
      const diamBar = document.getElementById('diameter-bar');
      if (diamScore && abcde.diameter) {
        diamScore.textContent = `${abcde.diameter.coverage_pct}% FOV`;
        diamStatus.textContent = abcde.diameter.status;
        diamBar.style.width = `${Math.min(100, abcde.diameter.coverage_pct * 1.5)}%`;
        diamBar.className = `metric-bar-fill fill-${abcde.diameter.level}`;
      }
    }

    // 4. Set Actionable Suggestions & Timeline
    urgencyValue.textContent = suggestions.urgency || 'Schedule a routine consultation.';

    suggestionsDos.innerHTML = '';
    (suggestions.dos || []).forEach(item => {
      const li = document.createElement('li');
      li.textContent = item;
      suggestionsDos.appendChild(li);
    });

    suggestionsDonts.innerHTML = '';
    (suggestions.donts || []).forEach(item => {
      const li = document.createElement('li');
      li.textContent = item;
      suggestionsDonts.appendChild(li);
    });

    // 5. Set Reliability Guardrails & Multi-Architecture Consensus
    const reliabilityCard = document.getElementById('reliability-card');
    const reliabilityIcon = document.getElementById('reliability-icon');
    const reliabilityTitle = document.getElementById('reliability-title');
    const reliabilityReason = document.getElementById('reliability-reason');
    const consensusGrid = document.getElementById('consensus-grid');

    if (rel) {
      reliabilityCard.className = `subcard status-${rel.level === 'warning' ? 'warning' : 'verified'}`;
      reliabilityIcon.textContent = rel.level === 'warning' ? '⚠️' : '✅';
      reliabilityTitle.textContent = rel.status;
      reliabilityReason.textContent = rel.reason;

      consensusGrid.innerHTML = '';
      if (rel.ensemble && rel.ensemble.length) {
        rel.ensemble.forEach(item => {
          const itemDiv = document.createElement('div');
          itemDiv.className = 'consensus-item';
          const epText = item.epoch ? ` • Ep ${item.epoch}` : '';
          const accText = item.test_acc ? ` • ${(item.test_acc * 100).toFixed(1)}% Acc` : '';
          itemDiv.innerHTML = `
            <div class="consensus-model-name">${item.name}<small style="display:block;color:var(--text-muted);font-size:0.75rem;">${epText}${accText}</small></div>
            <div class="consensus-pred-val">
              <span>${item.class_code.toUpperCase()}</span>
              <span class="conf">${(item.confidence * 100).toFixed(1)}%</span>
            </div>
          `;
          consensusGrid.appendChild(itemDiv);
        });
      }
    }

    // 6. Set 7-Class Softmax Probability Distribution
    probList.innerHTML = '';
    data.probabilities.forEach(item => {
      const probPercent = (item.probability * 100).toFixed(1);
      const isTop = item.is_top;

      const probRow = document.createElement('div');
      probRow.className = `prob-item ${isTop ? 'is-top' : ''}`;
      probRow.innerHTML = `
        <div class="prob-header">
          <span class="prob-name">${item.class_name} (${item.class_code.toUpperCase()})</span>
          <span class="prob-val">${probPercent}%</span>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" style="width: 0%"></div>
        </div>
      `;

      probList.appendChild(probRow);

      // Smooth bar fill animation
      requestAnimationFrame(() => {
        setTimeout(() => {
          const fill = probRow.querySelector('.prob-bar-fill');
          if (fill) fill.style.width = `${probPercent}%`;
        }, 50);
      });
    });

    // Smooth scroll down to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
});

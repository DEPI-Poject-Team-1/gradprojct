// ==================== DOM Elements ====================
// Mode Elements
const letterModeBtn = document.getElementById('letterModeBtn');
const videoModeBtn = document.getElementById('videoModeBtn');
const modeStatus = document.getElementById('modeStatus');

// Modal Elements
const devModal = document.getElementById('devModal');
const modalCloseBtn = document.getElementById('modalCloseBtn');
const continueBtn = document.getElementById('continueBtn');

// Camera & Detection Elements
const cameraSection = document.getElementById('cameraSection');
const controlsSection = document.getElementById('controlsSection');
const statsSection = document.getElementById('statsSection');
const sentenceSection = document.getElementById('sentenceSection');
const clearBtn = document.getElementById('clearBtn');

// Control Elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDisplay = document.getElementById('statusDisplay');

// Display Elements
const fpsDisplay = document.getElementById('fpsDisplay');
const confidenceDisplay = document.getElementById('confidenceDisplay');
const detectedLetter = document.getElementById('detectedLetter');
const sentenceOutput = document.getElementById('sentenceOutput');

// ==================== State Management ====================
const state = {
    currentMode: 'letter',
    isStreaming: false,
    sentenceBuffer: [],
    model: null,
    predHistory: [],
    stableLetter: '--',
    fps: 0
};

// Constants
const SMOOTH_WINDOW = 8;
let detectionInterval = null;
let videoElement = null;
let lastFrameTime = performance.now();
let frameCount = 0;

// ==================== Modal Functions ====================
function showDevModal() {
    devModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function hideDevModal() {
    devModal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

// ==================== Mode Switching ====================
function switchToLetterMode() {
    state.currentMode = 'letter';
    
    if (state.isStreaming) {
        stopDetection();
    }
    
    state.sentenceBuffer = [];
    state.predHistory = [];
    sentenceOutput.textContent = 'Your translated text will appear here...';
    
    modeStatus.textContent = '🟢 Letter Detection Mode Active';
    letterModeBtn.className = 'primary-btn';
    videoModeBtn.className = 'secondary-btn';
    
    cameraSection.classList.add('hidden-section');
    statsSection.classList.add('hidden-section');
    sentenceSection.classList.add('hidden-section');
    clearBtn.classList.add('hidden-section');
    controlsSection.classList.remove('hidden-section');
    
    statusDisplay.textContent = '🔴 Ready to start detection';
    statusDisplay.style.color = '';
    
    fpsDisplay.textContent = '0';
    confidenceDisplay.textContent = '0%';
    confidenceDisplay.style.color = '';
    detectedLetter.textContent = '--';
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    startBtn.classList.replace('secondary-btn', 'primary-btn');
    stopBtn.classList.replace('primary-btn', 'secondary-btn');
}

function switchToVideoMode() {
    showDevModal();
    letterModeBtn.className = 'primary-btn';
    videoModeBtn.className = 'secondary-btn';
    modeStatus.textContent = '🟢 Letter Detection Mode Active';
}

// ==================== Model Loading ====================
async function loadModel() {
    statusDisplay.textContent = '🔴 Ready to start detection';
    statusDisplay.style.color = '';
    return true;
}

// ==================== Camera Functions ====================
async function startCamera() {
    try {
        videoElement = document.createElement('video');
        videoElement.autoplay = true;
        videoElement.playsInline = true;
        videoElement.width = 640;
        videoElement.height = 480;
        videoElement.style.display = 'block';

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        videoElement.srcObject = stream;

        const rawCamFrame = document.getElementById('rawCamFrame');
        rawCamFrame.innerHTML = ''; // remove placeholder
        rawCamFrame.appendChild(videoElement);

        await new Promise(resolve => {
            videoElement.onloadedmetadata = () => resolve();
        });

        return true;
    } catch (error) {
        console.error('❌ Camera error:', error);
        statusDisplay.textContent = '❌ Camera not available';
        statusDisplay.style.color = '#ff6b6b';
        return false;
    }
}

function stopCamera() {
    if (videoElement && videoElement.srcObject) {
        const tracks = videoElement.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        videoElement.srcObject = null;
    }
}

// ==================== Helper Functions ====================
function getStablePrediction() {
    if (state.predHistory.length === 0) return 'nothing';
    
    const counts = {};
    state.predHistory.forEach(pred => {
        counts[pred] = (counts[pred] || 0) + 1;
    });
    
    return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
}

function updateSentence(pred, confidence) {
    // Only add to sentence if confidence is high enough
    if (confidence < 70) return;
    
    if (pred === 'del') {
        if (state.sentenceBuffer.length > 0) {
            state.sentenceBuffer.pop();
        }
    } else if (pred === 'space') {
        state.sentenceBuffer.push(' ');
    } else if (pred !== 'nothing' && pred !== 'unknown' && pred !== '--') {
        if (state.sentenceBuffer.length === 0 || 
            state.sentenceBuffer[state.sentenceBuffer.length - 1] !== pred) {
            state.sentenceBuffer.push(pred);
            
            if (state.sentenceBuffer.length > 20) {
                state.sentenceBuffer.shift();
            }
        }
    }
}

// ==================== Display Processed Image ====================
function displayProcessedImage(base64ImageData) {
    const processedCamFrame = document.getElementById('processedCamFrame');
    if (!processedCamFrame) return;

    let processedImg = document.getElementById('processedImage');
    if (!processedImg) {
        processedImg = document.createElement('img');
        processedImg.id = 'processedImage';
        processedImg.style.width = '100%';
        processedImg.style.borderRadius = '10px';
        processedImg.style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';

        processedCamFrame.innerHTML = ''; // Clear placeholder
        processedCamFrame.appendChild(processedImg);
    }

    processedImg.src = base64ImageData;
}

function updateUIDisplay(result, stablePred) {
    // Update detected letter with stable prediction
    detectedLetter.textContent = stablePred;
    
    // Update confidence
    confidenceDisplay.textContent = `${result.confidence.toFixed(1)}%`;
    
    // Color code confidence
    if (result.confidence > 85) {
        confidenceDisplay.style.color = '#51cf66';
    } else if (result.confidence > 70) {
        confidenceDisplay.style.color = '#ffd43b';
    } else {
        confidenceDisplay.style.color = '#ff6b6b';
    }
    
    // Update sentence
    sentenceOutput.textContent = state.sentenceBuffer.join('');
    
    // Add current detection info if not already present
    let currentDet = document.getElementById('currentDetection');
    if (!currentDet) {
        currentDet = document.createElement('div');
        currentDet.id = 'currentDetection';
        currentDet.style.fontSize = '0.9rem';
        currentDet.style.color = '#888';
        currentDet.style.marginTop = '5px';
        document.querySelector('.letter-display').appendChild(currentDet);
    }
    
    if (result.detected_letter && result.detected_letter !== '--') {
        currentDet.textContent = `Raw: ${result.detected_letter}`;
    } else {
        currentDet.textContent = '';
    }
}

// ==================== Detection Functions ====================
// ==================== Detection Functions ====================
async function predictLetter() {
    if (!videoElement) return null;

    try {
        // Capture current frame
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

        // Convert to blob
        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', 0.8);
        });

        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        console.log('📤 Sending frame for detection...');
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log('📥 API Response:', result);
        
        if (result.success) {
            // Display the processed image with landmarks
            if (result.processed_image) {
                displayProcessedImage(result.processed_image);
            }
            
            // Update state
            if (result.detected_letter !== "nothing") {
                state.predHistory.push(result.detected_letter);
                if (state.predHistory.length > SMOOTH_WINDOW) {
                    state.predHistory.shift();
                }
            }
            
            // Update UI displays
            detectedLetter.textContent = result.detected_letter;
            confidenceDisplay.textContent = `${result.confidence.toFixed(1)}%`;
            
            // Update sentence
            updateSentence(result.detected_letter, result.confidence);
            sentenceOutput.textContent = state.sentenceBuffer.join('');
            
            // Color code confidence
            if (result.confidence > 85) {
                confidenceDisplay.style.color = '#51cf66';
            } else if (result.confidence > 70) {
                confidenceDisplay.style.color = '#ffd43b';
            } else {
                confidenceDisplay.style.color = '#ff6b6b';
            }
            
            // Update current detection info
            let currentDet = document.getElementById('currentDetection');
            if (!currentDet) {
                currentDet = document.createElement('div');
                currentDet.id = 'currentDetection';
                currentDet.style.fontSize = '0.9rem';
                currentDet.style.color = '#888';
                currentDet.style.marginTop = '5px';
                document.querySelector('.letter-display').appendChild(currentDet);
            }
            
            if (result.detected_letter && result.detected_letter !== '--') {
                currentDet.textContent = `Confidence: ${result.confidence.toFixed(1)}%`;
            } else {
                currentDet.textContent = 'No hand detected';
            }
            
            return {
                letter: result.detected_letter,
                confidence: result.confidence
            };
        } else {
            console.error('Detection failed:', result.error);
            confidenceDisplay.textContent = '0%';
            confidenceDisplay.style.color = '#ff6b6b';
            detectedLetter.textContent = 'error';
        }
    } catch (error) {
        console.error('Prediction error:', error);
        confidenceDisplay.textContent = '0%';
        confidenceDisplay.style.color = '#ff6b6b';
        detectedLetter.textContent = 'error';
    }

    return null;
}

// ==================== Start Detection ====================
async function startDetection() {
    if (state.isStreaming) return;

    const modelLoaded = await loadModel();
    if (!modelLoaded) return;

    const cameraStarted = await startCamera();
    if (!cameraStarted) return;

    state.isStreaming = true;
    state.sentenceBuffer = [];
    state.predHistory = [];
    state.stableLetter = '--';

    startBtn.disabled = true;
    stopBtn.disabled = false;

    // Show camera & stats sections
    cameraSection.classList.remove('hidden-section');
    statsSection.classList.remove('hidden-section');
    sentenceSection.classList.remove('hidden-section');
    clearBtn.classList.remove('hidden-section');

    statusDisplay.textContent = '🟢 Detection Active — Show your hand!';
    statusDisplay.style.color = '#51cf66';
    sentenceOutput.textContent = '';

    startBtn.classList.replace('primary-btn', 'secondary-btn');
    stopBtn.classList.replace('secondary-btn', 'primary-btn');

    lastFrameTime = performance.now();
    frameCount = 0;

    // Start detection loop
    detectionInterval = setInterval(async () => {
        if (!state.isStreaming) return;

        // Calculate FPS
        const now = performance.now();
        frameCount++;
        if (now - lastFrameTime >= 1000) {
            const fps = Math.round((frameCount * 1000) / (now - lastFrameTime));
            fpsDisplay.textContent = fps;
            state.fps = fps;
            lastFrameTime = now;
            frameCount = 0;
        }

        // Make prediction
        await predictLetter();
    }, 100);
}

// ==================== Stop Detection ====================
function stopDetection() {
    if (!state.isStreaming) return;
    
    state.isStreaming = false;
    
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
    
    stopCamera();
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    
    cameraSection.classList.add('hidden-section');
    statsSection.classList.add('hidden-section');
    sentenceSection.classList.add('hidden-section');
    clearBtn.classList.add('hidden-section');
    
    statusDisplay.textContent = '🔴 Detection Stopped';
    statusDisplay.style.color = '#ff6b6b';
    
    fpsDisplay.textContent = '0';
    confidenceDisplay.textContent = '0%';
    confidenceDisplay.style.color = '';
    detectedLetter.textContent = '--';
    
    sentenceOutput.textContent = 'Your translated text will appear here...';
    
    startBtn.classList.replace('secondary-btn', 'primary-btn');
    stopBtn.classList.replace('primary-btn', 'secondary-btn');
    
    // Clear current detection text
    const currentDet = document.getElementById('currentDetection');
    if (currentDet) {
        currentDet.textContent = '';
    }
}

// ==================== Clear Sentence ====================
function clearSentence() {
    state.sentenceBuffer = [];
    state.predHistory = [];
    sentenceOutput.textContent = 'Your translated text will appear here...';
    detectedLetter.textContent = '--';
    
    const currentDet = document.getElementById('currentDetection');
    if (currentDet) {
        currentDet.textContent = '';
    }
}

// ==================== Event Listeners ====================
letterModeBtn.addEventListener('click', switchToLetterMode);
videoModeBtn.addEventListener('click', switchToVideoMode);
startBtn.addEventListener('click', startDetection);
stopBtn.addEventListener('click', stopDetection);
clearBtn.addEventListener('click', clearSentence);
modalCloseBtn.addEventListener('click', hideDevModal);
continueBtn.addEventListener('click', hideDevModal);

devModal.addEventListener('click', (e) => {
    if (e.target === devModal) {
        hideDevModal();
    }
});

// ==================== Keyboard Shortcuts ====================
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.key === ' ') {
        e.preventDefault();
    }
    
    if (e.ctrlKey && e.key === 'l') {
        switchToLetterMode();
    }
    
    if (e.ctrlKey && e.key === 'v') {
        switchToVideoMode();
    }
    
    if (e.key === ' ' && state.currentMode === 'letter') {
        if (state.isStreaming) {
            stopDetection();
        } else {
            startDetection();
        }
    }
    
    if (e.key === 'Escape') {
        if (state.isStreaming) {
            stopDetection();
        }
        if (devModal.style.display === 'flex') {
            hideDevModal();
        }
    }
    
    if (e.ctrlKey && e.key === 'c') {
        clearSentence();
    }
});

// ==================== Initialize ====================
document.addEventListener('DOMContentLoaded', () => {
    switchToLetterMode();
    
    if (typeof tf !== 'undefined') {
        tf.setBackend('webgl').then(() => {
            console.log('✅ TensorFlow.js backend initialized');
        }).catch(error => {
            console.warn('⚠️ WebGL not available, falling back to CPU');
            console.log('✅ TensorFlow.js backend initialized (CPU)');
        });
    }
    
    setTimeout(() => {
        console.log('ASL Detection System Ready');
        console.log('📝 Available shortcuts:');
        console.log('  Ctrl+L - Switch to Letter Mode');
        console.log('  Ctrl+V - Switch to Video Mode');
        console.log('  Space  - Start/Stop Detection');
        console.log('  Escape - Stop/Close Modal');
        console.log('  Ctrl+C - Clear Sentence');
    }, 1000);
});

// ==================== Cleanup ====================
window.addEventListener('beforeunload', () => {
    if (state.isStreaming) {
        stopDetection();
    }
    
    if (state.model && typeof tf !== 'undefined') {
        state.model.dispose();
    }
    
    if (typeof tf !== 'undefined') {
        tf.disposeVariables();
    }
});
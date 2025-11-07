// Enhanced dashboard with intelligent notification system and spoof detection
const token = localStorage.getItem('token');
const user = JSON.parse(localStorage.getItem('user') || '{}');

if (!token || !user.id || user.role !== 'admin') {
  localStorage.removeItem('token');
  localStorage.removeItem('user');
  window.location.href = '/login';
}

document.addEventListener("DOMContentLoaded", () => {
  // Generate animated particles
  const particlesContainer = document.getElementById('particles');
  for (let i = 0; i < 30; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    particle.style.left = Math.random() * 100 + '%';
    particle.style.top = Math.random() * 100 + '%';
    particle.style.animationDelay = Math.random() * 20 + 's';
    particle.style.animationDuration = (15 + Math.random() * 10) + 's';
    particlesContainer.appendChild(particle);
  }

  window.dashboard = new AttendanceDashboard();
});

// === SUCCESS SOUND ===
function playSuccessSound() {
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const oscillator = audioContext.createOscillator();
  const gainNode = audioContext.createGain();
  
  oscillator.connect(gainNode);
  gainNode.connect(audioContext.destination);
  
  oscillator.frequency.value = 523.25;
  oscillator.type = 'sine';
  
  gainNode.gain.setValueAtTime(0, audioContext.currentTime);
  gainNode.gain.linearRampToValueAtTime(0.3, audioContext.currentTime + 0.01);
  gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
  
  oscillator.start(audioContext.currentTime);
  oscillator.stop(audioContext.currentTime + 0.5);
}

// === MAIN DASHBOARD CLASS ===
class AttendanceDashboard {
  constructor() {
    this.socket = io();
    this.isSystemRunning = false;
    this.videoStream = null;
    this.liveStream = null;
    this.processingInterval = null;
    this.processingCanvas = null;
    this.cameraStopRequested = false;
    
    // Notification state management
    this.currentNotificationState = null;
    this.lastNotificationTime = 0;
    this.notificationDebounce = 1000; // 1 second debounce

    // === ORIGINAL DOM ELEMENTS ===
    this.captureBtn = document.getElementById("capturePhoto");
    this.recaptureBtn = document.getElementById("recapturePhoto");
    this.videoPreview = document.getElementById("videoPreview");
    this.photoCanvas = document.getElementById("photoCanvas");
    this.enrollSubmitBtn = document.getElementById("enrollSubmitBtn");
    this.enrollModal = document.getElementById("enroll-modal-backdrop");
    this.enrollForm = document.getElementById("enrollForm");

    this.startBtn = document.getElementById("startSystem");
    this.stopBtn = document.getElementById("stopSystem");
    this.statusIndicator = document.getElementById("status-indicator");

    this.liveFeedVideo = document.getElementById("liveFeedVideo");
    this.feedPlaceholder = document.getElementById("feedPlaceholder");

    this.recognitionOverlay = document.getElementById("recognitionOverlay");
    this.recognitionOverlayText = document.getElementById("recognitionOverlayText");
    this.overlayTimeout = null;

    this.presentCountEl = document.getElementById("presentCount");
    this.absentCountEl = document.getElementById("absentCount");
    this.attendanceRateEl = document.getElementById("attendanceRate");
    this.totalStudentsEl = document.getElementById("totalStudents");

    this.recentEventsList = document.getElementById("recentEvents");

    this.processingCanvas = document.createElement('canvas');
    this.processingCanvas.width = 320;  // Reduced for better performance
    this.processingCanvas.height = 240;

    // === NEW DOM ELEMENTS (from Claude) ===
    this.qualityFeedback = document.getElementById('qualityFeedback');
    this.qualityMessage = document.getElementById('qualityMessage');
    this.qualityScoreFill = document.getElementById('qualityScoreFill');
    this.qualityItems = document.getElementById('qualityItems');
    this.countdownOverlay = document.getElementById('countdownOverlay');
    this.frameIndicator = document.getElementById('frameIndicator');
    this.captureProgress = document.getElementById('captureProgress');
    this.frameCount = document.getElementById('frameCount');
    
    // === NEW STATE VARIABLES (from Claude) ===
    this.isCapturing = false;
    this.capturedFrames = [];
    this.targetFrameCount = 7;
    this.qualityCheckInterval = null;
    this.captureInterval = null;

    this.initializeEventListeners();
    this.loadInitialData();
  }

  initializeEventListeners() {
    this.startBtn.addEventListener("click", () => this.startSystem());
    this.stopBtn.addEventListener("click", () => this.stopSystem());

    document.getElementById('logout-btn').addEventListener('click', () => {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    });

    document.getElementById("enroll-btn").addEventListener("click", () => this.openModal());
    document.getElementById("close-modal-btn").addEventListener("click", () => this.closeModal());
    document.getElementById("cancel-modal-btn").addEventListener("click", () => this.closeModal());

    // === MODIFIED/NEW EVENT LISTENERS ===
    // REPLACED: captureBtn listener now calls multi-shot
    this.captureBtn.addEventListener("click", () => this.startMultiShotCapture());
    // REPLACED: recaptureBtn now resets the multi-shot UI and restarts camera
    this.recaptureBtn.addEventListener("click", () => {
        this.resetCaptureUI(); // Resets buttons
        this.startVideoStream(); // Restarts camera
        this.startQualityMonitoring(); // Restarts quality check
    });
    // REMOVED: this.enrollForm.addEventListener("submit", ...)
    // The new logic submits programmatically via this.submitEnrollment()
    // We add a listener to the submit button just in case, but it's disabled
    // until capture is complete.
    this.enrollSubmitBtn.addEventListener("click", (e) => {
        e.preventDefault(); // Prevent default form submission
        if (this.capturedFrames.length >= this.targetFrameCount) {
            this.submitEnrollment();
        } else {
            this.showNotification("Please capture photos first.", "warning");
        }
    });
    // === END OF MODIFICATIONS ===

    // Enhanced validation for form inputs
    document.getElementById("student-name").addEventListener("input", (e) => {
      e.target.value = e.target.value.replace(/[^A-Za-z\s]/g, '');
    });

    document.getElementById("parent-phone").addEventListener("input", (e) => {
      e.target.value = e.target.value.replace(/\D/g, '');
    });

    // SocketIO Listeners with intelligent state handling
    this.socket.on("recognition_status", (data) => {
      this.handleRecognitionStatus(data);
    });

    this.socket.on("attendance_update", (data) => {
      console.log('✅ Attendance update received:', data);
      this.handleAttendanceUpdate(data);
    });
    
    this.socket.on("recent_event", (data) => {
      this.handleRecentEvent(data);
    });
    
    // NEW: Listen for spoof/activity updates
    this.socket.on('activity_update', (data) => {
      this.handleActivityUpdate(data);
    });
    
    this.socket.on("system_started", () => this.updateSystemStatus(true));
    this.socket.on("system_stopped", () => this.updateSystemStatus(false));
  }

  handleActivityUpdate(data) {
    // (This function is from your original file, no changes needed)
    console.log('📊 Activity update:', data);
    
    const statusColors = {
      'ok': 'green',
      'flagged_for_review': 'yellow',
      'blocked': 'red'
    };
    
    const color = statusColors[data.status] || 'gray';
    const badge = `<span style="background:${color}; padding:2px 8px; border-radius:4px; color:white; font-size:0.85rem;">${data.status}</span>`;
    
    let message = `${data.name || 'Unknown'} - ${badge}`;
    if (data.spoof_type) {
      message += ` <small style="color:var(--color-text-secondary);">(${data.spoof_type}, conf=${(data.spoof_confidence || 0).toFixed(2)})</small>`;
    }
    
    this.addRecentEvent({
      student_name: message,
      time_in: data.timestamp
    });
    
    if (data.details) {
      const lastItem = this.recentEventsList.firstChild;
      if (lastItem) {
        lastItem.title = JSON.stringify(data.details, null, 2);
      }
    }
  }

  handleRecognitionStatus(data) {
    // (This function is from your original file, no changes needed)
    const now = Date.now();
    
    if (now - this.lastNotificationTime < this.notificationDebounce && 
        data.status === this.currentNotificationState) {
      return;
    }
    
    this.lastNotificationTime = now;
    this.currentNotificationState = data.status;
    
    switch(data.status) {
      case 'clear':
        this.showOverlay(null);
        break;
      case 'obstructed':
        this.showOverlay('⚠️ Camera is obstructed - Please remove obstruction', 'error', 10000);
        break;
      case 'unknown':
        this.showOverlay(data.message || 'Face not recognized', 'error', 3000);
        break;
      case 'verifying':
        this.showOverlay(data.message, 'recognizing', 5000);
        break;
      case 'already_marked':
        this.showOverlay(data.message, 'error', 3000);
        break;
      case 'cooldown':
        this.showOverlay(data.message, 'error', 2000);
        break;
      case 'error':
        this.showOverlay(data.message, 'error', 3000);
        break;
    }
  }

  handleRecentEvent(data) {
    // (This function is from your original file, no changes needed)
    const eventMessages = {
      'camera_obstructed': '🚫 Camera feed obstructed',
      'camera_resumed': '✅ Camera feed restored'
    };
    
    const message = eventMessages[data.type] || data.message;
    
    this.addRecentEvent({
      student_name: message,
      time_in: data.timestamp
    });
    
    console.log(`📋 Event logged: ${message}`);
  }

  // === Video Stream Methods ===

  async startVideoStream() {
    // (Original function from dashboard.js)
    if (this.videoStream) return;
    try {
      this.videoStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      this.videoPreview.srcObject = this.videoStream;
      this.videoPreview.style.display = "block";
      this.videoPreview.parentElement.querySelector('.feed-placeholder').style.display = "none";
    } catch (error) {
      this.showNotification(
        "Camera access denied. Please allow camera permissions.",
        "error"
      );
      this.closeModal();
    }
  }

  stopVideoStream() {
    // (Original function from dashboard.js)
    if (this.videoStream) {
      this.videoStream.getTracks().forEach((track) => {
        track.stop();
        console.log('Enrollment camera track stopped:', track.label);
      });
      this.videoStream = null;
    }
    if (this.videoPreview) {
      this.videoPreview.srcObject = null;
    }
  }

  async startLiveFeed() {
    // (Original function from dashboard.js)
    if (this.liveStream) {
      console.log('Live feed already running');
      return;
    }
    
    this.cameraStopRequested = false;
    
    try {
      console.log('Requesting camera access...');
      this.liveStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      
      this.liveFeedVideo.srcObject = this.liveStream;
      this.liveFeedVideo.style.display = "block";
      this.feedPlaceholder.style.display = "none";

      this.startFrameProcessing();
      console.log('Live camera started successfully');
    } catch (error) {
      console.error('Camera access error:', error);
      this.showNotification(
        "Camera access denied. Please allow camera permissions.",
        "error"
      );
      this.updateSystemStatus(false);
    }
  }

  stopLiveFeed() {
    // (Original function from dashboard.js)
    console.log('=== STOPPING LIVE FEED ===');
    
    this.cameraStopRequested = true;
    this.stopFrameProcessing();

    if (this.liveStream) {
      console.log('Stopping media stream tracks...');
      const tracks = this.liveStream.getTracks();
      tracks.forEach((track) => {
        console.log(`Stopping track: ${track.label} (state: ${track.readyState})`);
        track.stop();
      });
      this.liveStream = null;
      console.log('All tracks stopped');
    }
    
    if (this.liveFeedVideo) {
      console.log('Clearing video element...');
      this.liveFeedVideo.srcObject = null;
      this.liveFeedVideo.style.display = "none";
      this.liveFeedVideo.pause();
      this.liveFeedVideo.load();
    }
    
    if (this.feedPlaceholder) {
      this.feedPlaceholder.style.display = "flex";
    }
    
    this.showOverlay(null);
    
    console.log('=== LIVE FEED STOPPED SUCCESSFULLY ===');
  }

  startFrameProcessing() {
    // (Original function from dashboard.js)
    if (this.processingInterval) {
      console.log('Frame processing already running');
      return;
    }
    
    console.log('Starting frame processing...');
    this.processingInterval = setInterval(() => {
      if (!this.cameraStopRequested && this.isSystemRunning) {
        this.captureAndSendFrame();
      }
    }, 333);
  }

  stopFrameProcessing() {
    // (Original function from dashboard.js)
    if (this.processingInterval) {
      console.log('Stopping frame processing...');
      clearInterval(this.processingInterval);
      this.processingInterval = null;
      console.log('Frame processing stopped');
    }
  }

  captureAndSendFrame() {
    // (Original function from dashboard.js)
    if (!this.liveFeedVideo || !this.liveStream || !this.isSystemRunning || this.cameraStopRequested) {
      return;
    }

    try {
      const ctx = this.processingCanvas.getContext('2d');
      ctx.drawImage(
        this.liveFeedVideo,
        0, 0,
        this.processingCanvas.width,
        this.processingCanvas.height
      );
      const frameData = this.processingCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];
      this.socket.emit('process_frame', { frame: frameData });
    } catch (error) {
      console.error('Error capturing frame:', error);
    }
  }

  // === Enrollment Modal Methods (MERGED) ===

  async openModal() {
    if (this.isSystemRunning) {
      this.showNotification(
        "Please stop the system before enrolling a new student.",
        "warning"
      );
      return;
    }
    this.enrollModal.classList.add('show');
    this.resetCaptureUI(); // Resets buttons and canvas
    await this.startVideoStream(); // Starts the camera
    
    // NEW: Start quality monitoring
    // Small delay to ensure camera is ready
    setTimeout(() => {
        this.startQualityMonitoring();
    }, 500);
  }

  closeModal() {
    this.stopVideoStream(); // Stops the camera
    this.stopQualityMonitoring(); // NEW: Stops quality checks
    this.resetCapture(); // NEW: Resets capture state
    
    this.enrollModal.classList.remove('show');
    this.enrollForm.reset();
    
    const context = this.photoCanvas.getContext("2d");
    context.clearRect(0, 0, this.photoCanvas.width, this.photoCanvas.height);
  }

  resetCaptureUI() {
    // This function is now used to reset the UI for a new session
    this.videoPreview.style.display = "block";
    this.photoCanvas.style.display = "none";

    this.captureBtn.classList.remove("hidden");
    this.recaptureBtn.classList.add("hidden");

    this.enrollSubmitBtn.disabled = true;
    this.captureBtn.disabled = false;
    this.enrollSubmitBtn.innerHTML = '<i class="fas fa-check"></i> Submit';

    // NEW: Reset multi-shot state
    this.resetCapture();
  }

  // === Data Loading & API Methods ===
  // (These functions are from your original file, no changes needed)

  async loadInitialData() {
    this.loadStats();
    this.loadRecentAttendance();
  }

  async loadStats() {
    try {
      const response = await fetch("/api/stats", {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await response.json();
      this.presentCountEl.textContent = data.present_today;
      this.absentCountEl.textContent = data.absent_today;
      this.attendanceRateEl.textContent = `${Math.round(data.attendance_rate)}%`;
      this.totalStudentsEl.textContent = data.total_students;
    } catch (error) {
      console.error("Error loading stats:", error);
    }
  }

  async loadRecentAttendance() {
    try {
      const response = await fetch("/api/attendance", {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await response.json();
      this.recentEventsList.innerHTML = "";
      if (data.length === 0) {
        this.recentEventsList.innerHTML =
          '<li style="text-align: center; color: var(--color-text-secondary)">No events yet</li>';
        return;
      }
      data.forEach((event) => {
        this.addRecentEvent(event);
      });
    } catch (error) {
      console.error("Error loading recent attendance:", error);
    }
  }

  // === System & Socket Methods ===
  // (These functions are from your original file, no changes needed)

  startSystem() {
    console.log('Starting system...');
    this.socket.emit("start_system");
    this.startBtn.disabled = true;
    this.stopBtn.disabled = false;
  }

  stopSystem() {
    console.log('Stop button clicked');
    this.socket.emit("stop_system");
    this.stopBtn.disabled = true;
    this.startBtn.disabled = false;
    this.stopLiveFeed();
  }

  updateSystemStatus(isRunning) {
    console.log('System status update:', isRunning);
    this.isSystemRunning = isRunning;
    
    if (isRunning) {
      this.startBtn.disabled = true;
      this.stopBtn.disabled = false;
      this.statusIndicator.classList.remove("status-offline");
      this.statusIndicator.classList.add("status-online");
      this.statusIndicator.textContent = "● Online";
      this.startLiveFeed();
    } else {
      this.startBtn.disabled = false;
      this.stopBtn.disabled = true;
      this.statusIndicator.classList.remove("status-online");
      this.statusIndicator.classList.add("status-offline");
      this.statusIndicator.textContent = "● Offline";
      this.stopLiveFeed();
      this.showOverlay(null);
    }
  }

  handleAttendanceUpdate(data) {
    playSuccessSound();
    this.showOverlay(`✅ ${data.student_name} Marked!`, "success", 3000);
    this.showNotification(
      `${data.student_name} marked present! (+${data.points} points)`,
      "success"
    );
    this.addRecentEvent({
      student_name: data.student_name,
      time_in: data.timestamp,
    });
    this.updateStatsInstantly();
  }

  updateStatsInstantly() {
    // (This function is from your original file, no changes needed)
    const currentPresent = parseInt(this.presentCountEl.textContent) || 0;
    const newPresent = currentPresent + 1;
    this.animateStatChange(this.presentCountEl, currentPresent, newPresent);

    const currentAbsent = parseInt(this.absentCountEl.textContent) || 0;
    const newAbsent = Math.max(0, currentAbsent - 1);
    this.animateStatChange(this.absentCountEl, currentAbsent, newAbsent);

    const totalStudents = parseInt(this.totalStudentsEl.textContent) || 0;
    if (totalStudents > 0) {
      const currentRate = parseInt(this.attendanceRateEl.textContent) || 0;
      const newRate = Math.round((newPresent / totalStudents) * 100);
      this.animateStatChange(this.attendanceRateEl, currentRate, newRate, '%');
    }

    setTimeout(() => {
      this.loadStats();
    }, 200);
  }

  animateStatChange(element, oldValue, newValue, suffix = '') {
    // (This function is from your original file, no changes needed)
    element.style.transform = 'scale(1.2)';
    element.style.color = 'var(--color-accent)';
    element.style.transition = 'all 0.3s ease';

    const duration = 300;
    const steps = 10;
    const stepValue = (newValue - oldValue) / steps;
    const stepDuration = duration / steps;
    
    let currentStep = 0;
    const interval = setInterval(() => {
      currentStep++;
      const displayValue = Math.round(oldValue + (stepValue * currentStep));
      element.textContent = displayValue + suffix;
      
      if (currentStep >= steps) {
        clearInterval(interval);
        element.textContent = newValue + suffix;
        
        setTimeout(() => {
          element.style.transform = 'scale(1)';
          element.style.color = 'var(--color-text-primary)';
        }, 200);
      }
    }, stepDuration);
  }


  // === Helper & UI Methods ===
  // (These functions are from your original file, no changes needed)
  showOverlay(message, type, duration = 2000) {
    clearTimeout(this.overlayTimeout);

    if (!message) {
      this.recognitionOverlay.classList.remove("show");
      return;
    }

    this.recognitionOverlayText.textContent = message;
    this.recognitionOverlay.className = "recognition-overlay";
    this.recognitionOverlay.classList.add(type);
    this.recognitionOverlay.classList.add("show");

    this.overlayTimeout = setTimeout(() => {
      this.recognitionOverlay.classList.remove("show");
    }, duration);
  }

  addRecentEvent(data) {
    const placeholder = this.recentEventsList.querySelector("li");
    if (placeholder && placeholder.textContent.includes("No events yet")) {
      this.recentEventsList.innerHTML = "";
    }
    const item = document.createElement("li");
    let time = data.time_in;
    if (typeof time === 'number') {
      time = this.formatTime(time);
    } else if (time && time.includes("T")) {
      time = this.formatTime(new Date(time).getTime() / 1000);
    }
    item.innerHTML = `<span class="time">[${time || "Just now"}]</span> ${data.student_name}`;
    this.recentEventsList.prepend(item);
    while (this.recentEventsList.children.length > 20) {
      this.recentEventsList.removeChild(this.recentEventsList.lastChild);
    }
  }

  formatTime(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
    });
  }

  showNotification(message, type = "success") {
    // (This function is from your original file, no changes needed)
    const notification = document.createElement("div");
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);
    const style = document.createElement("style");
    style.innerHTML = `
      .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 12px;
        color: #fff;
        font-weight: 600;
        z-index: 2000;
        animation: slideIn 0.3s ease-out, fadeOut 0.3s ease-in 2.7s;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
      }
      .notification.success {
        background: linear-gradient(135deg, #10b981, #059669);
      }
      .notification.error {
        background: linear-gradient(135deg, #ef4444, #dc2626);
      }
      .notification.warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
      }
      @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
      }
      @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; transform: translateX(100%); }
      }
    `;
    document.head.appendChild(style);
    setTimeout(() => {
      notification.remove();
      style.remove();
    }, 3000);
  }

  // === ALL NEW METHODS FROM CLAUDE'S SCRIPT (MERGED) ===
  // (These are now part of the AttendanceDashboard class)

  startQualityMonitoring() {
    this.qualityFeedback.style.display = 'block';
    this.qualityCheckInterval = setInterval(() => {
      this.checkFrameQuality();
    }, 500);
  }

  stopQualityMonitoring() {
    if (this.qualityCheckInterval) {
      clearInterval(this.qualityCheckInterval);
      this.qualityCheckInterval = null;
    }
    if (this.qualityFeedback) {
        this.qualityFeedback.style.display = 'none';
    }
  }

  async checkFrameQuality() {
    // Use this.videoPreview (from original class)
    if (!this.videoPreview || !this.videoPreview.srcObject) {
        return;
    }
    
    try {
      const canvas = document.createElement('canvas');
      canvas.width = this.videoPreview.videoWidth;
      canvas.height = this.videoPreview.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(this.videoPreview, 0, 0);
      
      const frameData = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
      
      const response = await fetch('/api/assess-quality', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}` // 'token' is globally available
        },
        body: JSON.stringify({ frame: frameData })
      });
      
      const result = await response.json();
      
      if (result.success) {
        this.updateQualityUI(result);
      }
    } catch (error) {
      console.error('Quality check error:', error);
    }
  }

  updateQualityUI(result) {
    if (!result.has_face) {
      this.qualityFeedback.className = 'quality-feedback-overlay poor';
      this.qualityMessage.textContent = result.feedback.message;
      this.qualityScoreFill.style.width = '0%';
      this.qualityItems.innerHTML = '';
      return;
    }
    
    const feedback = result.feedback;
    this.qualityFeedback.className = `quality-feedback-overlay ${feedback.status}`;
    this.qualityMessage.textContent = feedback.message;
    const qualityScore = result.quality_score || 0;
    this.qualityScoreFill.style.width = `${qualityScore * 100}%`;
    
    if (feedback.items && feedback.items.length > 0) {
      this.qualityItems.innerHTML = feedback.items
        .map(item => {
          const icon = item.score >= 0.7 ? '✓' : '⚠️';
          return `<div class="quality-item">
                      <i>${icon}</i>
                      <span>${item.feedback}</span>
                  </div>`;
        })
        .join('');
    } else {
      this.qualityItems.innerHTML = '';
    }
  }

  async startMultiShotCapture() {
    if (this.isCapturing) {
      return;
    }
    
    // Check if form is valid before starting capture
    const studentId = document.getElementById("student-id").value.trim();
    if (!studentId) {
        this.showNotification("Please fill out the Student ID first.", "warning");
        return;
    }

    this.isCapturing = true;
    this.capturedFrames = [];
    
    // Hide capture button, show progress
    this.captureBtn.classList.add('hidden');
    this.recaptureBtn.classList.add('hidden');
    this.captureProgress.style.display = 'block';
    this.updateFrameCount();
    
    await this.showCountdown();
    
    this.frameIndicator.style.display = 'block';
    
    this.captureInterval = setInterval(() => {
      this.captureFrame();
    }, 400); 
  }

  async showCountdown() {
    this.countdownOverlay.style.display = 'block';
    for (let i = 3; i > 0; i--) {
      this.countdownOverlay.textContent = i;
      await this.sleep(1000);
    }
    this.countdownOverlay.textContent = 'GO!';
    await this.sleep(500);
    this.countdownOverlay.style.display = 'none';
  }

  captureFrame() {
    if (this.capturedFrames.length >= this.targetFrameCount) {
      this.finishCapture();
      return;
    }
    
    try {
      // Use this.videoPreview (from original class)
      const canvas = document.createElement('canvas');
      canvas.width = this.videoPreview.videoWidth;
      canvas.height = this.videoPreview.videoHeight;
      const ctx = canvas.getContext('2d');
      
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(this.videoPreview, 0, 0);
      
      const frameData = canvas.toDataURL('image/jpeg', 0.95).split(',')[1];
      this.capturedFrames.push(frameData);
      
      this.updateFrameCount();
      
      this.frameIndicator.style.borderColor = '#00E0F0';
      setTimeout(() => {
        this.frameIndicator.style.borderColor = 'rgba(0, 224, 240, 0.5)';
      }, 100);
      
    } catch (error) {
      console.error('Frame capture error:', error);
    }
  }

  updateFrameCount() {
    this.frameCount.textContent = this.capturedFrames.length;
  }

  finishCapture() {
    clearInterval(this.captureInterval);
    this.captureInterval = null;
    this.isCapturing = false;
    
    this.frameIndicator.style.display = 'none';
    this.captureProgress.style.display = 'none';
    
    // Stop the camera and quality checks
    this.stopVideoStream();
    this.stopQualityMonitoring();

    // Show the "recapture" and "submit" buttons
    this.recaptureBtn.classList.remove('hidden');
    this.enrollSubmitBtn.disabled = false;
    
    // Show the last captured frame on the canvas
    const lastFrameData = this.capturedFrames[this.capturedFrames.length - 1];
    const img = new Image();
    img.onload = () => {
        const ctx = this.photoCanvas.getContext('2d');
        this.photoCanvas.width = img.width;
        this.photoCanvas.height = img.height;
        ctx.clearRect(0, 0, img.width, img.height);
        
        // We need to re-flip the canvas image to display it correctly
        // (since it was captured from a mirrored video)
        // But the original `capturePhoto` did NOT re-flip, it drew the mirrored
        // image. To be consistent, we will also draw the mirrored image.
        // If you want it un-mirrored, uncomment the next two lines:
        // ctx.translate(img.width, 0);
        // ctx.scale(-1, 1);
        
        ctx.drawImage(img, 0, 0);
    }
    // We need to re-create the base64 string for the Image element
    img.src = "data:image/jpeg;base64," + this.capturedFrames[this.capturedFrames.length - 1];


    this.videoPreview.style.display = "none";
    this.photoCanvas.style.display = "block";

    // Automatically trigger enrollment submission
    this.submitEnrollment();
  }

  async submitEnrollment() {
    // This function replaces the old handleEnrollSubmit
    const enrollBtn = this.enrollSubmitBtn;
    enrollBtn.disabled = true;
    enrollBtn.innerHTML = '<span class="spinner"></span> Processing...';
    
    try {
      // Validate form
      const name = document.getElementById("student-name").value.trim();
      const studentId = document.getElementById("student-id").value.trim();
      const studentClass = document.getElementById("class").value.trim();
      const section = document.getElementById("section").value.trim();
      const parentPhone = document.getElementById("parent-phone").value.trim();

      if (!name || !studentId || !studentClass || !section || !parentPhone) {
        throw new Error("All fields are mandatory.");
      }
      if (!/^[A-Za-z\s]+$/.test(name)) {
        throw new Error("Name must contain only alphabets.");
      }
      if (parentPhone.length < 10) {
        throw new Error("Phone number must be at least 10 digits.");
      }
      
      const studentData = {
        name: name,
        student_id: studentId,
        class: studentClass,
        section: section,
        parent_phone: parentPhone,
      };

      // 1. Create Student
      console.log('Creating student...');
      const createResponse = await fetch("/api/students", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(studentData),
      });
      const createResult = await createResponse.json();
      if (!createResponse.ok) {
        // If student already exists, this might fail.
        // We should check for "Student ID already exists"
        if (createResult.message.includes("already exists")) {
            console.warn("Student already exists, proceeding to enroll face.");
        } else {
            throw new Error(createResult.message);
        }
      }
      
      // 2. Register Student Login (if not already registered)
      // This logic is from the original file, it's important
      console.log('Creating login...');
      const registerResponse = await fetch("/api/auth/register", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          username: studentData.student_id,
          email: studentData.student_id + '@school.com',
          password: 'student123',
          role: 'student',
          student_id: createResult.id || studentData.student_id // Use ID from creation or fallback
        }),
      });
      if (!registerResponse.ok) {
          const registerResult = await registerResponse.json();
          // It's ok if user already exists
          if (!registerResult.message.includes("already exists")) {
              console.warn("Could not register login: " + registerResult.message);
          }
      }

      // 3. Enroll Face (Multi-shot)
      console.log('Enrolling face (multi-shot)...');
      const enrollResponse = await fetch('/api/enroll-multishot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}` 
        },
        body: JSON.stringify({
          student_id: studentId,
          frames: this.capturedFrames // Send all captured frames
        })
      });
      
      const result = await enrollResponse.json();
      
      if (result.success) {
        this.showNotification(`✅ ${result.message || 'Student enrolled successfully!'}`, "success");
        this.loadInitialData(); // Reload stats
        this.closeModal(); // Close modal on success
      } else {
        throw new Error(result.message);
      }
    } catch (error) {
      console.error('Enrollment error:', error);
      this.showNotification(`❌ ${error.message}`, "error");
      // Don't reset capture, allow user to retry submission
      enrollBtn.disabled = false;
      enrollBtn.innerHTML = '<i class="fa-solid fa-check"></i> Submit';
      // Show recapture button so user can try again
      this.recaptureBtn.classList.remove('hidden');
      this.captureBtn.classList.add('hidden');
    }
  }

  resetCapture() {
    this.capturedFrames = [];
    this.isCapturing = false;
    if (this.frameIndicator) this.frameIndicator.style.display = 'none';
    if (this.captureProgress) this.captureProgress.style.display = 'none';
    if (this.frameCount) this.updateFrameCount();
    
    // Reset buttons
    if (this.captureBtn) this.captureBtn.classList.remove('hidden');
    if (this.recaptureBtn) this.recaptureBtn.classList.add('hidden');
    if (this.enrollSubmitBtn) {
        this.enrollSubmitBtn.disabled = true;
        this.enrollSubmitBtn.innerHTML = '<i class="fa-solid fa-check"></i> Submit';
    }
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
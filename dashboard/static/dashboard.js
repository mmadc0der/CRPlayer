class CRPlayerDashboard {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.metrics = {
            dataRate: 0,
            frameRate: 0,
            totalData: 0,
            lastUpdate: Date.now()
        };
        
        this.initElements();
        this.bindEvents();
        this.startMetricsUpdate();
        this.addLog('Dashboard initialized', 'info');
    }

    initElements() {
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.connectBtn = document.getElementById('connectBtn');
        this.disconnectBtn = document.getElementById('disconnectBtn');
        this.screenshotBtn = document.getElementById('screenshotBtn');
        this.screenCanvas = document.getElementById('screenCanvas');
        this.screenPlaceholder = document.getElementById('screenPlaceholder');
        this.dataRateEl = document.getElementById('dataRate');
        this.frameRateEl = document.getElementById('frameRate');
        this.totalDataEl = document.getElementById('totalData');
        this.logContainer = document.getElementById('logContainer');
        
        this.ctx = this.screenCanvas.getContext('2d');
    }

    bindEvents() {
        this.connectBtn.addEventListener('click', () => this.connect());
        this.disconnectBtn.addEventListener('click', () => this.disconnect());
        this.screenshotBtn.addEventListener('click', () => this.takeScreenshot());
        
        // Touch/click events on canvas for device interaction
        this.screenCanvas.addEventListener('click', (e) => this.handleCanvasClick(e));
        this.screenCanvas.addEventListener('touchstart', (e) => this.handleCanvasTouch(e));
    }

    connect() {
        if (this.isConnected) return;
        
        this.addLog('Attempting to connect to pipeline...', 'info');
        
        // Try to connect to WebSocket server - use same host as web page
        const wsUrl = `ws://${window.location.hostname}:8765/ws`;
        this.addLog(`Connecting to ${wsUrl}`, 'info');
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            this.isConnected = true;
            this.updateConnectionStatus();
            this.addLog('Connected to pipeline', 'info');
        };
        
        this.ws.onmessage = (event) => {
            this.handleMessage(event.data);
        };
        
        this.ws.onclose = () => {
            this.isConnected = false;
            this.updateConnectionStatus();
            this.addLog('Connection closed', 'warn');
        };
        
        this.ws.onerror = (error) => {
            this.addLog('Connection error: WebSocket server not available', 'error');
            this.isConnected = false;
            this.updateConnectionStatus();
        };
    }

    disconnect() {
        if (!this.isConnected || !this.ws) return;
        
        this.ws.close();
        this.addLog('Disconnected from pipeline', 'info');
    }

    updateConnectionStatus() {
        if (this.isConnected) {
            this.statusDot.classList.add('connected');
            this.statusText.textContent = 'Connected';
            this.connectBtn.disabled = true;
            this.disconnectBtn.disabled = false;
            this.screenshotBtn.disabled = false;
        } else {
            this.statusDot.classList.remove('connected');
            this.statusText.textContent = 'Disconnected';
            this.connectBtn.disabled = false;
            this.disconnectBtn.disabled = true;
            this.screenshotBtn.disabled = true;
            
            // Show placeholder
            this.screenCanvas.style.display = 'none';
            this.screenPlaceholder.style.display = 'block';
        }
    }

    handleMessage(data) {
        try {
            const message = JSON.parse(data);
            
            switch (message.type) {
                case 'frame':
                    this.displayFrame(message.data);
                    break;
                case 'metrics':
                    this.updateMetrics(message.data);
                    break;
                case 'log':
                    this.addLog(message.message, message.level);
                    break;
                default:
                    console.log('Unknown message type:', message.type);
            }
        } catch (e) {
            // Handle binary data (frame data)
            this.displayBinaryFrame(data);
        }
    }

    displayFrame(frameData) {
        // Hide placeholder, show canvas
        this.screenPlaceholder.style.display = 'none';
        this.screenCanvas.style.display = 'block';
        
        // Create image from base64 data
        const img = new Image();
        img.onload = () => {
            // Resize canvas to match image
            this.screenCanvas.width = img.width;
            this.screenCanvas.height = img.height;
            
            // Draw frame
            this.ctx.drawImage(img, 0, 0);
            
            // Update frame rate
            this.metrics.frameRate++;
        };
        img.src = `data:image/jpeg;base64,${frameData}`;
    }

    displayBinaryFrame(binaryData) {
        // For future implementation of binary frame data
        console.log('Received binary frame data:', binaryData.byteLength, 'bytes');
    }

    updateMetrics(metricsData) {
        this.metrics = { ...this.metrics, ...metricsData };
        
        // Update UI
        this.dataRateEl.textContent = this.formatBytes(this.metrics.dataRate);
        this.frameRateEl.textContent = Math.round(this.metrics.frameRate);
        this.totalDataEl.textContent = (this.metrics.totalData / (1024 * 1024)).toFixed(1);
    }

    handleCanvasClick(e) {
        if (!this.isConnected) return;
        
        const rect = this.screenCanvas.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * this.screenCanvas.width;
        const y = ((e.clientY - rect.top) / rect.height) * this.screenCanvas.height;
        
        // Send touch event to server
        this.sendTouchEvent(x, y);
        this.addLog(`Touch at (${Math.round(x)}, ${Math.round(y)})`, 'info');
    }

    handleCanvasTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = this.screenCanvas.getBoundingClientRect();
        const x = ((touch.clientX - rect.left) / rect.width) * this.screenCanvas.width;
        const y = ((touch.clientY - rect.top) / rect.height) * this.screenCanvas.height;
        
        this.sendTouchEvent(x, y);
    }

    sendTouchEvent(x, y) {
        if (!this.ws || !this.isConnected) return;
        
        const message = {
            type: 'touch',
            x: Math.round(x),
            y: Math.round(y)
        };
        
        this.ws.send(JSON.stringify(message));
    }

    takeScreenshot() {
        if (!this.screenCanvas || this.screenCanvas.style.display === 'none') return;
        
        // Create download link
        const link = document.createElement('a');
        link.download = `crplayer_screenshot_${Date.now()}.png`;
        link.href = this.screenCanvas.toDataURL();
        link.click();
        
        this.addLog('Screenshot saved', 'info');
    }

    startMetricsUpdate() {
        setInterval(() => {
            // Reset frame rate counter every second
            this.metrics.frameRate = 0;
            
            // Simulate some metrics if not connected (for demo)
            if (!this.isConnected) {
                this.updateMetrics({
                    dataRate: Math.random() * 1000,
                    totalData: this.metrics.totalData + Math.random() * 100
                });
            }
        }, 1000);
    }

    addLog(message, level = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            <span class="log-level-${level}">${level.toUpperCase()}</span>
            ${message}
        `;
        
        this.logContainer.appendChild(logEntry);
        
        // Keep only last 50 entries
        while (this.logContainer.children.length > 50) {
            this.logContainer.removeChild(this.logContainer.firstChild);
        }
        
        // Scroll to bottom
        this.logContainer.scrollTop = this.logContainer.scrollHeight;
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new CRPlayerDashboard();
});

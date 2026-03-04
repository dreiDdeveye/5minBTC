// ============================================
// MECHA DASHBOARD - Core JS
// ============================================

// UTC Clock
function updateClock() {
    const now = new Date();
    const utc = now.toISOString().slice(11, 19) + ' UTC';
    const el = document.getElementById('hud-clock');
    if (el) el.textContent = utc;
}
setInterval(updateClock, 1000);
updateClock();

// ============================================
// Global status WebSocket (non-index pages only; index.html has its own)
// ============================================
if (document.getElementById('ws-badge')) {
    // Index page handles its own WS — skip global one
} else (function connectStatusWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/api/ws/price`);
    const dot = document.getElementById('status-dot');
    const statusText = dot ? dot.querySelector('.status-text') : null;
    const statusDot = dot ? dot.querySelector('.dot') : null;
    const footerPrice = document.getElementById('current-price');

    ws.onopen = () => {
        if (statusText) statusText.textContent = 'ONLINE';
        if (statusDot) statusDot.classList.add('online');
    };

    ws.onmessage = (event) => {
        const d = JSON.parse(event.data);
        if (footerPrice) {
            footerPrice.textContent = '$' + Number(d.price).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }
    };

    ws.onclose = () => {
        if (statusText) statusText.textContent = 'OFFLINE';
        if (statusDot) statusDot.classList.remove('online');
        setTimeout(connectStatusWS, 3000);
    };

    ws.onerror = () => ws.close();
})();

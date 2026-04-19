(function () {
  "use strict";

  var CONFIG = {
    keyHoldMs: 180,
    actionCooldownMs: 700
  };

  var keyTimers = {};
  var lastActionAt = 0;
  var socket = null;
  var enabled = false;
  var statusEl = null;
  var toggleBtn = null;
  var panelEl = null;
  var overlayEl = null;
  var overlayCtx = null;

  var KEY_MAP = {
    ArrowLeft: { key: "ArrowLeft", keyCode: 37, which: 37 },
    ArrowUp: { key: "ArrowUp", keyCode: 38, which: 38 },
    ArrowRight: { key: "ArrowRight", keyCode: 39, which: 39 },
    ArrowDown: { key: "ArrowDown", keyCode: 40, which: 40 },
    KeyA: { key: "a", keyCode: 65, which: 65 },
    KeyW: { key: "w", keyCode: 87, which: 87 },
    KeyD: { key: "d", keyCode: 68, which: 68 },
    KeyS: { key: "s", keyCode: 83, which: 83 },
    Space: { key: " ", keyCode: 32, which: 32 }
  };

  function dispatchKeyEvent(type, code) {
    var map = KEY_MAP[code];
    if (!map) return;

    var evt = new KeyboardEvent(type, {
      key: map.key,
      code: code,
      bubbles: true,
      cancelable: true
    });

    try {
      Object.defineProperty(evt, "keyCode", { get: function () { return map.keyCode; } });
      Object.defineProperty(evt, "which", { get: function () { return map.which; } });
      Object.defineProperty(evt, "charCode", { get: function () { return type === "keypress" ? map.which : 0; } });
    } catch (e) {}

    window.dispatchEvent(evt);
    document.dispatchEvent(evt);
  }

  function pressGameKey(code) {
    clearTimeout(keyTimers[code]);
    dispatchKeyEvent("keydown", code);
    keyTimers[code] = setTimeout(function () {
      dispatchKeyEvent("keyup", code);
      delete keyTimers[code];
    }, CONFIG.keyHoldMs);
  }

  function emitMove(direction) {
    if (direction === "left") {
      pressGameKey("ArrowLeft");
      pressGameKey("KeyA");
    } else if (direction === "right") {
      pressGameKey("ArrowRight");
      pressGameKey("KeyD");
    } else if (direction === "up") {
      pressGameKey("ArrowUp");
      pressGameKey("KeyW");
    } else if (direction === "down") {
      pressGameKey("ArrowDown");
      pressGameKey("KeyS");
    }
  }

  function emitAction() {
    var now = Date.now();
    if (now - lastActionAt < CONFIG.actionCooldownMs) return;
    lastActionAt = now;
    pressGameKey("Space");
  }

  function setStatus(text, active) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.classList.toggle("is-on", !!active);
    statusEl.classList.toggle("is-off", !active);
  }

  function handleRemoteMove(move) {
    if (!move || move === "idle") return;
    move.split("+").forEach(function (part) { emitMove(part); });
  }

  function drawOverlay(data) {
    if (!overlayCtx) return;

    overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);
    overlayCtx.strokeStyle = "rgba(101, 213, 255, 0.65)";
    overlayCtx.lineWidth = 1;
    overlayCtx.strokeRect(0.5, 0.5, overlayEl.width - 1, overlayEl.height - 1);

    overlayCtx.beginPath();
    overlayCtx.moveTo(overlayEl.width / 3, 0);
    overlayCtx.lineTo(overlayEl.width / 3, overlayEl.height);
    overlayCtx.moveTo((overlayEl.width * 2) / 3, 0);
    overlayCtx.lineTo((overlayEl.width * 2) / 3, overlayEl.height);
    overlayCtx.moveTo(0, overlayEl.height * 0.3);
    overlayCtx.lineTo(overlayEl.width, overlayEl.height * 0.3);
    overlayCtx.moveTo(0, overlayEl.height * 0.75);
    overlayCtx.lineTo(overlayEl.width, overlayEl.height * 0.75);
    overlayCtx.stroke();

    if (data && data.tracked && typeof data.x === "number" && typeof data.y === "number") {
      var px = data.x * overlayEl.width;
      var py = data.y * overlayEl.height;

      overlayCtx.strokeStyle = "rgba(255, 234, 87, 0.95)";
      overlayCtx.beginPath();
      overlayCtx.moveTo(overlayEl.width / 2, overlayEl.height / 2);
      overlayCtx.lineTo(px, py);
      overlayCtx.stroke();

      overlayCtx.fillStyle = "rgba(255, 84, 84, 1)";
      overlayCtx.beginPath();
      overlayCtx.arc(px, py, 6, 0, Math.PI * 2);
      overlayCtx.fill();
    }

    overlayCtx.fillStyle = "rgba(8, 12, 20, 0.8)";
    overlayCtx.fillRect(0, 0, overlayEl.width, 60); // Taller header to fit medical data
    overlayCtx.fillStyle = "#e7eefb";
    overlayCtx.font = "12px Segoe UI";
    overlayCtx.fillText("Move: " + ((data && data.move) || "idle"), 8, 16);
    
    if (data && typeof data.tremor_score !== 'undefined') {
        overlayCtx.fillStyle = "#ff6b6b";
        overlayCtx.fillText("Tremor: " + data.tremor_score.toFixed(2), 8, 34);
        overlayCtx.fillStyle = "#feca57";
        overlayCtx.fillText("Pinch Ext: " + data.pinch_amplitude.toFixed(2), 130, 34);
        overlayCtx.fillStyle = "#54a0ff";
        overlayCtx.fillText("Flicks: " + (data.flicks_count || 0) + " | Pinches: " + (data.total_pinches || 0), 8, 52);
    }
  }

  function stopSocketMode() {
    enabled = false;
    if (socket) {
      socket.close();
      socket = null;
    }
    if (toggleBtn) toggleBtn.textContent = "Enable Python MediaPipe";
    setStatus("Offline", false);
    if (panelEl) panelEl.classList.remove("connected");
    drawOverlay({ move: "idle", tracked: false });
  }

  function startSocketMode() {
    return new Promise(function (resolve, reject) {
      try {
        socket = new WebSocket("ws://127.0.0.1:8765");
      } catch (error) {
        reject(error);
        return;
      }

      socket.onopen = function () {
        enabled = true;
        if (toggleBtn) toggleBtn.textContent = "Disable Python MediaPipe";
        setStatus("Connected", true);
        if (panelEl) panelEl.classList.add("connected");
        resolve();
      };

      socket.onmessage = function (event) {
        try {
          if (!enabled) return;
          var data = JSON.parse(event.data);
          handleRemoteMove(data.move);
          if (data.action) emitAction();
          drawOverlay(data);
        } catch (error) {
          console.warn("Invalid MediaPipe payload", error);
        }
      };

      socket.onerror = function (error) {
        setStatus("Socket Error", false);
        reject(error);
      };

      socket.onclose = function () {
        enabled = false;
        if (toggleBtn) toggleBtn.textContent = "Enable Python MediaPipe";
        setStatus("Offline", false);
        if (panelEl) panelEl.classList.remove("connected");
      };
    });
  }

  function createUi() {
    panelEl = document.createElement("div");
    panelEl.className = "cv-panel";

    var titleEl = document.createElement("div");
    titleEl.className = "cv-title";
    titleEl.textContent = "Vision Control";

    var subtitleEl = document.createElement("div");
    subtitleEl.className = "cv-subtitle";
    subtitleEl.textContent = "Python MediaPipe Bridge";

    var rowEl = document.createElement("div");
    rowEl.className = "cv-row";

    toggleBtn = document.createElement("button");
    toggleBtn.className = "cv-btn";
    toggleBtn.textContent = "Enable Python MediaPipe";

    statusEl = document.createElement("span");
    statusEl.className = "cv-status is-off";
    statusEl.textContent = "Offline";

    rowEl.appendChild(toggleBtn);
    rowEl.appendChild(statusEl);

    var hintEl = document.createElement("div");
    hintEl.className = "cv-hint";
    hintEl.textContent = "Use hand direction to move, pinch for action.";

    panelEl.appendChild(titleEl);
    panelEl.appendChild(subtitleEl);
    panelEl.appendChild(rowEl);
    panelEl.appendChild(hintEl);
    document.body.appendChild(panelEl);

    overlayEl = document.createElement("canvas");
    overlayEl.width = 260;
    overlayEl.height = 195;
    overlayEl.className = "cv-overlay-canvas";
    document.body.appendChild(overlayEl);
    overlayCtx = overlayEl.getContext("2d");
    drawOverlay({ move: "idle", tracked: false });

    toggleBtn.addEventListener("click", async function () {
      if (enabled) {
        stopSocketMode();
        return;
      }
      try {
        setStatus("Connecting...", false);
        await startSocketMode();
      } catch (error) {
        setStatus("Unavailable", false);
        console.error("Python MediaPipe socket failed:", error);
      }
    });
  }

  window.addEventListener("beforeunload", function () { stopSocketMode(); });
  window.addEventListener("load", createUi);
})();

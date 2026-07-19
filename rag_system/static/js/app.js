(function () {
  'use strict';

  const SESSION_ID_KEY = 'rag_session_id';
  const DARK_MODE_KEY = 'rag_dark_mode';

  const chatContainer = document.getElementById('chat-container');
  const welcomeMessage = document.getElementById('welcome-message');
  const messageInput = document.getElementById('message-input');
  const sendBtn = document.getElementById('send-btn');
  const fileInput = document.getElementById('file-input');
  const uploadArea = document.getElementById('upload-area');
  const defaultFilesEl = document.getElementById('default-files');
  const userFilesEl = document.getElementById('user-files');
  const clearChatBtn = document.getElementById('clear-chat-btn');
  const darkModeToggle = document.getElementById('dark-mode-toggle');
  const mobileMenuBtn = document.getElementById('mobile-menu-btn');
  const sidebar = document.getElementById('sidebar');
  const addTextBtn = document.getElementById('add-text-btn');
  const addTextPanel = document.getElementById('add-text-panel');
  const addTextSubmit = document.getElementById('add-text-submit');
  const addTextCancel = document.getElementById('add-text-cancel');
  const textTitle = document.getElementById('text-title');
  const textContent = document.getElementById('text-content');
  const modalOverlay = document.getElementById('modal-overlay');
  const modalMessage = document.getElementById('modal-message');
  const modalConfirm = document.getElementById('modal-confirm');
  const modalCancel = document.getElementById('modal-cancel');
  const modalClose = document.getElementById('modal-close');
  const toastContainer = document.getElementById('toast-container');

  let sessionId = localStorage.getItem(SESSION_ID_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID ? crypto.randomUUID() : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
    localStorage.setItem(SESSION_ID_KEY, sessionId);
  }

  let isSubmitting = false;
  let deleteTarget = null;

  // Dark mode
  const savedDark = localStorage.getItem(DARK_MODE_KEY);
  if (savedDark === 'true') {
    document.documentElement.classList.add('dark');
    darkModeToggle.checked = true;
  }

  darkModeToggle.addEventListener('change', function () {
    if (this.checked) {
      document.documentElement.classList.add('dark');
      localStorage.setItem(DARK_MODE_KEY, 'true');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem(DARK_MODE_KEY, 'false');
    }
  });

  // Mobile menu
  mobileMenuBtn.addEventListener('click', function () {
    sidebar.classList.toggle('open');
  });

  document.addEventListener('click', function (e) {
    if (window.innerWidth <= 768 && sidebar.classList.contains('open')) {
      if (!sidebar.contains(e.target) && e.target !== mobileMenuBtn) {
        sidebar.classList.remove('open');
      }
    }
  });

  // Auto-resize textarea
  messageInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 150) + 'px';
    sendBtn.disabled = !this.value.trim();
  });

  messageInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  sendBtn.addEventListener('click', sendMessage);

  // File upload
  uploadArea.addEventListener('click', function () {
    fileInput.click();
  });

  uploadArea.addEventListener('dragover', function (e) {
    e.preventDefault();
    e.stopPropagation();
    this.classList.add('dragover');
  });

  uploadArea.addEventListener('dragleave', function (e) {
    e.preventDefault();
    e.stopPropagation();
    this.classList.remove('dragover');
  });

  uploadArea.addEventListener('drop', function (e) {
    e.preventDefault();
    e.stopPropagation();
    this.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      handleFiles(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener('change', function () {
    if (this.files.length) {
      handleFiles(this.files);
      this.value = '';
    }
  });

  // Language toggle
  function initLangToggle(containerId) {
    var container = document.getElementById(containerId);
    if (!container) return;
    var options = container.querySelectorAll('.lang-option');
    options.forEach(function (opt) {
      opt.addEventListener('click', function () {
        options.forEach(function (o) { o.classList.remove('active'); });
        this.classList.add('active');
      });
    });
  }
  initLangToggle('lang-toggle');
  initLangToggle('text-lang-toggle');

  function getSelectedLang(containerId) {
    var container = document.getElementById(containerId);
    if (!container) return 'en';
    var active = container.querySelector('.lang-option.active');
    return active ? active.getAttribute('data-lang') : 'en';
  }

  // Add text panel
  addTextBtn.addEventListener('click', function () {
    addTextPanel.style.display = 'flex';
    addTextBtn.style.display = 'none';
  });

  addTextCancel.addEventListener('click', function () {
    addTextPanel.style.display = 'none';
    addTextBtn.style.display = 'block';
    textTitle.value = '';
    textContent.value = '';
  });

  addTextSubmit.addEventListener('click', function () {
    const title = textTitle.value.trim();
    const content = textContent.value.trim();
    if (!title) { showToast('Please enter a title.', 'error'); return; }
    if (!content) { showToast('Please enter some content.', 'error'); return; }
    ingestText(title, content);
  });

  // Clear chat
  clearChatBtn.addEventListener('click', function () {
    fetch('/api/chat/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: 'session_id=' + encodeURIComponent(sessionId),
    }).then(function () {
      chatContainer.innerHTML = '';
      chatContainer.appendChild(welcomeMessage);
    }).catch(function () {
      chatContainer.innerHTML = '';
      chatContainer.appendChild(welcomeMessage);
    });
  });

  // Modal
  function showModal(message, onConfirm) {
    modalMessage.textContent = message;
    modalOverlay.style.display = 'flex';
    deleteTarget = onConfirm;
  }

  function hideModal() {
    modalOverlay.style.display = 'none';
    deleteTarget = null;
  }

  modalConfirm.addEventListener('click', function () {
    if (typeof deleteTarget === 'function') {
      deleteTarget();
    }
    hideModal();
  });

  modalCancel.addEventListener('click', hideModal);
  modalClose.addEventListener('click', hideModal);
  modalOverlay.addEventListener('click', function (e) {
    if (e.target === this) hideModal();
  });

  // Toast
  function showToast(message, type) {
    const toast = document.createElement('div');
    toast.className = 'toast ' + (type || 'info');
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(function () {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(20px)';
      toast.style.transition = 'opacity 0.3s, transform 0.3s';
      setTimeout(function () { toast.remove(); }, 300);
    }, 3000);
  }

  // API helpers
  function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isSubmitting) return;

    messageInput.value = '';
    messageInput.style.height = 'auto';
    sendBtn.disabled = true;

    addMessage(text, 'user');
    showLoading();

    isSubmitting = true;

    const formData = new FormData();
    formData.append('message', text);
    formData.append('session_id', sessionId);

    fetch('/api/chat', {
      method: 'POST',
      body: formData,
    })
    .then(function (res) {
      if (!res.ok) throw new Error('Server error');
      return res.json();
    })
    .then(function (data) {
      removeLoading();
      addAssistantMessage(data);
    })
    .catch(function (err) {
      removeLoading();
      addMessage('Error: ' + err.message, 'error');
    })
    .finally(function () {
      isSubmitting = false;
    });
  }

  function handleFiles(files) {
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const ext = '.' + file.name.split('.').pop().toLowerCase();
      const supported = ['.txt', '.md', '.pdf', '.docx', '.csv', '.json', '.parquet'];
      if (!supported.includes(ext)) {
        showToast('Unsupported file type: ' + ext, 'error');
        continue;
      }
      uploadFile(file);
    }
  }

  function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);
    formData.append('language', getSelectedLang('lang-toggle'));

    showUploadProgress(true, 'Uploading ' + file.name + '...');

    fetch('/api/ingest/file', {
      method: 'POST',
      body: formData,
    })
    .then(function (res) {
      if (!res.ok) throw new Error('Upload failed');
      return res.json();
    })
    .then(function (data) {
      showToast('Ingested "' + file.name + '" (' + data.chunks_added + ' chunks) using ' + data.model_type + ' model', 'success');
      refreshFiles();
    })
    .catch(function (err) {
      showToast('Failed to ingest: ' + err.message, 'error');
    })
    .finally(function () {
      showUploadProgress(false);
    });
  }

  function showUploadProgress(show, label) {
    const icon = document.getElementById('upload-icon');
    const text = document.getElementById('upload-text');
    const hint = document.getElementById('upload-hint');
    const progress = document.getElementById('upload-progress');
    const fill = document.getElementById('progress-fill');
    const labelEl = document.getElementById('progress-label');
    const area = document.getElementById('upload-area');

    if (show) {
      icon.style.display = 'none';
      text.style.display = 'none';
      hint.style.display = 'none';
      progress.style.display = 'block';
      labelEl.textContent = label || 'Processing...';
      area.classList.add('processing');
      fill.style.width = '0%';
      setTimeout(function () { fill.style.width = '60%'; }, 100);
    } else {
      fill.style.width = '100%';
      setTimeout(function () {
        progress.style.display = 'none';
        icon.style.display = '';
        text.style.display = '';
        hint.style.display = '';
        area.classList.remove('processing');
        fill.style.width = '0%';
      }, 400);
    }
  }

  function ingestText(title, content) {
    const formData = new FormData();
    formData.append('content', content);
    formData.append('title', title);
    formData.append('session_id', sessionId);
    formData.append('language', getSelectedLang('text-lang-toggle'));

    showToast('Ingesting "' + title + '"...', 'info');

    fetch('/api/ingest/text', {
      method: 'POST',
      body: formData,
    })
    .then(function (res) {
      if (!res.ok) throw new Error('Ingest failed');
      return res.json();
    })
    .then(function (data) {
      showToast('Ingested "' + title + '" (' + data.chunks_added + ' chunks) using ' + data.model_type + ' model', 'success');
      addTextPanel.style.display = 'none';
      addTextBtn.style.display = 'block';
      textTitle.value = '';
      textContent.value = '';
      refreshFiles();
    })
    .catch(function (err) {
      showToast('Failed to ingest: ' + err.message, 'error');
    });
  }

  function deleteFile(filename) {
    showModal('Are you sure you want to delete "' + filename + '"?', function () {
      fetch('/api/files/user?session_id=' + encodeURIComponent(sessionId) + '&filename=' + encodeURIComponent(filename), {
        method: 'DELETE',
      })
      .then(function (res) {
        if (!res.ok) throw new Error('Delete failed');
        return res.json();
      })
      .then(function () {
        showToast('Deleted "' + filename + '"', 'success');
        refreshFiles();
      })
      .catch(function (err) {
        showToast('Failed to delete: ' + err.message, 'error');
      });
    });
  }

  // Chat UI helpers
  function addMessage(content, role) {
    hideWelcome();

    const div = document.createElement('div');
    div.className = 'message ' + role;

    if (role === 'user') {
      div.textContent = content;
    } else {
      div.innerHTML = content;
    }

    chatContainer.appendChild(div);
    scrollToBottom();
    return div;
  }

  function addAssistantMessage(data) {
    const div = document.createElement('div');
    div.className = 'message assistant';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    if (typeof marked !== 'undefined' && marked.parse) {
      contentDiv.innerHTML = marked.parse(data.answer, { breaks: true });
    } else {
      contentDiv.textContent = data.answer;
    }
    div.appendChild(contentDiv);

    if (data.sources && data.sources.length > 0) {
      const sourcesDiv = document.createElement('div');
      sourcesDiv.className = 'message-sources';
      data.sources.forEach(function (src) {
        const badge = document.createElement('span');
        badge.className = 'source-badge';
        badge.textContent = src;
        sourcesDiv.appendChild(badge);
      });
      div.appendChild(sourcesDiv);
    }

    if (data.latency_summary) {
      const latencyDiv = document.createElement('div');
      latencyDiv.className = 'message-latency';
      latencyDiv.textContent = 'Retrieved ' + data.chunks_retrieved + ' chunks | ' + data.latency_summary;
      div.appendChild(latencyDiv);
    }

    chatContainer.appendChild(div);
    scrollToBottom();
  }

  let loadingEl = null;

  function showLoading() {
    hideWelcome();
    loadingEl = document.createElement('div');
    loadingEl.className = 'message assistant';
    const dots = document.createElement('div');
    dots.className = 'loading-dots';
    dots.innerHTML = '<span></span><span></span><span></span>';
    loadingEl.appendChild(dots);
    chatContainer.appendChild(loadingEl);
    scrollToBottom();
  }

  function removeLoading() {
    if (loadingEl) {
      loadingEl.remove();
      loadingEl = null;
    }
  }

  function hideWelcome() {
    if (welcomeMessage && welcomeMessage.parentNode) {
      welcomeMessage.remove();
    }
  }

  function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  // File listing
  function refreshFiles() {
    loadDefaultFiles();
    loadUserFiles();
  }

  function loadDefaultFiles() {
    fetch('/api/files/default')
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (!data.files || data.files.length === 0) {
          defaultFilesEl.innerHTML = '<p class="loading-text">No built-in files.</p>';
          return;
        }
        defaultFilesEl.innerHTML = '';
        data.files.forEach(function (f) {
          const item = document.createElement('div');
          item.className = 'file-item';
          item.innerHTML =
            '<span class="file-name">' + escapeHtml(f.name) + '</span>' +
            '<span class="file-meta">' + f.chunks + ' chunks</span>';
          defaultFilesEl.appendChild(item);
        });
      })
      .catch(function () {
        defaultFilesEl.innerHTML = '<p class="loading-text">Failed to load.</p>';
      });
  }

  function loadUserFiles() {
    fetch('/api/files/user?session_id=' + encodeURIComponent(sessionId))
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (!data.files || data.files.length === 0) {
          userFilesEl.innerHTML = '<p class="loading-text">No files uploaded yet.</p>';
          return;
        }
        userFilesEl.innerHTML = '';
        data.files.forEach(function (f) {
          const item = document.createElement('div');
          item.className = 'file-item';
          item.innerHTML =
            '<span class="file-name">' + escapeHtml(f.name) + '</span>' +
            '<span class="file-meta">' + f.chunks + ' ch, ' + f.model + '</span>' +
            '<button class="file-delete-btn" data-filename="' + escapeAttr(f.name) + '">\u2716</button>';
          item.querySelector('.file-delete-btn').addEventListener('click', function () {
            deleteFile(f.name);
          });
          userFilesEl.appendChild(item);
        });
      })
      .catch(function () {
        userFilesEl.innerHTML = '<p class="loading-text">Failed to load.</p>';
      });
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  function escapeAttr(str) {
    return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  refreshFiles();
})();

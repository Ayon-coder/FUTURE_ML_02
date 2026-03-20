/**
 * TicketIQ — Frontend Application
 * ================================
 * Handles API integration, dynamic result rendering,
 * metrics visualization, and interactive UI features.
 */

const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const API_BASE = isLocal ? 'http://localhost:5000/api' : '/api';

// ── DOM Elements ────────────────────────────────────────────
const classifyForm = document.getElementById('classifyForm');
const ticketSubject = document.getElementById('ticketSubject');
const ticketDescription = document.getElementById('ticketDescription');
const classifyBtn = document.getElementById('classifyBtn');
const emptyState = document.getElementById('emptyState');
const resultsContent = document.getElementById('resultsContent');
const apiStatusDot = document.querySelector('.status-dot');
const apiStatusText = document.querySelector('.status-text');

// ── Category & Priority Color Mapping ───────────────────────
const CATEGORY_COLORS = {
    'Billing inquiry': { class: 'billing', color: '#f59e0b' },
    'Technical issue': { class: 'technical', color: '#ef4444' },
    'Cancellation request': { class: 'cancellation', color: '#8b5cf6' },
    'Product inquiry': { class: 'product', color: '#06b6d4' },
    'Refund request': { class: 'refund', color: '#f97316' }
};

const PRIORITY_COLORS = {
    'Critical': { class: 'critical', color: '#ef4444' },
    'High': { class: 'high', color: '#f59e0b' },
    'Medium': { class: 'medium', color: '#3b82f6' },
    'Low': { class: 'low', color: '#10b981' }
};

// ── Initialize ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
    loadMetrics();
    setupExampleChips();
    setupNavigation();
});

// ── API Health Check ────────────────────────────────────────
async function checkApiHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
        const data = await res.json();
        
        if (data.models_loaded) {
            apiStatusDot.classList.add('connected');
            apiStatusDot.classList.remove('error');
            apiStatusText.textContent = 'API Connected';
        } else {
            apiStatusDot.classList.add('error');
            apiStatusText.textContent = 'Models not loaded';
        }
    } catch (e) {
        apiStatusDot.classList.add('error');
        apiStatusText.textContent = 'API Offline';
    }
}

// ── Form Submission ─────────────────────────────────────────
classifyForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const subject = ticketSubject.value.trim();
    const description = ticketDescription.value.trim();
    
    if (!subject && !description) return;
    
    // Show loading state
    classifyBtn.classList.add('loading');
    classifyBtn.disabled = true;
    
    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subject, description })
        });
        
        const data = await res.json();
        
        if (data.success) {
            displayResults(data.prediction);
        } else {
            showError(data.error || 'Classification failed');
        }
    } catch (e) {
        if (e instanceof SyntaxError) {
            showError('API Timeout: The prediction took longer than 10 seconds to execute. Please try again.');
        } else {
            showError(`Fetch Error: ${e.message}`);
        }
    } finally {
        classifyBtn.classList.remove('loading');
        classifyBtn.disabled = false;
    }
});

// ── Display Classification Results ──────────────────────────
function displayResults(prediction) {
    emptyState.style.display = 'none';
    resultsContent.style.display = 'flex';
    
    // Category
    const categoryBadge = document.getElementById('categoryBadge');
    const catInfo = CATEGORY_COLORS[prediction.category] || { class: '', color: '#6366f1' };
    categoryBadge.textContent = prediction.category;
    categoryBadge.className = `category-badge ${catInfo.class}`;
    
    // Category confidence
    const catConf = (prediction.category_confidence * 100).toFixed(1);
    document.getElementById('categoryConfidence').textContent = `${catConf}%`;
    document.getElementById('categoryFill').style.width = `${catConf}%`;
    
    // Category probabilities
    const catProbsEl = document.getElementById('categoryProbs');
    catProbsEl.innerHTML = '';
    if (prediction.category_probabilities) {
        for (const [cls, prob] of Object.entries(prediction.category_probabilities)) {
            const pct = (prob * 100).toFixed(1);
            const color = CATEGORY_COLORS[cls]?.color || '#6366f1';
            catProbsEl.innerHTML += `
                <div class="prob-row">
                    <span class="prob-label">${cls}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: ${pct}%; background: ${color}"></div>
                    </div>
                    <span class="prob-value">${pct}%</span>
                </div>
            `;
        }
    }
    
    // Priority
    const priorityBadge = document.getElementById('priorityBadge');
    const priInfo = PRIORITY_COLORS[prediction.priority] || { class: '', color: '#3b82f6' };
    priorityBadge.textContent = prediction.priority;
    priorityBadge.className = `priority-badge ${priInfo.class}`;
    
    // Priority confidence
    const priConf = (prediction.priority_confidence * 100).toFixed(1);
    document.getElementById('priorityConfidence').textContent = `${priConf}%`;
    document.getElementById('priorityFill').style.width = `${priConf}%`;
    
    // Priority probabilities
    const priProbsEl = document.getElementById('priorityProbs');
    priProbsEl.innerHTML = '';
    if (prediction.priority_probabilities) {
        for (const [cls, prob] of Object.entries(prediction.priority_probabilities)) {
            const pct = (prob * 100).toFixed(1);
            const color = PRIORITY_COLORS[cls]?.color || '#3b82f6';
            priProbsEl.innerHTML += `
                <div class="prob-row">
                    <span class="prob-label">${cls}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: ${pct}%; background: ${color}"></div>
                    </div>
                    <span class="prob-value">${pct}%</span>
                </div>
            `;
        }
    }
    
    // Scroll to results on mobile
    if (window.innerWidth <= 900) {
        document.getElementById('resultsCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// ── Error Display ───────────────────────────────────────────
function showError(message) {
    emptyState.style.display = 'none';
    resultsContent.style.display = 'flex';
    resultsContent.innerHTML = `
        <div class="result-block" style="text-align: center; color: #ef4444;">
            <div style="font-size: 2rem; margin-bottom: 12px;">⚠️</div>
            <p>${message}</p>
        </div>
    `;
}

// ── Load Metrics ────────────────────────────────────────────
async function loadMetrics() {
    const loading = document.getElementById('metricsLoading');
    const content = document.getElementById('metricsContent');
    
    try {
        const res = await fetch(`${API_BASE}/metrics`);
        const data = await res.json();
        
        if (data.success) {
            loading.style.display = 'none';
            content.style.display = 'block';
            renderMetrics(data.metrics);
        } else {
            loading.innerHTML = '<p style="color: var(--text-muted);">Metrics unavailable. Train models first.</p>';
        }
    } catch (e) {
        loading.innerHTML = '<p style="color: var(--text-muted);">Cannot load metrics. Start the API server first.</p>';
    }
}

// ── Render Metrics ──────────────────────────────────────────
function renderMetrics(metrics) {
    // Category accuracy ring
    const catAcc = metrics.category.accuracy;
    animateRing('catRingFill', catAcc);
    document.getElementById('catAccValue').textContent = `${(catAcc * 100).toFixed(1)}%`;
    
    // Priority accuracy ring
    const priAcc = metrics.priority.accuracy;
    animateRing('priRingFill', priAcc);
    document.getElementById('priAccValue').textContent = `${(priAcc * 100).toFixed(1)}%`;
    
    // Mini metrics
    document.getElementById('catMiniMetrics').innerHTML = `
        <div class="mini-metric">
            <span class="mini-metric-value">${(metrics.category.precision_weighted * 100).toFixed(1)}%</span>
            <span class="mini-metric-label">Precision</span>
        </div>
        <div class="mini-metric">
            <span class="mini-metric-value">${(metrics.category.recall_weighted * 100).toFixed(1)}%</span>
            <span class="mini-metric-label">Recall</span>
        </div>
        <div class="mini-metric">
            <span class="mini-metric-value">${(metrics.category.f1_weighted * 100).toFixed(1)}%</span>
            <span class="mini-metric-label">F1-Score</span>
        </div>
    `;
    
    document.getElementById('priMiniMetrics').innerHTML = `
        <div class="mini-metric">
            <span class="mini-metric-value">${(metrics.priority.precision_weighted * 100).toFixed(1)}%</span>
            <span class="mini-metric-label">Precision</span>
        </div>
        <div class="mini-metric">
            <span class="mini-metric-value">${(metrics.priority.recall_weighted * 100).toFixed(1)}%</span>
            <span class="mini-metric-label">Recall</span>
        </div>
        <div class="mini-metric">
            <span class="mini-metric-value">${(metrics.priority.f1_weighted * 100).toFixed(1)}%</span>
            <span class="mini-metric-label">F1-Score</span>
        </div>
    `;
    
    // Per-class tables
    renderPerClassTable('categoryTable', metrics.category.per_class);
    renderPerClassTable('priorityTable', metrics.priority.per_class);
    
    // Confusion matrices
    renderConfusionMatrix('categoryCM', metrics.category.confusion_matrix, metrics.category.class_names);
    renderConfusionMatrix('priorityCM', metrics.priority.confusion_matrix, metrics.priority.class_names);
    
    // Update stats
    document.getElementById('statTickets').textContent = metrics.category.total_samples.toLocaleString();
}

// ── Animate Accuracy Ring ───────────────────────────────────
function animateRing(elementId, value) {
    const circle = document.getElementById(elementId);
    const circumference = 2 * Math.PI * 52; // r=52
    const offset = circumference * (1 - value);
    
    // Small delay for animation effect
    setTimeout(() => {
        circle.style.strokeDashoffset = offset;
    }, 300);
}

// ── Render Per-Class Table ──────────────────────────────────
function renderPerClassTable(tableId, perClass) {
    const tbody = document.querySelector(`#${tableId} tbody`);
    tbody.innerHTML = '';
    
    for (const [cls, metrics] of Object.entries(perClass)) {
        const f1Class = metrics.f1_score >= 0.5 ? 'metric-highlight' : '';
        tbody.innerHTML += `
            <tr>
                <td>${cls}</td>
                <td>${(metrics.precision * 100).toFixed(1)}%</td>
                <td>${(metrics.recall * 100).toFixed(1)}%</td>
                <td class="${f1Class}">${(metrics.f1_score * 100).toFixed(1)}%</td>
                <td>${metrics.support}</td>
            </tr>
        `;
    }
}

// ── Render Confusion Matrix ─────────────────────────────────
function renderConfusionMatrix(containerId, matrix, classNames) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    if (!matrix || !classNames) return;
    
    // Abbreviate class names for display
    const shortNames = classNames.map(n => {
        if (n.length > 10) return n.split(' ')[0].substring(0, 8);
        return n;
    });
    
    // Find max value for color scaling
    const maxVal = Math.max(...matrix.flat());
    
    // Header row
    let headerHtml = '<div class="cm-header-row"><div class="cm-cell cm-label"></div>';
    shortNames.forEach(name => {
        headerHtml += `<div class="cm-cell cm-header">${name}</div>`;
    });
    headerHtml += '</div>';
    container.innerHTML += headerHtml;
    
    // Data rows
    matrix.forEach((row, i) => {
        let rowHtml = `<div class="cm-row"><div class="cm-cell cm-label">${shortNames[i]}</div>`;
        row.forEach((val, j) => {
            const intensity = maxVal > 0 ? val / maxVal : 0;
            const isDiagonal = i === j;
            const bgColor = isDiagonal
                ? `rgba(99, 102, 241, ${0.1 + intensity * 0.5})`
                : `rgba(255, 255, 255, ${intensity * 0.08})`;
            const extraClass = isDiagonal ? 'cm-diagonal' : '';
            
            rowHtml += `<div class="cm-cell cm-value ${extraClass}" style="background: ${bgColor}">${val}</div>`;
        });
        rowHtml += '</div>';
        container.innerHTML += rowHtml;
    });
}

// ── Example Chips ───────────────────────────────────────────
function setupExampleChips() {
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            ticketSubject.value = chip.dataset.subject;
            ticketDescription.value = chip.dataset.desc;
            
            // Visual feedback
            chip.style.transform = 'scale(0.95)';
            setTimeout(() => { chip.style.transform = ''; }, 150);
            
            // Auto-focus the classify button
            classifyBtn.focus();
        });
    });
}

// ── Navigation ──────────────────────────────────────────────
function setupNavigation() {
    // Active nav link tracking
    const sections = document.querySelectorAll('.section, .hero');
    const navLinks = document.querySelectorAll('.nav-link');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.id;
                navLinks.forEach(link => {
                    link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
                });
            }
        });
    }, { threshold: 0.3 });
    
    sections.forEach(section => {
        if (section.id) observer.observe(section);
    });
    
    // Navbar background on scroll
    window.addEventListener('scroll', () => {
        const navbar = document.getElementById('navbar');
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(10, 11, 20, 0.95)';
        } else {
            navbar.style.background = 'rgba(10, 11, 20, 0.85)';
        }
    });
}

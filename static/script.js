// Bookmark functionality
let bookmarks = JSON.parse(localStorage.getItem('stockBookmarks')) || [];

function toggleBookmarksList() {
    const bookmarksList = document.getElementById('bookmarks-list');
    bookmarksList.classList.toggle('show');
    if (bookmarksList.classList.contains('show')) {
        renderBookmarks();
    }
}

function renderBookmarks() {
    const bookmarksList = document.getElementById('bookmarks-list');
    bookmarksList.innerHTML = '';
    if (bookmarks.length === 0) {
        bookmarksList.innerHTML = '<div class="no-bookmarks">No bookmarks yet</div>';
        return;
    }
    bookmarks.forEach((bookmark, index) => {
        const bookmarkItem = document.createElement('div');
        bookmarkItem.className = 'bookmark-item';
        bookmarkItem.innerHTML = `
            <span onclick="loadBookmarkedStock('${bookmark.ticker}')">${bookmark.name} (${bookmark.ticker})</span>
            <button class="delete-bookmark" onclick="deleteBookmark(${index}, event)">
                <i class="fas fa-trash-alt"></i>
            </button>
        `;
        bookmarksList.appendChild(bookmarkItem);
    });
}

function loadBookmarkedStock(ticker) {
    document.getElementById('ticker-input').value = ticker;
    loadStockData();
    document.getElementById('bookmarks-list').classList.remove('show');
}

function deleteBookmark(index, event) {
    event.stopPropagation();
    bookmarks.splice(index, 1);
    localStorage.setItem('stockBookmarks', JSON.stringify(bookmarks));
    renderBookmarks();
}

function toggleBookmark() {
    const ticker = document.getElementById('ticker-input').value.trim();
    if (!ticker) return;
    const stockName = document.getElementById('stock-name').textContent;
    if (stockName === 'Stock Dashboard') {
        alert('Please search for a stock first');
        return;
    }
    const existingIndex = bookmarks.findIndex(b => b.ticker === ticker.toUpperCase());
    if (existingIndex >= 0) {
        bookmarks.splice(existingIndex, 1);
        document.getElementById('bookmark-btn').innerHTML = '<i class="far fa-bookmark"></i> Bookmark';
    } else {
        bookmarks.push({ ticker: ticker.toUpperCase(), name: stockName });
        document.getElementById('bookmark-btn').innerHTML = '<i class="fas fa-bookmark"></i> Bookmarked';
    }
    localStorage.setItem('stockBookmarks', JSON.stringify(bookmarks));
    renderBookmarks();
}

function normalizeBuySignal(value, recommendation, probability) {
    if (value === true || value === 'true' || value === 'True' || value === 1 || value === '1') return true;
    if (value === false || value === 'false' || value === 'False' || value === 0 || value === '0') return false;
    if (typeof recommendation === 'string') {
        return recommendation.toLowerCase().includes('buy');
    }
    if (typeof probability === 'number') {
        return probability >= 55;
    }
    return null;
}

function getBuySignalText(data) {
    if (data == null) return 'N/A';
    const signal = normalizeBuySignal(data.buy_signal, data.buy_recommendation, data.buy_probability);
    if (signal === true) return 'Yes';
    if (signal === false) return 'No';
    return 'N/A';
}

function resolveTradingViewSymbol(ticker) {
    const normalized = (ticker || '').trim().toUpperCase();
    if (!normalized) return 'NASDAQ:AAPL';
    if (normalized.includes(':')) return normalized;

    const exchangeMap = {
        AAPL: 'NASDAQ:AAPL',
        MSFT: 'NASDAQ:MSFT',
        AMZN: 'NASDAQ:AMZN',
        TSLA: 'NASDAQ:TSLA',
        NVDA: 'NASDAQ:NVDA',
        META: 'NASDAQ:META',
        GOOGL: 'NASDAQ:GOOGL',
        GOOG: 'NASDAQ:GOOG',
        AMD: 'NASDAQ:AMD',
        INTC: 'NASDAQ:INTC',
        NFLX: 'NASDAQ:NFLX',
        ORCL: 'NASDAQ:ORCL',
        PEP: 'NASDAQ:PEP',
        ADBE: 'NASDAQ:ADBE',
        CRM: 'NASDAQ:CRM',
        QCOM: 'NASDAQ:QCOM',
        AVGO: 'NASDAQ:AVGO',
        PYPL: 'NASDAQ:PYPL',
        IBM: 'NYSE:IBM',
        BA: 'NYSE:BA',
        DIS: 'NYSE:DIS',
        JPM: 'NYSE:JPM',
        V: 'NYSE:V',
        MA: 'NYSE:MA',
        PG: 'NYSE:PG',
        KO: 'NYSE:KO',
        WMT: 'NYSE:WMT',
        T: 'NYSE:T',
        XOM: 'NYSE:XOM',
        CVX: 'NYSE:CVX',
        LLY: 'NYSE:LLY',
        MRK: 'NYSE:MRK',
        ABBV: 'NYSE:ABBV',
        JNJ: 'NYSE:JNJ',
        PFE: 'NYSE:PFE',
        BAC: 'NYSE:BAC',
        GS: 'NYSE:GS',
        C: 'NYSE:C',
        SPY: 'AMEX:SPY',
        QQQ: 'NASDAQ:QQQ',
        'BRK.B': 'NYSE:BRK.B'
    };

    return exchangeMap[normalized] || normalized;
}

// ── Buy Signal UI Renderer ───────────────────────────────────────────────────
function renderBuySignal(data) {
    const isBuy = normalizeBuySignal(data.buy_signal, data.buy_recommendation, data.buy_probability) === true;
    const growthFactor = data.growth_factor ?? 0;
    const card = document.getElementById('buy-signal-card');

    // Badge
    const badge = document.getElementById('buy-signal-badge');
    badge.textContent = isBuy ? 'BUY' : 'HOLD';
    badge.className = 'signal-badge ' + (isBuy ? 'badge-buy' : 'badge-hold');

    // Card border colour
    card.classList.remove('card-buy', 'card-hold');
    card.classList.add(isBuy ? 'card-buy' : 'card-hold');

    // Text fields
    document.getElementById('buy-recommendation-detail').textContent = data.buy_recommendation ?? '—';
    document.getElementById('buy-confidence-detail').textContent     = data.confidence         ?? '—';
    document.getElementById('buy-growth-factor-detail').textContent  = growthFactor;
    document.getElementById('buy-risk-level-detail').textContent     = data.risk_level         ?? '—';

    // Growth factor bar (0–10 → 0–100%)
    const bar = document.getElementById('buy-pressure-bar');
    const pct = Math.min(100, growthFactor * 10);
    bar.style.width = pct + '%';
    bar.style.backgroundColor = growthFactor >= 7 ? '#1e88e5'
                               : growthFactor >= 5 ? '#42a5f5'
                               : '#90caf9';

    // Buy factor breakdown
    const factors = document.getElementById('buy-factors');
    const lines = [];
    if (data.revenue_growth != null && data.revenue_growth > 0)
        lines.push(`✓ Revenue grew ${data.revenue_growth.toFixed(1)}% QoQ`);
    if (data.asset_growth != null && data.asset_growth > 0)
        lines.push(`✓ Assets grew ${data.asset_growth.toFixed(1)}% QoQ`);
    if (data.earnings_surprise === 1.0)
        lines.push(`✓ Earnings beat last quarter`);
    if (data.volume_anomaly_ratio != null && data.volume_anomaly_ratio > 1.5)
        lines.push(`✓ Unusual volume (${data.volume_anomaly_ratio.toFixed(1)}× average) — heightened interest`);
    if (data.change_percentage > 1)
        lines.push(`✓ Price up ${data.change_percentage.toFixed(2)}% today`);
    if (data.revenue_growth != null && data.revenue_growth <= 0)
        lines.push(`✗ Revenue flat or declining (${data.revenue_growth.toFixed(1)}% QoQ)`);
    if (data.asset_growth != null && data.asset_growth <= 0)
        lines.push(`✗ Assets flat or declining (${data.asset_growth.toFixed(1)}% QoQ)`);
    if (data.earnings_surprise === -1.0)
        lines.push(`✗ Earnings missed last quarter`);
    if (lines.length === 0)
        lines.push('— Insufficient data to evaluate buy triggers');

    factors.innerHTML = lines.map(l => {
        const cls = l.startsWith('✓') ? 'factor-positive'
                  : l.startsWith('✗') ? 'factor-negative'
                  : '';
        return `<div class="signal-factor-item ${cls}">${l}</div>`;
    }).join('');
}

// ── Sell Signal UI Renderer ──────────────────────────────────────────────────
function renderSellSignal(data) {
    const isSell = data.sell_signal;
    const pressure = data.sell_pressure ?? 0;
    const card = document.getElementById('sell-signal-card');

    // Badge
    const badge = document.getElementById('sell-signal-badge');
    badge.textContent = isSell ? 'SELL' : 'HOLD';
    badge.className = 'signal-badge ' + (isSell ? 'badge-sell' : 'badge-hold');

    // Card border colour
    card.classList.remove('card-sell', 'card-hold');
    card.classList.add(isSell ? 'card-sell' : 'card-hold');

    // Text fields
    document.getElementById('sell-recommendation').textContent = data.sell_recommendation ?? '—';
    document.getElementById('sell-confidence').textContent     = data.sell_confidence     ?? '—';
    document.getElementById('sell-pressure').textContent       = pressure;
    document.getElementById('liability-ratio').textContent     =
        data.liability_ratio != null ? data.liability_ratio : '—';

    // Sell pressure bar (0–10 → 0–100%)
    const bar = document.getElementById('sell-pressure-bar');
    const pct = Math.min(100, pressure * 10);
    bar.style.width = pct + '%';
    bar.style.backgroundColor = pressure >= 7 ? '#e53935'
                               : pressure >= 5 ? '#fb8c00'
                               : '#43a047';

    // Sell factor breakdown
    const factors = document.getElementById('sell-factors');
    const lines = [];
    if (data.revenue_growth != null && data.revenue_growth < 0)
        lines.push(`⚠ Revenue declined ${Math.abs(data.revenue_growth).toFixed(1)}% QoQ`);
    if (data.asset_growth != null && data.asset_growth < 0)
        lines.push(`⚠ Assets shrank ${Math.abs(data.asset_growth).toFixed(1)}% QoQ`);
    if (data.earnings_surprise === -1.0)
        lines.push(`⚠ Earnings missed last quarter`);
    if (data.change_percentage < -2)
        lines.push(`⚠ Price dropped ${Math.abs(data.change_percentage).toFixed(2)}% today`);
    if (data.liability_ratio != null && data.liability_ratio > 0.6)
        lines.push(`⚠ High liability ratio: ${data.liability_ratio}`);
    if (data.volume_anomaly_ratio != null && data.volume_anomaly_ratio > 2.0 && data.sell_signal)
        lines.push(`⚠ Volume spike (${data.volume_anomaly_ratio.toFixed(1)}×) on a down day`);
    if (lines.length === 0)
        lines.push('✓ No significant sell triggers detected');

    factors.innerHTML = lines.map(l => {
        const cls = l.startsWith('✓') ? 'factor-positive' : 'factor-warning';
        return `<div class="signal-factor-item ${cls}">${l}</div>`;
    }).join('');
}

// ── Main data loader ─────────────────────────────────────────────────────────
function showInlineError(message) {
    const errorEl = document.getElementById('error-message');
    if (!errorEl) return;
    errorEl.textContent = message;
    errorEl.style.display = 'block';
}

function clearInlineError() {
    const errorEl = document.getElementById('error-message');
    if (errorEl) {
        errorEl.textContent = '';
        errorEl.style.display = 'none';
    }
}

function setFallbackValues() {
    document.getElementById('buy-signal').textContent = 'N/A';
    document.getElementById('confidence').textContent = '—';
    document.getElementById('risk-level').textContent = '—';
    document.getElementById('hero-action').textContent = 'Could not load stock data';
    document.getElementById('hero-advice').textContent = 'Check the ticker and try again.';
}

function formatCurrency(value) {
    if (value === null || value === undefined || value === '') return 'N/A';
    const numberValue = Number(value);
    if (!Number.isFinite(numberValue)) return 'N/A';
    return '$' + numberValue.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function loadStockData() {
    const tickerInput = document.getElementById('ticker-input').value.trim();
    const ticker = tickerInput || 'AAPL';
    clearInlineError();

    if (!tickerInput) {
        showInlineError('No ticker was entered. Defaulting to AAPL. Type a ticker and press Search to view it.');
    }

    fetch(`/api/data?ticker=${ticker}`)
        .then(res => {
            if (!res.ok) throw new Error(`Server error: ${res.status}`);
            return res.json();
        })
        .then(data => {
            if (data.error) {
                showInlineError(`API error: ${data.error}`);
                setFallbackValues();
                return;
            }

            // ── Buy signal (write this first so it's never left as —) ──
            const buySignalText = getBuySignalText(data);
            const isBuyOverview = buySignalText === 'Yes';

            document.getElementById('buy-signal').textContent = buySignalText;

            const revVal = data.revenue_growth != null ? data.revenue_growth : '—';
            const astVal = data.asset_growth   != null ? data.asset_growth   : '—';
            const gfVal  = data.growth_factor  != null ? data.growth_factor  : '—';

            document.getElementById('revenue-growth').textContent  = data.revenue_growth != null ? `${data.revenue_growth}%`   : '—';
            document.getElementById('asset-growth').textContent    = data.asset_growth   != null ? `${data.asset_growth}%`     : '—';
            document.getElementById('growth-factor').textContent   = data.growth_factor  != null ? `${data.growth_factor} / 10` : '—';

            // Financial snapshot detail rows (fin-rows section)
            const revDetail = document.getElementById('rev-growth-detail');
            const astDetail = document.getElementById('asset-growth-detail');
            const gfDetail  = document.getElementById('growth-factor-detail');
            if (revDetail) revDetail.textContent = data.revenue_growth != null ? `${data.revenue_growth}%`    : '—%';
            if (astDetail) astDetail.textContent = data.asset_growth   != null ? `${data.asset_growth}%`      : '—%';
            if (gfDetail)  gfDetail.textContent  = data.growth_factor  != null ? `${data.growth_factor} / 10` : '— / 10';

            // Bookmark state
            const isBookmarked = bookmarks.some(b => b.ticker === ticker.toUpperCase());
            document.getElementById('bookmark-btn').innerHTML = isBookmarked
                ? '<i class="fas fa-bookmark"></i> Bookmarked'
                : '<i class="far fa-bookmark"></i> Bookmark';

            // Quarterly table
            const tableBody = document.querySelector('#quarterly-table tbody');
            tableBody.innerHTML = '';
            if (data.quarterly_data && data.quarterly_data.length > 0) {
                data.quarterly_data.forEach(quarter => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${quarter.period || 'N/A'}</td>
                        <td>${formatCurrency(quarter.revenue)}</td>
                        <td>${formatCurrency(quarter.expenses)}</td>
                        <td>${formatCurrency(quarter.assets)}</td>
                        <td>${formatCurrency(quarter.liabilities)}</td>
                    `;
                    tableBody.appendChild(row);
                });
            } else {
                const row = document.createElement('tr');
                row.innerHTML = '<td colspan="5">No quarterly data available</td>';
                tableBody.appendChild(row);
            }

            // Key metrics
            document.getElementById('stock-name').textContent     = data.stock_name || ticker.toUpperCase();
            document.getElementById('current-price').textContent  = data.current_price != null ? data.current_price : 'N/A';
            document.getElementById('change').textContent         = data.change != null ? data.change : 'N/A';
            document.getElementById('change-percent').textContent = data.change_percentage != null ? data.change_percentage : 'N/A';
            document.getElementById('volume').textContent         = data.volume != null ? data.volume : 'N/A';
            document.getElementById('confidence').textContent     = data.confidence != null ? data.confidence : '—';
            document.getElementById('risk-level').textContent     = data.risk_level != null ? data.risk_level : '—';
            document.getElementById('recommendation').textContent = data.buy_recommendation ? `Recommendation: ${data.buy_recommendation}` : 'Recommendation: No data available.';
            document.getElementById('hero-action').textContent    = isBuyOverview ? 'BUY this stock' : 'HOLD and watch the market';
            document.getElementById('hero-advice').textContent    = data.buy_recommendation ? data.buy_recommendation : 'The tool is still learning. Try another ticker.';
            document.getElementById('simple-tip').textContent     = 'This tool is for learning and basic guidance only.';

            // Signal cards
            renderBuySignal(data);
            renderSellSignal(data);

            // ── UPGRADED HIGH-ACCURACY TRADINGVIEW ENGINE ──
            const chartSymbol = data.chart_symbol || resolveTradingViewSymbol(ticker);
            const chartContainer = document.getElementById('tradingview_chart');
            if (chartContainer) {
                chartContainer.innerHTML = '';
            }
            if (window.TradingView) {
                new TradingView.widget({
                    "autosize": false,
                    "width": "100%",
                    "height": 560,
                    "symbol": chartSymbol,
                    "interval": "D",
                    "timezone": "America/New_York", // Rooted to exchange time to align candle closes
                    "theme": "dark",
                    "style": "1",
                    "locale": "en",
                    "toolbar_bg": "#0b0f0c",
                    "enable_publishing": false,
                    "hide_top_toolbar": false,
                    "allow_symbol_change": true,
                    "container_id": "tradingview_chart",
                    "with_themename": true,
                    
                    // Technical overlays to replicate ML configurations 
                    "studies": [
                        "MASimple@tv-basicstudies",
                        "RSI@tv-basicstudies",
                        "Volume@tv-basicstudies"
                    ],
                    "disabled_features": [
                        "header_screenshot",
                        "use_localstorage_for_settings"
                    ],
                    "enabled_features": [
                        "study_templates",
                        "side_toolbar_in_popup"
                    ]
                });
            }
        })
        .catch(err => {
            showInlineError(`Failed to load stock data: ${err.message}`);
            setFallbackValues();
        });
}

// Close bookmarks dropdown on outside click
document.addEventListener('click', function(event) {
    const bookmarksList = document.getElementById('bookmarks-list');
    if (!event.target.closest('.bookmarks-dropdown') && bookmarksList.classList.contains('show')) {
        bookmarksList.classList.remove('show');
    }
});

document.addEventListener('DOMContentLoaded', function() {
    loadStockData();
    renderBookmarks();
});
const landingScreen = document.getElementById('landing-screen');
const forecastScreen = document.getElementById('forecast-screen');

const startBtn = document.getElementById('start-btn');
const landingStatus = document.getElementById('landing-status');

const backBtn = document.getElementById('back-btn');
const dateInput = document.getElementById('date-input');
const forecastBtn = document.getElementById('forecast-btn');
const forecastStatus = document.getElementById('forecast-status');
const forecastRange = document.getElementById('forecast-range');
const cardsContainer = document.getElementById('cards-container');

function showLanding() {
  landingScreen.classList.remove('hidden');
  forecastScreen.classList.add('hidden');
}

function showForecast() {
  landingScreen.classList.add('hidden');
  forecastScreen.classList.remove('hidden');
}

function setLandingStatus(message, type) {
  landingStatus.textContent = message || '';
  landingStatus.className = 'landing-status';
  if (type === 'error') landingStatus.classList.add('error');
  if (type === 'success') landingStatus.classList.add('success');
}

function setForecastStatus(message, type) {
  forecastStatus.textContent = message || '';
  forecastStatus.className = 'forecast-status';
  if (type === 'error') forecastStatus.classList.add('error');
  if (type === 'success') forecastStatus.classList.add('success');
}

function renderForecast(resp) {
  // Range dữ liệu — 1 dòng ngắn gọn
  forecastRange.textContent = ``;

  if (!resp.items || resp.items.length === 0) {
    cardsContainer.innerHTML = '<p class="muted">Không có dự báo để hiển thị.</p>';
    setForecastStatus('Không có dữ liệu dự báo.', 'error');
    return;
  }

  const cardsHtml = resp.items
    .map((item) => {
      const gtLines =
        item.gt_temp !== null && item.abs_error !== null
          ? `
        <div class="card-row">
          <span>Thực tế</span><span>${item.gt_temp.toFixed(1)}°C</span>
        </div>
        <div class="card-row">
          <span>Sai số</span><span>${item.abs_error.toFixed(1)}°C</span>
        </div>
      `
          : '';

      return `
        <div class="card">
          <div class="card-top">
            <span>t+${item.horizon_days}</span>
            <span>${item.target_date}</span>
          </div>
          <div class="card-temp">${item.pred_temp.toFixed(1)}°C</div>
          ${gtLines}
        </div>
      `;
    })
    .join('');

  cardsContainer.innerHTML = `<div class="cards-grid">${cardsHtml}</div>`;
  setForecastStatus(``, 'success');
}

// load default date cho input (dùng ở page 2)
async function loadDefaultDateIntoInput() {
  try {
    const res = await fetch('/default-date');
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Không lấy được ngày gốc mặc định.');
    }
    const data = await res.json();
    dateInput.value = data.current_date;
  } catch (err) {
    setForecastStatus(err.message, 'error');
  }
}

// gọi /predict với ngày trong ô dateInput
async function doPredict(useCurrentInput = true) {
  setLandingStatus('Đang tải dự báo...', null);
  setForecastStatus('Đang dự báo...', null);
  startBtn.disabled = true;
  forecastBtn.disabled = true;

  try {
    let cur = dateInput.value;

    if (!useCurrentInput || !cur) {
      const resDefault = await fetch('/default-date');
      if (!resDefault.ok) {
        const errData = await resDefault.json().catch(() => ({}));
        throw new Error(errData.detail || 'Không lấy được ngày gốc mặc định.');
      }
      const defaultData = await resDefault.json();
      cur = defaultData.current_date;
      dateInput.value = cur;
    }

    const resPredict = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current_date: cur })
    });
    const data = await resPredict.json();
    if (!resPredict.ok) {
      throw new Error(data.detail || 'Lỗi khi dự báo dữ liệu.');
    }

    renderForecast(data);
    setLandingStatus('', null); // không spam text ở page 1 nữa
  } catch (err) {
    setLandingStatus(err.message, 'error');
    setForecastStatus(err.message, 'error');
  } finally {
    startBtn.disabled = false;
    forecastBtn.disabled = false;
  }
}

// === Event wiring ===

window.addEventListener('load', () => {
  showLanding();
});

// Nút Page 1: lấy default-date, forecast, rồi sang page 2
startBtn.addEventListener('click', async () => {
  await loadDefaultDateIntoInput();
  await doPredict(false);
  showForecast();
});

// Nút "Trang đầu" ở Page 2
backBtn.addEventListener('click', () => {
  showLanding();
});

// Nút "Dự báo ngày này" ở Page 2
forecastBtn.addEventListener('click', async () => {
  if (!dateInput.value) {
    await loadDefaultDateIntoInput();
  }
  await doPredict(true);
});

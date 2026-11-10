const svg = document.querySelector("#chart");
const controls = [...document.querySelectorAll("[data-scenario]")];
const status = document.querySelector("#load-status");
const replay = document.querySelector("#replay");
const number = new Intl.NumberFormat("en-US");
const time = new Intl.DateTimeFormat("en-CA", {
  dateStyle: "medium", timeStyle: "long", timeZone: "UTC",
});
const notes = {
  baseline: "The batch uses the same income distribution as the training data.",
  income_shift: "The same applicants have higher incomes. Their debt and credit scores stay the same.",
};

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function svgElement(tag, attributes, label) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (label !== undefined) element.textContent = label;
  return element;
}

function drawChart(reference, batch, maxIncome, scenarioName) {
  const width = svg.clientWidth;
  const height = svg.clientHeight;
  const left = 42;
  const right = width - 22;
  const top = 10;
  const bottom = height - 26;
  const x = (income) => left + income / maxIncome * (right - left);
  const y = (share) => bottom - share * (bottom - top);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren(
    svgElement("title", { id: "chart-title" }, `${scenarioName} income distribution`),
    svgElement("desc", { id: "chart-description" },
      "The dashed line is the training reference. The solid line is the prediction batch."),
  );

  for (const share of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(
      svgElement("line", {
        x1: left, y1: y(share), x2: right, y2: y(share),
        class: share === 0 ? "axis" : "gridline",
      }),
      svgElement("text", { x: left - 8, y: y(share) + 4, "text-anchor": "end" },
        `${share * 100}%`),
    );
  }

  const ticks = width < 420 ? 2 : 4;
  for (let tick = 0; tick <= ticks; tick += 1) {
    const income = maxIncome * tick / ticks;
    svg.append(svgElement("text", { x: x(income), y: height - 6, "text-anchor": "middle" },
      income === 0 ? "$0" : `$${income / 1000}k`));
  }

  for (const [incomes, line] of [[reference, "reference-line"], [batch, "batch-line"]]) {
    let path = `M ${left} ${bottom}`;
    incomes.forEach((income, i) => {
      path += ` H ${x(income).toFixed(2)} V ${y((i + 1) / incomes.length).toFixed(2)}`;
    });
    svg.append(svgElement("path", { d: `${path} H ${right}`, class: `curve ${line}` }));
  }
}

function sortedIncome(values) {
  if (!Array.isArray(values) || !values.length || !values.every((v) => Number.isFinite(v) && v > 0)) {
    throw new Error("Invalid income sample");
  }
  return [...values].sort((a, b) => a - b);
}

async function loadReplay() {
  const response = await fetch("./replay.json");
  if (!response.ok) throw new Error(`Replay HTTP ${response.status}`);
  const data = await response.json();
  if (data.schema_version !== 1 || data.synthetic !== true) throw new Error("Invalid replay");

  const reference = sortedIncome(data.reference_income);
  const batches = Object.fromEntries(controls.map(({ dataset }) => {
    const result = data.scenarios[dataset.scenario];
    const income = sortedIncome(result.income);
    if (income.length !== result.sample_count || !Number.isFinite(result.ks_p_value)) {
      throw new Error("Invalid scenario result");
    }
    return [dataset.scenario, income];
  }));
  const maxIncome = Math.ceil(Math.max(...reference, ...Object.values(batches).flat()) / 40_000) * 40_000;
  const completedAt = document.querySelector("#completed-at");
  completedAt.textContent = time.format(new Date(data.completed_at));
  completedAt.dateTime = data.completed_at;
  setText("model-version", data.model.version);
  setText("reference-count", number.format(reference.length));
  setText("threshold", data.drift_threshold);

  let active = controls[0];
  function renderChart() {
    drawChart(reference, batches[active.dataset.scenario], maxIncome, active.textContent);
  }
  function select(button) {
    active = button;
    for (const control of controls) control.setAttribute("aria-pressed", String(control === active));
    const scenario = active.dataset.scenario;
    const result = data.scenarios[scenario];
    setText("decision", result.drift_detected ? "Drift detected" : "No drift detected");
    setText("p-value", result.ks_p_value < 0.001
      ? result.ks_p_value.toExponential(2) : result.ks_p_value.toFixed(4));
    setText("scenario-note", notes[scenario]);
    setText("batch-count", number.format(result.sample_count));
    renderChart();
  }

  status.hidden = true;
  replay.hidden = false;
  select(active);
  for (const control of controls) control.addEventListener("click", () => select(control));
  new ResizeObserver(renderChart).observe(svg);
}

loadReplay().catch((error) => {
  console.error(error);
  replay.hidden = true;
  status.hidden = false;
  status.textContent = "The verified replay couldn’t load. Reload the page to try again.";
});

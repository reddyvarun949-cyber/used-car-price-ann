import { useEffect, useState } from "react";
import { getOptions, predictPrice } from "./api.js";

// Starting values for the form (a typical 5-year-old petrol hatchback).
const START_FORM = {
  brand: "",
  model_segment: "Hatchback",
  year: "2020",
  km_driven: "45000",
  fuel: "Petrol",
  transmission: "Manual",
  owners: "1",
  city: "",
  seller_type: "Individual",
  mileage_kmpl: "21",
  engine_cc: "1200",
};

// Dropdown fields: form name -> label shown to the user.
const SELECT_FIELDS = [
  { name: "brand", label: "Brand" },
  { name: "model_segment", label: "Body type" },
  { name: "fuel", label: "Fuel" },
  { name: "transmission", label: "Transmission" },
  { name: "city", label: "City" },
  { name: "seller_type", label: "Seller" },
];

// Number fields. "optional" fields may be left empty; mileage and engine are disabled for electric cars.
const NUMBER_FIELDS = [
  { name: "year", label: "Year of manufacture", step: "1" },
  { name: "km_driven", label: "Kilometres driven", step: "100" },
  { name: "owners", label: "Owners so far (4 = four or more)", step: "1" },
  { name: "mileage_kmpl", label: "Mileage (km per litre)", step: "0.1", notForElectric: true },
  { name: "engine_cc", label: "Engine size (cc, optional)", step: "50", notForElectric: true, optional: true },
];

function App() {
  const [options, setOptions] = useState(null);   // allowed values from the API
  const [form, setForm] = useState(START_FORM);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});
  const [loading, setLoading] = useState(false);

  const isElectric = form.fuel === "Electric";

  // Load the dropdown values once when the page opens.
  useEffect(() => {
    getOptions()
      .then((data) => {
        setOptions(data);
        setForm((f) => ({ ...f, brand: data.categories.brand[0], city: data.categories.city[0] }));
      })
      .catch((err) => setError(err.message));
  }, []);

  function handleChange(event) {
    const { name, value } = event.target;
    const next = { ...form, [name]: value };
    // Electric cars have no fuel mileage or engine size: clear them so the API accepts the request.
    if (name === "fuel" && value === "Electric") {
      next.mileage_kmpl = "";
      next.engine_cc = "";
      next.transmission = "Automatic";
    }
    setForm(next);
    setFieldErrors({ ...fieldErrors, [name]: undefined }); // clear the message for this field
  }

  // Simple checks in the browser; the API checks everything again.
  function checkForm() {
    const problems = {};
    for (const field of NUMBER_FIELDS) {
      if (field.notForElectric && isElectric) continue;
      const value = form[field.name];
      if (value === "" && !field.optional) problems[field.name] = "Required";
      else if (value !== "" && Number.isNaN(Number(value))) problems[field.name] = "Must be a number";
    }
    return problems;
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setResult(null);
    setError("");
    const problems = checkForm();
    if (Object.keys(problems).length > 0) {
      setFieldErrors(problems);
      return;
    }
    setLoading(true);
    try {
      const data = await predictPrice(form);
      setResult(data);
    } catch (err) {
      setError(err.message);
      setFieldErrors(err.fields || {});
    } finally {
      setLoading(false);
    }
  }

  if (!options && !error) {
    return <main className="page"><p className="muted">Loading form…</p></main>;
  }

  return (
    <main className="page">
      <header>
        <h1>Used Car Price Estimator</h1>
        <p className="muted">
          Describe your car to get an instant price estimate in lakh rupees from a neural network,
          with a gradient boosting model as a second opinion.
        </p>
      </header>

      {error && <div className="alert">{error}</div>}

      {options && (
        <form onSubmit={handleSubmit} className="card" noValidate>
          <div className="grid">
            {SELECT_FIELDS.map((field) => (
              <label key={field.name}>
                {field.label}
                <select name={field.name} value={form[field.name]} onChange={handleChange}>
                  {options.categories[field.name].map((value) => (
                    <option key={value} value={value}>{value}</option>
                  ))}
                </select>
                {fieldErrors[field.name] && <span className="hint error">{fieldErrors[field.name]}</span>}
              </label>
            ))}

            {NUMBER_FIELDS.map((field) => {
              const [min, max] = options.numeric_ranges[field.name];
              const disabled = field.notForElectric && isElectric;
              let hint = `Allowed: ${min} – ${max}`;
              if (disabled) hint = "Not used for electric cars";
              if (fieldErrors[field.name]) hint = fieldErrors[field.name];
              return (
                <label key={field.name}>
                  {field.label}
                  <input
                    type="number"
                    name={field.name}
                    value={form[field.name]}
                    min={min}
                    max={max}
                    step={field.step}
                    disabled={disabled}
                    onChange={handleChange}
                    className={fieldErrors[field.name] ? "invalid" : ""}
                  />
                  <span className="hint">{hint}</span>
                </label>
              );
            })}
          </div>

          <button type="submit" disabled={loading}>
            {loading ? "Estimating…" : "Estimate price"}
          </button>
        </form>
      )}

      {result && (
        <section className="card result">
          <div className="estimates">
            <div className="estimate main">
              <p className="label">Neural network (ANN)</p>
              <p className="price">₹ {result.predicted_price_lakhs.toFixed(2)} lakh</p>
              <p className="small">
                Likely range: ₹ {result.likely_range_lakhs[0].toFixed(2)} – {result.likely_range_lakhs[1].toFixed(2)} lakh
              </p>
              <p className="muted small">{result.model}</p>
            </div>
            <div className="estimate">
              <p className="label">Gradient boosting (second opinion)</p>
              <p className="price">₹ {result.comparison.predicted_price_lakhs.toFixed(2)} lakh</p>
              <p className="muted small">{result.comparison.model}</p>
            </div>
          </div>
          <p className="explain">
            {result.difference_pct <= 10
              ? `The two models agree within ${result.difference_pct}%, so the estimate is fairly reliable.`
              : `The two models differ by ${result.difference_pct}%. This car may be unusual — treat the estimate with care.`}
            {" "}
            On data like this, tree-based gradient boosting is usually as accurate as a neural network;
            showing both helps you spot unusual listings.
          </p>
          <p className="muted small">{result.note}</p>
        </section>
      )}
    </main>
  );
}

export default App;

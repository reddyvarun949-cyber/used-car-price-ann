"""Generate the synthetic used-car listings dataset used in this project.

Run from the project root:  python data/generate_data.py
The output is data/used_cars.csv. The seed is fixed, so everyone gets the same file.

How a price is made (all prices in lakh rupees, 1 lakh = Rs 1,00,000):
    new-car price  = segment base price x brand factor x engine-size factor x fuel factor
    used price     = new price x first-year drop x (yearly value kept) ** age x km factor
                     x owner, transmission, seller and city adjustments x random noise (about +-9%)
Some brands (Toyota, Maruti) keep their value better than others, and in Delhi old diesel cars sell for
much less (they cannot be registered after 10 years). These interactions are what a model has to discover.

A few rows are made messy on purpose so the notebook can practise cleaning:
missing values, duplicate listings, "Bangalore" written instead of "Bengaluru", and impossible km values
(typing mistakes such as 4,500,000 km).
"""
import os

import numpy as np
import pandas as pd

SEED = 42
N = 6000
DATA_YEAR = 2025          # the listings were "collected" in 2025, so car_age = 2025 - year

# brand -> (share of listings, price factor vs an average brand, value kept per year, segment probabilities h/s/SUV)
BRANDS = {
    "Maruti":   (0.26, 0.90, 0.910, (0.65, 0.20, 0.15)),
    "Hyundai":  (0.18, 1.00, 0.895, (0.45, 0.25, 0.30)),
    "Tata":     (0.13, 0.95, 0.885, (0.40, 0.15, 0.45)),
    "Mahindra": (0.10, 1.05, 0.890, (0.00, 0.05, 0.95)),
    "Honda":    (0.09, 1.10, 0.895, (0.25, 0.55, 0.20)),
    "Toyota":   (0.09, 1.30, 0.915, (0.15, 0.35, 0.50)),
    "Kia":      (0.08, 1.10, 0.890, (0.05, 0.15, 0.80)),
    "Renault":  (0.04, 0.80, 0.860, (0.70, 0.00, 0.30)),
    "Skoda":    (0.03, 1.20, 0.870, (0.00, 0.60, 0.40)),
}
SEGMENT_BASE = {"Hatchback": 6.0, "Sedan": 10.0, "SUV": 14.0}          # new price in lakhs
SEGMENT_CC = {"Hatchback": (1100, 120), "Sedan": (1400, 150), "SUV": (1750, 300)}
SEGMENT_KMPL = {"Hatchback": 20.0, "Sedan": 17.0, "SUV": 14.5}
CITIES = {"Delhi": 0.97, "Mumbai": 1.03, "Bengaluru": 1.05, "Chennai": 1.00, "Pune": 1.01,
          "Hyderabad": 1.02, "Kolkata": 0.95, "Jaipur": 0.96}


def make_dataset(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    names = list(BRANDS)
    brand = rng.choice(names, size=n, p=[BRANDS[b][0] for b in names])
    segment = np.array([rng.choice(list(SEGMENT_BASE), p=BRANDS[b][3]) for b in brand])

    # Fuel: electric cars only exist from 2019 onwards in this market
    year = rng.integers(2010, DATA_YEAR, size=n)                      # 2010 ... 2024
    fuel = rng.choice(["Petrol", "Diesel", "CNG", "Electric"], size=n, p=[0.56, 0.28, 0.11, 0.05])
    fuel = np.where((fuel == "Electric") & (year < 2019), "Petrol", fuel)
    fuel = np.where((fuel == "CNG") & (segment == "SUV"), "Petrol", fuel)   # CNG SUVs are rare
    age = DATA_YEAR - year

    # Kilometres: about 10,000 km a year, diesel cars are driven more
    km = age * rng.lognormal(np.log(10000), 0.35, n) * np.where(fuel == "Diesel", 1.35, 1.0)
    km = (km + rng.integers(500, 6000, n)).round(-2)

    # Automatic gearboxes are more common in newer and pricier cars; all electric cars are automatic
    p_auto = 0.12 + 0.02 * (year - 2010) + np.where(segment == "SUV", 0.10, 0.0)
    transmission = np.where(rng.random(n) < p_auto, "Automatic", "Manual")
    transmission = np.where(fuel == "Electric", "Automatic", transmission)

    owners = np.minimum(1 + rng.poisson(0.06 * age), 4)               # 1 = first owner ... 4 = four or more
    city = rng.choice(list(CITIES), size=n)
    seller = rng.choice(["Individual", "Dealer"], size=n, p=[0.62, 0.38])

    cc_mean = np.array([SEGMENT_CC[s][0] for s in segment])
    cc_sd = np.array([SEGMENT_CC[s][1] for s in segment])
    engine_cc = (rng.normal(cc_mean, cc_sd) / 50).round() * 50
    engine_cc = engine_cc.clip(800, 2800)
    engine_cc = np.where(fuel == "Diesel", np.maximum(engine_cc, 1200), engine_cc)

    kmpl = np.array([SEGMENT_KMPL[s] for s in segment])
    kmpl = kmpl * np.select([fuel == "Diesel", fuel == "CNG"], [1.22, 1.30], 1.0) * (1200 / engine_cc) ** 0.25
    kmpl = (kmpl + rng.normal(0, 1.0, n)).round(1)

    # ---- price ----
    brand_factor = np.array([BRANDS[b][1] for b in brand])
    keep_per_year = np.array([BRANDS[b][2] for b in brand])
    base = np.array([SEGMENT_BASE[s] for s in segment])
    new_price = (base * brand_factor * (engine_cc / cc_mean) ** 0.6
                 * pd.Series(fuel).map({"Petrol": 1.0, "Diesel": 1.12, "CNG": 1.05, "Electric": 1.45}).to_numpy())
    price = new_price * 0.88 * keep_per_year ** age
    expected_km = np.maximum(age, 1) * 10000
    price *= np.exp(-0.25 * (km - expected_km) / 100000)           # more km than usual -> cheaper
    price *= 1 - 0.06 * (owners - 1)
    price *= np.where(transmission == "Automatic", 1.07, 1.0)
    price *= np.where(seller == "Dealer", 1.05, 1.0)
    price *= np.array([CITIES[c] for c in city])
    price *= np.where((city == "Delhi") & (fuel == "Diesel") & (age >= 10), 0.70, 1.0)
    price *= rng.lognormal(0, 0.09, n)                              # market noise
    price = price.round(2)

    df = pd.DataFrame({
        "car_id": np.arange(1, n + 1),
        "brand": brand, "model_segment": segment, "year": year, "km_driven": km.astype(int),
        "fuel": fuel, "transmission": transmission, "owners": owners, "city": city, "seller_type": seller,
        "mileage_kmpl": kmpl.astype(float), "engine_cc": engine_cc.astype(float), "price_lakhs": price,
    })
    # Electric cars have no petrol mileage and no engine displacement: these are "not applicable", not mistakes
    df.loc[df["fuel"] == "Electric", ["mileage_kmpl", "engine_cc"]] = np.nan

    # ---- realistic messiness ----
    not_ev = df.index[df["fuel"] != "Electric"].to_numpy()
    df.loc[rng.choice(not_ev, 70, replace=False), "mileage_kmpl"] = np.nan        # seller left it empty
    df.loc[rng.choice(not_ev, 45, replace=False), "engine_cc"] = np.nan
    df.loc[rng.choice(n, 35, replace=False), "transmission"] = np.nan
    df.loc[rng.choice(n, 25, replace=False), "seller_type"] = np.nan
    blr = df.index[df["city"] == "Bengaluru"].to_numpy()
    df.loc[rng.choice(blr, 40, replace=False), "city"] = "Bangalore"              # old spelling of the same city
    typo = rng.choice(n, 12, replace=False)
    df.loc[typo, "km_driven"] = df.loc[typo, "km_driven"] * 100                    # an extra "00" typed by mistake

    dupes = df.sample(80, random_state=seed)                                      # listing posted twice
    df = pd.concat([df, dupes], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "used_cars.csv")
    data = make_dataset()
    data.to_csv(out, index=False)
    print(f"Wrote {len(data)} rows to {out}")
    print(data["price_lakhs"].describe().round(2).to_string())

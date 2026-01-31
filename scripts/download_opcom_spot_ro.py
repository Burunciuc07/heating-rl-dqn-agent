import os, re, time, datetime as dt
import pandas as pd
import requests

# pagina raportului (RO)
PAGE_URL = "https://www.opcom.ro/grafice-ip-raportPIP-si-volumTranzactionat/ro"

OUT_FILE = "data/environment/spot_prices_ro.csv"
START = dt.date(2025, 11, 1)
END   = dt.date(2026, 1, 31)

def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)

def extract_60min_prices(html: str):
    # Căutăm secțiunea "Pret Mediu 60 min" și apoi valorile pe intervale 1..24.
    # În pagina ta apar ca tabel, dar în HTML sunt de obicei "1|514,79" etc.
    # Extragem 24 numere (cu . mii și , zecimal).
    # Notă: dacă OPCOM schimbă markup-ul, trebuie ajustat regex-ul.

    # 1) restrângem zona ca să nu prindem alte tabele
    m = re.search(r"Pret\\s*Mediu\\s*60\\s*min\\s*\\[?Lei/MWh\\]?(.*?)(ROPEX_DAM\\s*\\[Lei/MWh\\]|$)", html, flags=re.I|re.S)
    if not m:
        raise ValueError("Nu găsesc secțiunea 'Pret Mediu 60 min' în HTML.")

    block = m.group(1)

    # 2) prindem perechi interval + valoare (valoare cu ',' zecimal și '.' mii)
    pairs = re.findall(r">\\s*(\\d{1,2})\\s*<.*?>\\s*([0-9\\.]+,[0-9]+)\\s*<", block, flags=re.S)
    # fallback: dacă nu apare cu '>' '<', prindem simplu "1|514,79" sau "1 514,79"
    if len(pairs) < 24:
        pairs = re.findall(r"(\\b\\d{1,2}\\b)\\s*\\|\\s*([0-9\\.]+,[0-9]+)", block)

    # dedupe + sort by interval
    out = {}
    for k, v in pairs:
        i = int(k)
        if 1 <= i <= 24 and i not in out:
            out[i] = v

    if len(out) != 24:
        raise ValueError(f"M-am așteptat la 24 valori, am găsit {len(out)}. (probabil s-a schimbat formatul)")

    # convert "1.437,66" -> 1437.66
    prices = [float(out[i].replace(".", "").replace(",", ".")) for i in range(1, 25)]
    return prices

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    all_prices = []
    all_timestamps = []

    for d in daterange(START, END):
        # aici e partea sensibilă: pagina are un selector de dată; fără un parametru oficial,
        # uneori site-ul folosește POST / querystring / sesiune.
        # Încercăm parametri uzuali; dacă nu merge, trebuie copiat exact request-ul din browser.
        params_try = [
            {"date": d.strftime("%d/%m/%Y")},
            {"data": d.strftime("%d/%m/%Y")},
            {"reportDate": d.strftime("%d/%m/%Y")},
        ]

        html = None
        for params in params_try:
            r = requests.get(PAGE_URL, params=params, timeout=30)
            if r.status_code == 200 and ("Pret Mediu 60 min" in r.text or "Pret Mediu 60 min" in r.text.replace("\xa0"," ")):
                html = r.text
                break

        if html is None:
            # ca fallback, ia pagina “default” și dă eroare explicită
            r = requests.get(PAGE_URL, timeout=30)
            raise ValueError(
                f"Nu reușesc să setez data {d}. "
                f"Trebuie să îmi trimiți request-ul exact (URL + params) pe care îl face butonul de export."
            )

        prices = extract_60min_prices(html)
        all_prices.extend(prices)

        # timestamps pe ore (opțional, util)
        for h in range(24):
            all_timestamps.append(dt.datetime(d.year, d.month, d.day, h))

        time.sleep(0.3)

    out = pd.DataFrame({
        "timestamp": all_timestamps,
        "price_lei_mwh": all_prices
    })
    out.to_csv(OUT_FILE, index=False)
    print("Saved:", OUT_FILE, "rows:", len(out))

if __name__ == "__main__":
    main()

import pandas as pd
import requests
import streamlit as st

API = "http://localhost:8000"
st.set_page_config(page_title="VentIA", page_icon="📈", layout="wide")
st.title("📈 VentIA · Proyección de ventas")
st.caption("ARIMA · Prophet · Árbol de decisión · XGBoost · LightGBM · GRU · Híbrido CNN-GRU")


def api_get(path):
    r = requests.get(f"{API}{path}", timeout=60)
    r.raise_for_status()
    return r.json()


try:
    models, data = api_get("/models"), api_get("/dataset")
except Exception:
    st.error("No se pudo conectar con el backend. Ejecuta: `cd backend && uvicorn main:app --reload --port 8000`")
    st.stop()

names = {m["id"]: m["name"] for m in models}
with st.sidebar:
    st.header("1 · Datos")
    up = st.file_uploader("CSV o ZIP (fecha + ventas)", type=["csv", "zip"])
    if up and st.session_state.get("up") != (up.name, up.size):
        r = requests.post(f"{API}/dataset", files={"file": (up.name, up.getvalue())}, timeout=600)
        if r.ok:
            st.session_state.pop("res", None)
            st.session_state["up"] = (up.name, up.size)
            st.rerun()
        st.error(r.json().get("detail", "Error al cargar"))
    st.caption(f"**{data['name']}**  \n{data['start']} → {data['end']} · {data['days']} días")

    st.header("2 · Modelos")
    sel = st.multiselect("Modelos a ejecutar", [m["id"] for m in models], format_func=lambda i: names[i],
                         default=[m["id"] for m in models if m["available"]])
    with st.expander("¿Qué hace cada modelo?"):
        for m in models:
            st.markdown(f"**{m['name']}** — {m['desc']}" + ("" if m["available"] else " _(no instalado)_"))

    st.header("3 · Configuración")
    mode = st.radio("Modo", ["Validar con datos reales", "Proyectar al futuro"])
    H = st.slider("Horizonte (días)", 7, 180, 30)
    country = st.selectbox("Feriados", ["Ninguno", "CO", "EC", "US"])
    run = st.button("Ejecutar proyección", type="primary", use_container_width=True)

if run and sel:
    results, bar = [], st.progress(0.0)
    mode_key = "backtest" if mode.startswith("Validar") else "future"
    for i, mid in enumerate(sel):
        bar.progress(i / len(sel), text=f"Entrenando {names[mid]}… (los modelos complejos pueden tardar)")
        r = requests.post(f"{API}/forecast", json={"models": [mid], "horizon": H, "mode": mode_key,
                                                   "country": country}, timeout=3600)
        if not r.ok:
            st.error(r.json().get("detail", r.text))
            break
        res = r.json()
        results += res["results"]
    bar.empty()
    if results:
        st.session_state["res"] = {"mode": mode_key, "dates": res["dates"], "results": results, "H": H}

hist = pd.Series(data["values"], index=pd.to_datetime(data["dates"]))
res = st.session_state.get("res")
df = pd.DataFrame({"Real": hist})
if res:
    for r in res["results"]:
        if r["pred"]:
            df[r["name"]] = pd.Series(r["pred"], index=pd.to_datetime(res["dates"]))
df = df.sort_index().iloc[-(240 + (res["H"] if res and res["mode"] == "future" else 0)):]
st.subheader("Gráfica")
st.line_chart(df)

if res:
    ok = [r for r in res["results"] if r["pred"]]
    for r in res["results"]:
        if r["error"]:
            st.warning(f"{r['name']}: {r['error']}")
    if res["mode"] == "backtest" and ok:
        t = pd.DataFrame([{"Modelo": r["name"], **r["metrics"], "Tiempo (s)": r["seconds"]} for r in ok]).sort_values("RMSLE")
        st.success(f"Mejor modelo (menor RMSLE): **{t.iloc[0]['Modelo']}**")
        st.subheader(f"Métricas en los últimos {res['H']} días (menor es mejor)")
        st.dataframe(t.set_index("Modelo"), use_container_width=True)
    elif ok:
        t = pd.DataFrame([{"Modelo": r["name"], "Total proyectado": round(sum(r["pred"])),
                           "Promedio diario": round(sum(r["pred"]) / len(r["pred"])), "Tiempo (s)": r["seconds"]} for r in ok])
        st.subheader(f"Proyección a {res['H']} días")
        st.dataframe(t.set_index("Modelo"), use_container_width=True)
else:
    st.info("Elige modelos en el panel izquierdo y pulsa **Ejecutar proyección**.")

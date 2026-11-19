import time
 
import altair as alt
import pandas as pd
import requests
import streamlit as st
 
import imagenes as im
 
API = "http://localhost:8000"
st.set_page_config(page_title="VentIA · Predicción de ventas", page_icon=":material/show_chart:", layout="wide")
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&display=swap');
.block-container {max-width: 1080px; padding-top: 1.6rem;}
.stApp h1, .stApp h2, .stApp h3, .serif {font-family: 'Source Serif 4', Georgia, serif; font-weight: 600;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 6px; border-color: #D9E1E8;}
 
.barra {display: flex; align-items: center; gap: .7rem; padding-bottom: .9rem; margin-bottom: 2rem;
        border-bottom: 1px solid #D9E1E8;}
.marca {font-family: 'Source Serif 4', Georgia, serif; font-weight: 700; font-size: 1.35rem; color: #1D4E89;}
.barra .sub {margin-left: auto; font-size: .9rem; color: #6B7C8C;}
 
.eyebrow {font-size: .8rem; letter-spacing: .09em; text-transform: uppercase; font-weight: 700;
          color: #2E8B6E; margin-bottom: .8rem;}
.titulo {font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 2.6rem; line-height: 1.15;
         margin: 0 0 1rem 0; color: #1F2D3A;}
.lead {font-size: 1.12rem; line-height: 1.55; color: #44566A; margin: 0;}
 
.dato {border-top: 2px solid #1D4E89; padding-top: .7rem; margin-top: 1.6rem;}
.dato b {display: block; margin-bottom: .25rem;}
.dato p {margin: 0; color: #44566A; font-size: .97rem; line-height: 1.5;}
 
.pasos {display: flex; flex-wrap: wrap; gap: .4rem 1.6rem; margin: 1.4rem 0 .2rem 0; font-size: .92rem; color: #6B7C8C;}
.pasos b {color: #1D4E89; margin-right: .3rem;}
 
.sec {display: flex; justify-content: space-between; align-items: center; margin: 2.6rem 0 1rem 0;
      padding-bottom: .7rem; border-bottom: 1px solid #D9E1E8;}
.sec-n {font-size: .78rem; letter-spacing: .09em; text-transform: uppercase; font-weight: 700; color: #6B7C8C;}
.sec-n.ok {color: #2E8B6E;}
.sec-t {font-family: 'Source Serif 4', Georgia, serif; font-size: 1.65rem; font-weight: 600; color: #1F2D3A;}
 
.mod-top {display: flex; justify-content: space-between; align-items: center; margin-bottom: .4rem;}
.etq {font-size: .72rem; letter-spacing: .08em; font-weight: 700; color: #6B7C8C;}
.etq.rec {color: #2E8B6E;}
.etq.fast {color: #C9822B;}
.mini {font-size: .9rem; color: #6B7C8C;}
</style>
""", unsafe_allow_html=True)
 
PALETA = ["#6B7C8C", "#1D4E89", "#2E8B6E", "#C9822B", "#9B4F5F", "#4A90A4"]
ICONOS_MODELO = {"naive": im.REFERENCIA, "lightgbm": im.ARBOLES, "gru": im.RNN, "lstm": im.RNN}
ss = st.session_state
 
 
# ----------------------------------------------------------------------------- utilidades
def api_get(path):
    r = requests.get(f"{API}{path}", timeout=60)
    r.raise_for_status()
    return r.json()
 
 
def err_msg(r):
    try:
        d = r.json().get("detail")
    except Exception:
        d = None
    return d if isinstance(d, str) else f"Ocurrió un error ({r.status_code}). Inténtalo de nuevo."
 
 
def seccion(num, titulo, icono, hecho=False):
    etiqueta = f"Paso {num}" + (" · completado" if hecho else "")
    st.markdown(f'<div class="sec"><div><div class="sec-n {"ok" if hecho else ""}">{etiqueta}</div>'
                f'<div class="sec-t">{titulo}</div></div>{im.img(icono, 44, 44)}</div>', unsafe_allow_html=True)
 
 
def clase_etiqueta(b):
    return {"Recomendado": "rec", "Rápido": "fast"}.get(b, "")
 
 
def fmt_tiempo(seg):
    seg = int(seg)
    return f"{seg // 60}:{seg % 60:02d}"
 
 
# ----------------------------------------------------------------------------- encabezado
st.markdown(f'<div class="barra">{im.img(im.LOGO, 34, 34)}<span class="marca">VentIA</span>'
            '<span class="sub">Proyección de ventas</span></div>', unsafe_allow_html=True)
 
portada_izq, portada_der = st.columns([6, 5], vertical_alignment="center")
with portada_izq:
    st.markdown("""
<div class="eyebrow">Predicción de ventas con redes neuronales</div>
<h1 class="titulo">Anticipa cómo se van a vender tus próximos días</h1>
<p class="lead">Sube el historial de ventas de tu negocio. La página prueba varios modelos con tus propios datos
y te muestra cuál acierta más y qué podría pasar en los días que vienen.</p>
""", unsafe_allow_html=True)
with portada_der:
    st.markdown(im.img(im.portada(), extra="width:100%;height:auto;"), unsafe_allow_html=True)
 
try:
    models, data = api_get("/models"), api_get("/dataset")
except Exception:
    st.error("No se pudo conectar con el servidor de cálculo (backend). Ábrelo en otra terminal con:")
    st.code("cd backend\nuvicorn main:app --reload --port 8000")
    st.stop()
 
d1, d2, d3 = st.columns(3, gap="large")
d1.markdown('<div class="dato"><b>Qué necesitas</b><p>Un archivo con la fecha y las ventas de cada día, de al menos '
            '120 días. Lo ideal es entre uno y dos años.</p></div>', unsafe_allow_html=True)
d2.markdown('<div class="dato"><b>Qué obtienes</b><p>Una predicción de tus próximos días, o una prueba de qué tan '
            'bien acierta cada modelo con tus datos reales.</p></div>', unsafe_allow_html=True)
d3.markdown('<div class="dato"><b>Cuánto tarda</b><p>Segundos con los modelos rápidos y varios minutos con las redes '
            'neuronales. Verás una barra de progreso.</p></div>', unsafe_allow_html=True)
 
st.markdown('<div class="pasos"><span><b>1</b>Datos</span><span><b>2</b>Columnas</span><span><b>3</b>Objetivo</span>'
            '<span><b>4</b>Modelos</span><span><b>5</b>Resultados</span></div>', unsafe_allow_html=True)
 
with st.expander("Qué puede y qué no puede hacer esta página"):
    st.markdown("""
**Sí puede**
- Detectar la fecha y las ventas en tu archivo; tú confirmas las columnas.
- Sumar por día si tu archivo tiene varias filas por día (por tienda o por producto).
- Probar los modelos con tus últimos días reales para medir su precisión.
- Predecir los próximos días y dejarte descargar el resultado.
 
**No puede**
- Predecir bien con pocos datos: con menos de un año de historial la predicción es poco confiable.
- Trabajar con datos semanales o mensuales: necesita datos diarios.
- Adivinar cambios bruscos que nunca aparecieron en tu historial, como una promoción especial, una crisis o un cierre.
- Garantizar la exactitud: cuanto más lejos en el futuro, menos precisa es la predicción.
""")
 
# ----------------------------------------------------------------------------- paso 1: datos
ss.setdefault("up_n", 0)
seccion(1, "Sube tus datos de ventas", im.SUBIR, hecho=bool(ss.get("confirmed")))
st.markdown("Acepta archivos **CSV, Excel (.xlsx) o ZIP**. Si tu archivo tiene varias filas por día, por ejemplo una "
            "por producto, las sumamos.")
col_up, col_ej = st.columns([3, 1])
up = col_up.file_uploader("Arrastra tu archivo aquí", type=["csv", "xlsx", "zip"],
                          label_visibility="collapsed", key=f"up{ss['up_n']}")
col_ej.markdown('<div class="mini">¿Solo quieres probar?</div>', unsafe_allow_html=True)
if col_ej.button("Usar datos de ejemplo"):
    requests.post(f"{API}/dataset/example", timeout=60)
    for k in ("preview", "confirmed", "res", "up_key", "up_err"):
        ss.pop(k, None)
    ss["up_n"] += 1
    st.rerun()
 
if up:
    key = (up.name, up.size)
    if ss.get("up_key") != key:
        ss["up_key"] = key
        for k in ("confirmed", "res"):
            ss.pop(k, None)
        try:
            with st.spinner("Leyendo tu archivo…"):
                r = requests.post(f"{API}/dataset/preview", files={"file": (up.name, up.getvalue())}, timeout=600)
            ss["preview"], ss["up_err"] = (r.json(), None) if r.ok else (None, err_msg(r))
        except requests.RequestException:
            ss["preview"], ss["up_err"] = None, "Se perdió la conexión con el servidor. Revisa que el backend siga abierto."
    if ss.get("up_err"):
        st.error(ss["up_err"])
 
# ----------------------------------------------------------------------------- paso 2: columnas
pv = ss.get("preview")
if pv:
    seccion(2, "Confirma las columnas", im.COLUMNAS, hecho=bool(ss.get("confirmed")))
    st.markdown(f"Leí **{pv['rows']:,}** filas de **{pv['name']}**. Así se ve el comienzo del archivo:")
    st.dataframe(pd.DataFrame(pv["sample"]), hide_index=True)
    cols = pv["columns"]
    if pv["suggest_date"] and pv["suggest_value"]:
        st.info("Detecté las columnas por ti. Revisa que sean las correctas y confirma.")
    else:
        st.warning("No pude detectar todas las columnas. Elígelas tú: necesito una con fechas y otra con ventas.")
    a, b, c = st.columns(3, gap="large")
    i_f = cols.index(pv["suggest_date"]) if pv["suggest_date"] in cols else 0
    i_v = cols.index(pv["suggest_value"]) if pv["suggest_value"] in cols else min(1, len(cols) - 1)
    col_f = a.selectbox("Columna de fechas", cols, index=i_f, key=f"cf_{pv['name']}",
                        format_func=lambda x: f"{x}  (sugerida)" if x == pv["suggest_date"] else x)
    col_v = b.selectbox("Columna de ventas", cols, index=i_v, key=f"cv_{pv['name']}",
                        format_func=lambda x: f"{x}  (sugerida)" if x == pv["suggest_value"] else x)
    opciones = {"Rellenar con 0 (no hubo ventas)": "cero", "Estimar con los días vecinos": "interpolar"}
    rel = c.radio("Si faltan días en el archivo", list(opciones), key=f"rel_{pv['name']}",
                  help="Por ejemplo, si tu negocio cierra los domingos y no hay filas esos días, 'Rellenar con 0' "
                       "es lo correcto. Si fue un olvido al registrar, es mejor estimarlos.")
    if st.button("Confirmar columnas", type="primary"):
        r = requests.post(f"{API}/dataset/confirm", json={"date_col": col_f, "value_col": col_v,
                                                          "fill": opciones[rel]}, timeout=600)
        if r.ok:
            ss["confirmed"] = r.json()
            ss.pop("res", None)
            st.rerun()
        st.error(err_msg(r))
    if ss.get("confirmed"):
        st.success(f"Usaremos {ss['confirmed']['days']:,} días de ventas.")
        for w in ss["confirmed"].get("warnings", []):
            st.warning(w)
 
# ----------------------------------------------------------------------------- datos en uso
hist = pd.Series(data["values"], index=pd.to_datetime(data["dates"]))
st.markdown("&nbsp;", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown(f"**Datos en uso:** {data['name']}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Periodo", f"{data['start']} a {data['end']}")
    m2.metric("Días", f"{data['days']:,}")
    m3.metric("Ventas promedio por día", f"{hist.mean():,.0f}")
    st.line_chart(hist, height=150, color="#1D4E89")
    if pv and not ss.get("confirmed"):
        st.caption("Estos todavía no son los datos de tu archivo: confirma las columnas en el paso 2.")
 
# ----------------------------------------------------------------------------- paso 3: objetivo
seccion(3, "Elige qué quieres hacer", im.OBJETIVO)
mode = st.radio("Objetivo", ["Probar la precisión", "Predecir el futuro"], horizontal=True, label_visibility="collapsed")
backtest = mode == "Probar la precisión"
if backtest:
    st.caption("Escondemos los últimos días de tu historial, le pedimos al modelo que los adivine y comparamos con lo "
               "que realmente pasó. Así sabes qué tan confiable es.")
else:
    st.caption("Entrenamos con todo tu historial y proyectamos los días siguientes al último dato.")
max_h = min(90, data["days"] - 120) if backtest else 90
if max_h < 7:
    st.error("Para probar la precisión necesitas al menos 127 días de datos. Sube un historial más largo "
             "o elige 'Predecir el futuro'.")
    H = 7
elif max_h == 7:
    H = 7
    st.caption("Con tu cantidad de datos solo se pueden evaluar 7 días.")
else:
    H = st.slider("¿Cuántos días quieres evaluar?" if backtest else "¿Cuántos días quieres predecir?",
                  7, max_h, min(30, max_h))
    if H > 30:
        st.caption("A más de 30 días la precisión baja bastante: tómalo como una tendencia, no como un dato exacto.")
 
# ----------------------------------------------------------------------------- paso 4: modelos
seccion(4, "Elige los modelos", im.MODELOS)
st.caption("Puedes marcar varios y compararlos. Si no sabes cuáles, deja los que vienen marcados.")
cols_m = st.columns(len(models))
elegidos = []
for col, m in zip(cols_m, models):
    with col, st.container(border=True):
        st.markdown(f'<div class="mod-top">{im.img(ICONOS_MODELO.get(m["id"], im.MODELOS), 36, 36)}'
                    f'<span class="etq {clase_etiqueta(m["badge"])}">{m["badge"].upper()}</span></div>',
                    unsafe_allow_html=True)
        st.markdown(f"**{m['name']}**")
        st.caption(m["plain"])
        st.caption(f"Tiempo aproximado: {m['speed'].lower()}")
        if not m["available"]:
            st.warning("No está instalado en este computador.")
        if st.checkbox("Usar este modelo", value=m["default"] and m["available"], key=f"m_{m['id']}",
                       disabled=not m["available"]):
            elegidos.append(m["id"])
usa_redes = any(i in ("gru", "lstm") for i in elegidos)
quality = "rapida"
if usa_redes:
    q = st.radio("Calidad de las redes neuronales",
                 ["Rápida (recomendada para probar)", "Completa (más estable, tarda unas 3 veces más)"],
                 horizontal=True)
    quality = "rapida" if q.startswith("Rápida") else "completa"
with st.expander("Opciones avanzadas"):
    pais = st.selectbox("Feriados a tener en cuenta", ["Ninguno", "CO", "EC", "US"],
                        format_func=lambda x: {"Ninguno": "Ninguno", "CO": "Colombia", "EC": "Ecuador",
                                               "US": "Estados Unidos"}[x])
 
# ----------------------------------------------------------------------------- ejecutar
st.markdown("&nbsp;", unsafe_allow_html=True)
ejecutando = bool(ss.get("job"))
run = st.button("Ejecutar análisis", type="primary", disabled=(not elegidos) or max_h < 7 or ejecutando)
if not elegidos:
    st.caption("Marca al menos un modelo para poder ejecutar.")
if run:
    r = requests.post(f"{API}/forecast/start", json={"models": elegidos, "horizon": H,
                                                    "mode": "backtest" if backtest else "future",
                                                    "country": pais, "quality": quality}, timeout=60)
    if r.ok:
        ss["job"] = {"id": r.json()["job_id"], "t0": time.time()}
        ss.pop("res", None)
        st.rerun()
    st.error(err_msg(r))
 
if ss.get("job"):
    job = ss["job"]
    barra = st.progress(0.0, text="Preparando los datos…")
    while True:
        try:
            r = requests.get(f"{API}/forecast/status/{job['id']}", timeout=30)
        except requests.RequestException:
            ss.pop("job", None)
            st.error("Se perdió la conexión con el servidor. Revisa que el backend siga abierto y vuelve a ejecutar.")
            break
        if not r.ok:
            ss.pop("job", None)
            st.error(err_msg(r))
            break
        s = r.json()
        if s["status"] == "running":
            barra.progress(min(float(s["progress"]), 0.99),
                           text=f"{s['message']}  ·  {fmt_tiempo(time.time() - job['t0'])} transcurridos")
            time.sleep(1.5)
        elif s["status"] == "done":
            ss["res"] = s["result"]
            ss["res_seg"] = s["seconds"]
            ss.pop("job", None)
            st.rerun()
        else:
            ss.pop("job", None)
            st.error(s["error"] or "El análisis falló.")
            break
 
# ----------------------------------------------------------------------------- resultados
res = ss.get("res")
if res and not ss.get("job"):
    ok = [r for r in res["results"] if r["pred"]]
    for r in res["results"]:
        if r["error"]:
            st.warning(f"{r['name']} no pudo ejecutarse: {r['error']}")
    if not ok:
        st.stop()
    es_bt = res["mode"] == "backtest"
    fechas_p = pd.to_datetime(res["dates"])
 
    seccion(5, "Resultados", im.RESULTADOS, hecho=True)
    st.caption(f"Análisis terminado en {fmt_tiempo(ss.get('res_seg', 0))}.")
    ventana = res["horizon"] + 90 if es_bt else 120
    h = hist.iloc[-ventana:]
    largo = [pd.DataFrame({"Fecha": h.index, "Ventas": h.values, "Serie": "Real"})]
    for r in ok:
        largo.append(pd.DataFrame({"Fecha": fechas_p, "Ventas": r["pred"], "Serie": r["name"]}))
    largo = pd.concat(largo)
    orden = ["Real"] + [r["name"] for r in ok]
    lineas = alt.Chart(largo).mark_line(strokeWidth=2).encode(
        x=alt.X("Fecha:T", title=None),
        y=alt.Y("Ventas:Q", title="Ventas", scale=alt.Scale(zero=False)),
        color=alt.Color("Serie:N", sort=orden, scale=alt.Scale(domain=orden, range=PALETA[:len(orden)]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("Fecha:T"), "Serie:N", alt.Tooltip("Ventas:Q", format=",.0f")])
    regla = alt.Chart(pd.DataFrame({"Fecha": [fechas_p[0]]})).mark_rule(strokeDash=[5, 5], color="#888").encode(x="Fecha:T")
    st.altair_chart((lineas + regla).properties(height=380, width="container"))
    st.caption("La línea punteada marca dónde empiezan las predicciones.")
 
    pred_df = pd.DataFrame({"Fecha": res["dates"], **{r["name"]: r["pred"] for r in ok}})
    if es_bt:
        tabla = pd.DataFrame([{"Modelo": r["name"], "Error promedio por día (MAE)": r["metrics"]["MAE"],
                               "Error en % (MAPE)": r["metrics"]["MAPE"], "RMSE": r["metrics"]["RMSE"],
                               "RMSLE": r["metrics"]["RMSLE"], "Tiempo (s)": r["seconds"]} for r in ok]
                             ).sort_values("RMSLE")
        mejor = tabla.iloc[0]
        mape = mejor["Error en % (MAPE)"]
        st.success(f"Mejor modelo: {mejor['Modelo']}. En los últimos {res['horizon']} días se equivocó, en promedio, "
                   f"por {mejor['Error promedio por día (MAE)']:,.0f} unidades al día"
                   + (f" ({mape:.1f}%)." if pd.notna(mape) else "."))
        base = tabla[tabla["Modelo"].str.startswith("Naive")]
        if len(base) and mejor["Modelo"] != base.iloc[0]["Modelo"]:
            mej = 100 * (1 - mejor["Error promedio por día (MAE)"] / base.iloc[0]["Error promedio por día (MAE)"])
            if mej > 0:
                st.info(f"Supera al modelo de referencia: se equivoca un {mej:.0f}% menos.")
        st.dataframe(tabla.set_index("Modelo"))
        with st.expander("Cómo leer estas medidas"):
            st.markdown("""
Todas miden cuánto se equivocó el modelo, así que **menor es mejor**.
- **MAE:** cuántas unidades se equivoca en promedio cada día. Es la más fácil de entender.
- **MAPE:** lo mismo, pero en porcentaje de las ventas reales.
- **RMSE:** parecida al MAE, pero castiga más los errores grandes.
- **RMSLE:** mide el error en proporción, útil cuando las ventas varían mucho de tamaño. Con ella elegimos el mejor modelo.
""")
    else:
        tabla = pd.DataFrame([{"Modelo": r["name"], "Total proyectado": round(sum(r["pred"])),
                               "Promedio por día": round(sum(r["pred"]) / len(r["pred"])),
                               "Tiempo (s)": r["seconds"]} for r in ok])
        st.success(f"Proyección de los próximos {res['horizon']} días lista.")
        st.dataframe(tabla.set_index("Modelo"))
    if res["horizon"] > 14:
        st.warning("Las predicciones a más de 14 días son orientativas: sirven para ver la tendencia, "
                   "no para contar con el número exacto de cada día.")
    st.download_button("Descargar las predicciones (CSV)", pred_df.to_csv(index=False).encode("utf-8"),
                       file_name="predicciones_ventia.csv", mime="text/csv")
elif not ss.get("job"):
    st.info("Cuando estés listo, pulsa Ejecutar análisis. Los resultados aparecerán aquí.")

"""Ilustraciones de la página (logo, portada e íconos), dibujadas como SVG para no depender de archivos externos."""
import base64
 
AZUL, VERDE, OCRE, TINTA = "#1D4E89", "#2E8B6E", "#C9822B", "#1F2D3A"
SUAVE, ARENA = "#E4ECF2", "#F1E7D3"
 
 
def _uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
 
 
def img(svg: str, ancho=None, alto=None, extra="") -> str:
    """Devuelve una etiqueta <img> lista para usar dentro de st.markdown(..., unsafe_allow_html=True)."""
    medidas = (f'width="{ancho}" ' if ancho else "") + (f'height="{alto}" ' if alto else "")
    return f'<img src="{_uri(svg)}" {medidas}style="display:block;{extra}" alt="">'
 
 
def _icono(contenido: str) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" fill="none" stroke="%s" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round">%s</svg>' % (AZUL, contenido))
 
 
LOGO = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><rect width="40" height="40" rx="9" fill="%s"/>'
        '<rect x="9" y="22" width="5" height="9" rx="1.5" fill="#fff"/>'
        '<rect x="17.5" y="16" width="5" height="15" rx="1.5" fill="#fff"/>'
        '<rect x="26" y="9" width="5" height="22" rx="1.5" fill="#8fd9b9"/></svg>' % AZUL)
 
# --- íconos de cada paso ---------------------------------------------------------------------------
SUBIR = _icono('<path d="M13 6h15l8 8v28H13z"/><path d="M28 6v8h8"/><path d="M24.5 36V23"/><path d="M19 28.5l5.5-5.5 5.5 5.5"/>')
COLUMNAS = _icono('<rect x="19" y="10" width="11" height="28" fill="%s" stroke="none" opacity=".35"/>'
                  '<rect x="8" y="10" width="32" height="28" rx="3"/><path d="M19 10v28M30 10v28M8 19h32M8 28h32"/>' % VERDE)
OBJETIVO = _icono('<rect x="8" y="12" width="32" height="28" rx="3"/><path d="M8 20h32M16 8v7M32 8v7"/>'
                  '<rect x="26" y="26" width="8" height="8" rx="1.5" fill="%s" stroke="none" opacity=".6"/>'
                  '<path d="M15 27h4M15 33h4"/>' % OCRE)
MODELOS = _icono('<path d="M12 24L24 12M12 24l12 12M24 12l12 12M24 36l12-12"/>'
                 '<circle cx="12" cy="24" r="4" fill="#fff"/><circle cx="24" cy="12" r="4" fill="#fff"/>'
                 '<circle cx="24" cy="36" r="4" fill="#fff"/><circle cx="36" cy="24" r="4" fill="%s" stroke="%s"/>' % (VERDE, VERDE))
RESULTADOS = _icono('<path d="M8 8v32h32"/><path d="M14 32l8-10 6 6 10-14"/><circle cx="38" cy="14" r="3" fill="%s" stroke="%s"/>' % (VERDE, VERDE))
 
# --- íconos por tipo de modelo ---------------------------------------------------------------------
REFERENCIA = _icono('<path d="M8 30h32" stroke-dasharray="3 4"/><path d="M8 18h32"/><path d="M8 14v8M40 14v8"/>')
ARBOLES = _icono('<path d="M24 40V26M24 26L14 16M24 26l10-10M14 16v-6M34 16v-6"/>'
                 '<circle cx="14" cy="9" r="3" fill="#fff"/><circle cx="34" cy="9" r="3" fill="#fff"/>'
                 '<circle cx="24" cy="26" r="3" fill="%s" stroke="%s"/>' % (VERDE, VERDE))
RNN = _icono('<path d="M36 19a13 13 0 1 0 1.5 9"/><path d="M37 10v9h-9"/><circle cx="24" cy="24" r="3" fill="%s" stroke="%s"/>' % (VERDE, VERDE))
 
 
def portada() -> str:
    """Ilustración principal: historial de ventas (barras y línea) con la predicción punteada."""
    hist = [30, 44, 52, 48, 58, 74, 40, 34, 48, 56, 52, 64, 82, 46, 38, 50, 60, 56, 68, 88, 50, 42]
    futuro = [54, 64, 60, 72, 92, 54, 46, 58, 66, 62]
    x0, paso, base, esc = 86, 11.5, 252, 1.45
    barras, pts = [], []
    for i, v in enumerate(hist):
        x, y = x0 + i * paso, base - v * esc
        barras.append('<rect x="%.1f" y="%.1f" width="7" height="%.1f" rx="1.5" fill="%s" opacity=".28"/>' % (x - 3.5, y, v * esc, AZUL))
        pts.append("%.1f,%.1f" % (x, y))
    xf = x0 + (len(hist) - 1) * paso
    pf = ["%.1f,%.1f" % (xf, base - hist[-1] * esc)]
    for j, v in enumerate(futuro, start=1):
        pf.append("%.1f,%.1f" % (xf + j * paso, base - v * esc))
    corte = xf + paso / 2
    grid = "".join('<path d="M70 %d H448" stroke="#D9E2EA" stroke-width="1"/>' % y for y in (84, 132, 180, 228))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 340">'
        '<ellipse cx="262" cy="176" rx="236" ry="150" fill="%(suave)s"/>'
        '<circle cx="478" cy="62" r="22" fill="%(arena)s"/><circle cx="46" cy="290" r="14" fill="%(arena)s"/>'
        '<rect x="46" y="44" width="420" height="236" rx="14" fill="#fff" stroke="#D5DEE6"/>'
        '%(grid)s'
        '<rect x="%(corte).1f" y="58" width="%(ancho).1f" height="196" fill="%(verde)s" opacity=".07"/>'
        '<path d="M%(corte).1f 58V254" stroke="%(verde)s" stroke-width="1.5" stroke-dasharray="4 4"/>'
        '%(barras)s'
        '<polyline points="%(pts)s" fill="none" stroke="%(azul)s" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"/>'
        '<polyline points="%(pf)s" fill="none" stroke="%(verde)s" stroke-width="2.6" stroke-dasharray="6 5" stroke-linejoin="round" stroke-linecap="round"/>'
        '<path d="M70 254H448" stroke="#AEBBC7" stroke-width="1.5"/>'
        '<rect x="368" y="214" width="112" height="84" rx="12" fill="#fff" stroke="#D5DEE6"/>'
        '<path d="M368 238H480V226a12 12 0 0 0-12-12H380a12 12 0 0 0-12 12z" fill="%(ocre)s"/>'
        '%(cal)s'
        '</svg>'
    ) % dict(suave=SUAVE, arena=ARENA, grid=grid, corte=corte, ancho=448 - corte, verde=VERDE, azul=AZUL, ocre=OCRE,
             barras="".join(barras), pts=" ".join(pts), pf=" ".join(pf),
             cal="".join('<circle cx="%d" cy="%d" r="3.2" fill="%s" opacity="%s"/>' % (
                 384 + c * 16, 252 + r * 14, VERDE if (r, c) == (1, 3) else "#9FB0BF", 1 if (r, c) == (1, 3) else .7)
                 for r in range(3) for c in range(6)))
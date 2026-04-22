"""
Proyección mensual y semanal de cierre de campañas comerciales
Combina tres métodos de estimación con ponderación adaptativa.

Métodos:
  1. Extrapolación por ritmo diario del mes en curso
  2. Referencia al mes anterior (estabiliza proyecciones tempranas)
  3. Prophet estacional (Meta) — aprende patrones anuales del histórico

Lógica de pesos:
  - Primeros días del mes : más peso a Prophet y mes anterior
  - Últimos días del mes  : casi todo el peso al ritmo real actual
  - Piso de seguridad     : la proyección nunca puede ser menor que el real acumulado

El código se ejecuta automáticamente mes a mes sin cambios manuales.
Los resultados se envían por correo en formato HTML ejecutivo.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitacora_operaciones.xlsx")

# Detectar columnas TC dinámicamente (tolerante a espacios/mayúsculas)
_cols_excel = pd.read_excel(FILE_PATH, nrows=0).columns.tolist()

def _col_tc(keyword):
    matches = [c for c in _cols_excel if c.strip().startswith("TC") and keyword in c]
    if not matches:
        raise ValueError(f"No se encontró columna TC con '{keyword}'. Disponibles: {_cols_excel}")
    return matches[0]

COL_TC_TITULAR   = _col_tc("TITULAR")
COL_TC_ADICIONAL = _col_tc("ADICIONAL")

# Columnas del Excel
COLUMNAS = [
    "PRODUCTO_A Nro. Op.",
    "PRODUCTO_A Monto MM$",
    "PRODUCTO_B Nro. Op.",
    "PRODUCTO_B Monto MM$",
    COL_TC_TITULAR,
    COL_TC_ADICIONAL,
]

# Agrupación de campañas
CAMPAÑAS = {
    "PRODUCTO_A": {
        "qty":    "PRODUCTO_A Nro. Op.",
        "monto":  "PRODUCTO_A Monto MM$",
        "ticket": True,
    },
    "PRODUCTO_B": {
        "qty":    "PRODUCTO_B Nro. Op.",
        "monto":  "PRODUCTO_B Monto MM$",
        "ticket": True,
    },
    "TDC_TITULAR": {
        "qty":    COL_TC_TITULAR,
        "monto":  None,
        "ticket": False,
    },
    "TDC_ADICIONAL": {
        "qty":    COL_TC_ADICIONAL,
        "monto":  None,
        "ticket": False,
    },
}

# ── Configuración de correo ────────────────────────────────────
# Las credenciales se leen de variables de entorno (nunca en el código)
SMTP_SERVER    = "smtp-mail.outlook.com"
SMTP_PORT      = 587
SMTP_USER      = os.environ.get("SMTP_USER", "reporte@miempresa.com")
SMTP_PASSWORD  = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM_NAME = "Reporting Automático"

TO_LIST = [
    os.environ.get("MAIL_TO_1", "gerente1@miempresa.com"),
    os.environ.get("MAIL_TO_2", "gerente2@miempresa.com"),
]
CC_LIST = [os.environ.get("MAIL_CC", "control@miempresa.com")]

# ─────────────────────────────────────────────────────────────
# 2. CARGA Y PREPARACIÓN
# ─────────────────────────────────────────────────────────────
df = pd.read_excel(FILE_PATH, header=0)
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA").reset_index(drop=True)
df["mes"] = df["FECHA"].dt.to_period("M")

# Acumulado mensual real
monthly = df.groupby("mes")[COLUMNAS].sum().reset_index()
monthly["mes_dt"] = monthly["mes"].dt.to_timestamp()
monthly["dias"]   = df.groupby("mes")["FECHA"].count().values

# ─────────────────────────────────────────────────────────────
# 3. IDENTIFICAR MES EN CURSO Y PARÁMETROS DE AVANCE
# ─────────────────────────────────────────────────────────────
mes_actual   = df["mes"].max()
mes_anterior = mes_actual - 1

dias_registrados  = int(monthly.loc[monthly["mes"] == mes_actual,  "dias"].values[0])
dias_mes_anterior = int(monthly.loc[monthly["mes"] == mes_anterior, "dias"].values[0])

meses_completos   = monthly[monthly["mes"] < mes_actual]
dias_promedio_mes = meses_completos["dias"].mean()
fraccion          = dias_registrados / dias_promedio_mes

datos_mes_anterior = df[df["mes"] == mes_anterior]
datos_mes_actual   = df[df["mes"] == mes_actual]

promedio_diario_mes_anterior = {col: datos_mes_anterior[col].mean() for col in COLUMNAS}
promedio_diario_mes_actual   = {col: datos_mes_actual[col].mean()   for col in COLUMNAS}

print("=" * 70)
print(f"  PROYECCIÓN DE CIERRE MENSUAL Y SEMANAL")
print("=" * 70)
print(f"  Mes en curso        : {mes_actual}")
print(f"  Días con datos      : {dias_registrados}")
print(f"  Días prom. mes tipo : {dias_promedio_mes:.1f}")
print(f"  Avance del mes      : {fraccion:.1%}")
print("=" * 70)


# ─────────────────────────────────────────────────────────────
# 4. FUNCIÓN PRINCIPAL DE PROYECCIÓN MENSUAL
# ─────────────────────────────────────────────────────────────
def proyectar(monthly_df, col, mes_actual, fraccion,
              dias_promedio_mes, dias_mes_anterior,
              prom_diario_actual, prom_diario_anterior):

    real_parcial = monthly_df.loc[monthly_df["mes"] == mes_actual, col].values[0]

    # ── Método 1: Extrapolación por ritmo actual ──────────────
    dias_base = max(dias_promedio_mes, dias_registrados)
    proy_ritmo_actual = prom_diario_actual[col] * dias_base

    # ── Método 2: Referencia al mes anterior ─────────────────
    proy_mes_anterior = prom_diario_anterior[col] * dias_base

    # ── Método 3: Prophet estacional ─────────────────────────
    historico = monthly_df[monthly_df["mes"] < mes_actual][["mes_dt", col]].copy()
    historico.columns = ["ds", "y"]
    historico = historico.dropna()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        interval_width=0.80,
    )
    model.fit(historico)
    future   = model.make_future_dataframe(periods=1, freq="MS")
    forecast = model.predict(future)
    fila_pred = forecast[forecast["ds"] == mes_actual.to_timestamp()]
    proy_prophet = fila_pred["yhat"].values[0]
    ic_lower     = fila_pred["yhat_lower"].values[0]
    ic_upper     = fila_pred["yhat_upper"].values[0]

    # Cap de seguridad: Prophet no puede superar 1.25× el ritmo real
    if proy_ritmo_actual > 0:
        proy_prophet = min(proy_prophet, proy_ritmo_actual * 1.25)

    # ── Ponderación adaptativa ────────────────────────────────
    # Prophet se elimina al superar 70% del mes (los datos reales ya dominan).
    peso_prophet      = max(0.0, 0.15 * (1 - fraccion / 0.70)) if fraccion < 0.70 else 0.0
    peso_ritmo_actual = min(0.85, fraccion * 1.20)
    peso_mes_anterior = max(0.05, 1 - peso_ritmo_actual - peso_prophet)

    # Normalizar para que sumen 1
    total_peso = peso_ritmo_actual + peso_mes_anterior + peso_prophet
    peso_ritmo_actual /= total_peso
    peso_mes_anterior /= total_peso
    peso_prophet      /= total_peso

    proy_final = (
        peso_ritmo_actual * proy_ritmo_actual +
        peso_mes_anterior * proy_mes_anterior +
        peso_prophet      * proy_prophet
    )

    # Piso de seguridad: nunca menor que lo ya registrado
    proy_final = max(proy_final, real_parcial)

    return {
        "real_parcial"      : real_parcial,
        "proy_ritmo_actual" : round(proy_ritmo_actual),
        "proy_mes_anterior" : round(proy_mes_anterior),
        "proy_prophet"      : round(proy_prophet),
        "proyeccion_final"  : round(proy_final),
        "ic_lower"          : round(ic_lower),
        "ic_upper"          : round(ic_upper),
        "pesos"             : (peso_ritmo_actual, peso_mes_anterior, peso_prophet),
    }


# ─────────────────────────────────────────────────────────────
# 5. EJECUTAR PARA TODAS LAS COLUMNAS
# ─────────────────────────────────────────────────────────────
resultados = {}
for col in COLUMNAS:
    print(f"  Modelando: {col[:55]}...")
    resultados[col] = proyectar(
        monthly, col, mes_actual, fraccion,
        dias_promedio_mes, dias_mes_anterior,
        promedio_diario_mes_actual,
        promedio_diario_mes_anterior,
    )

# ─────────────────────────────────────────────────────────────
# 6. CALCULAR TICKET PROMEDIO Y ARMAR RESUMEN DE CAMPAÑAS
# ─────────────────────────────────────────────────────────────
def ticket(monto, qty):
    """Ticket promedio seguro (evita división por cero)."""
    return round(monto / qty, 1) if qty and qty > 0 else 0

resumen_campañas = {}
for nombre, cfg in CAMPAÑAS.items():
    r_qty = resultados[cfg["qty"]]
    entry = {
        "real_qty"     : r_qty["real_parcial"],
        "proy_qty"     : r_qty["proyeccion_final"],
        "ic_lower_qty" : r_qty["ic_lower"],
        "ic_upper_qty" : r_qty["ic_upper"],
        "pesos"        : r_qty["pesos"],
    }
    if cfg["monto"]:
        r_mon = resultados[cfg["monto"]]
        entry.update({
            "real_monto"  : r_mon["real_parcial"],
            "proy_monto"  : r_mon["proyeccion_final"],
            "ic_lower_monto": r_mon["ic_lower"],
            "ic_upper_monto": r_mon["ic_upper"],
            "real_ticket" : ticket(r_mon["real_parcial"],     r_qty["real_parcial"]),
            "proy_ticket" : ticket(r_mon["proyeccion_final"], r_qty["proyeccion_final"]),
        })
    resumen_campañas[nombre] = entry

p = resumen_campañas["PRODUCTO_A"]["pesos"]
print(f"\n  Pesos → Ritmo actual: {p[0]:.0%} | Mes anterior: {p[1]:.0%} | Prophet: {p[2]:.0%}\n")

for nombre, e in resumen_campañas.items():
    print(f"  ── {nombre} {'─'*(52-len(nombre))}")
    hdr = f"  {'INDICADOR':<30} {'REAL PARCIAL':>14}  {'PROY. CIERRE':>14}  {'RANGO 80%':>24}"
    print(hdr)
    rango_qty = f"[{e['ic_lower_qty']:,.0f} – {e['ic_upper_qty']:,.0f}]"
    print(f"  {'Cantidad (Nro. Op.)':<30} {e['real_qty']:>14,.0f}  {e['proy_qty']:>14,.0f}  {rango_qty:>24}")
    if "proy_monto" in e:
        rango_mon = f"[{e['ic_lower_monto']:,.1f} – {e['ic_upper_monto']:,.1f}]"
        print(f"  {'Monto (MM$)':<30} {e['real_monto']:>14,.1f}  {e['proy_monto']:>14,.1f}  {rango_mon:>24}")
        print(f"  {'Ticket Promedio (MM$)':<30} {e['real_ticket']:>14,.2f}  {e['proy_ticket']:>14,.2f}")
    print()


# ─────────────────────────────────────────────────────────────
# 6B. PROYECCIÓN SEMANAL (lunes a viernes)
# ─────────────────────────────────────────────────────────────
DIAS_SEMANA = 5

ultima_fecha     = df["FECHA"].max()
lunes_actual     = ultima_fecha - pd.Timedelta(days=ultima_fecha.weekday())
lunes_anterior   = lunes_actual - pd.Timedelta(weeks=1)
viernes_anterior = lunes_anterior + pd.Timedelta(days=4)

datos_semana_actual   = df[df["FECHA"] >= lunes_actual]
datos_semana_anterior = df[(df["FECHA"] >= lunes_anterior) & (df["FECHA"] <= viernes_anterior)]
dias_semana_registrados = datos_semana_actual["FECHA"].nunique()
fraccion_semana = dias_semana_registrados / DIAS_SEMANA

# Promedio histórico de las últimas 4 semanas completas
_semanas_hist = []
for _w in range(1, 5):
    _lun = lunes_actual - pd.Timedelta(weeks=_w)
    _vie = _lun + pd.Timedelta(days=4)
    _s   = df[(df["FECHA"] >= _lun) & (df["FECHA"] <= _vie)]
    if not _s.empty:
        _semanas_hist.append(_s[COLUMNAS].sum())

prom_hist_semanas = (
    pd.concat(_semanas_hist, axis=1).mean(axis=1).to_dict()
    if _semanas_hist else {col: 0 for col in COLUMNAS}
)

def proyectar_semana(col):
    real_parcial  = float(datos_semana_actual[col].sum())
    prom_actual   = datos_semana_actual[col].mean()   if dias_semana_registrados > 0       else 0.0
    prom_anterior = datos_semana_anterior[col].mean() if not datos_semana_anterior.empty else prom_actual

    proy_ritmo = prom_actual   * DIAS_SEMANA
    proy_ant   = prom_anterior * DIAS_SEMANA
    proy_hist  = prom_hist_semanas.get(col, 0)

    w_ritmo = min(fraccion_semana * 1.1, 0.70)
    w_ant   = max(0.15, 0.25 - fraccion_semana * 0.15)
    w_hist  = max(0.10, 1 - w_ritmo - w_ant)
    total   = w_ritmo + w_ant + w_hist
    w_ritmo /= total; w_ant /= total; w_hist /= total

    proy_final_s = round(w_ritmo * proy_ritmo + w_ant * proy_ant + w_hist * proy_hist)
    proy_final_s = max(proy_final_s, int(real_parcial))

    return {
        "real_parcial"    : real_parcial,
        "proyeccion_final": proy_final_s,
        "pesos"           : (w_ritmo, w_ant, w_hist),
    }

resultados_semana = {col: proyectar_semana(col) for col in COLUMNAS}

resumen_semana = {}
for nombre_s, cfg_s in CAMPAÑAS.items():
    r_qty_s = resultados_semana[cfg_s["qty"]]
    entry_s = {
        "real_qty" : r_qty_s["real_parcial"],
        "proy_qty" : r_qty_s["proyeccion_final"],
        "pesos"    : r_qty_s["pesos"],
    }
    if cfg_s["monto"]:
        r_mon_s = resultados_semana[cfg_s["monto"]]
        entry_s.update({
            "real_monto" : r_mon_s["real_parcial"],
            "proy_monto" : r_mon_s["proyeccion_final"],
            "real_ticket": ticket(r_mon_s["real_parcial"],     r_qty_s["real_parcial"]),
            "proy_ticket": ticket(r_mon_s["proyeccion_final"], r_qty_s["proyeccion_final"]),
        })
    resumen_semana[nombre_s] = entry_s

print("=" * 70)
print(f"  PROYECCIÓN SEMANAL — semana del {lunes_actual.strftime('%d/%m/%Y')}")
print("=" * 70)
for nombre_s, e_s in resumen_semana.items():
    print(f"  ── {nombre_s} {'─'*(52-len(nombre_s))}")
    print(f"  {'Cantidad real':<30} {e_s['real_qty']:>14,.0f}  →  proy: {e_s['proy_qty']:>10,.0f}")
    if "proy_monto" in e_s:
        print(f"  {'Monto real (MM$)':<30} {e_s['real_monto']:>14,.1f}  →  proy: {e_s['proy_monto']:>10,.1f}")
    print()


# ─────────────────────────────────────────────────────────────
# 7. CONSTRUIR CUERPO HTML DEL CORREO
# ─────────────────────────────────────────────────────────────
def build_html_email(mes_actual, fraccion, dias_registrados, resumen_campañas,
                     resumen_semana, lunes_actual, fraccion_semana, dias_semana_registrados):
    fecha_gen = datetime.now().strftime("%d/%m/%Y %H:%M")

    def fmt_n(v): return f"{v:,.0f}"
    def fmt_m(v): return f"{v:,.1f}"
    def fmt_t(v): return f"{v:,.2f}"

    styles = """
    <style>
      body { font-family: Calibri, Arial, sans-serif; font-size: 13px; color: #222; }
      h2   { color: #2563eb; margin-bottom: 4px; }
      .meta { color: #555; font-size: 12px; margin-bottom: 20px; }
      .campana-title {
        background: #2563eb; color: white;
        padding: 6px 14px; font-size: 14px; font-weight: bold;
        border-radius: 4px 4px 0 0;
      }
      table { border-collapse: collapse; width: 100%; max-width: 640px; margin-bottom: 4px; }
      th {
        background: #1e293b; color: #fff;
        padding: 7px 12px; text-align: center; font-size: 12px;
      }
      th.lbl { text-align: left; }
      td { padding: 6px 12px; border: 1px solid #ddd; text-align: right; font-size: 13px; }
      td.label { text-align: left; font-weight: bold; background: #f5f5f5; color: #333; width: 200px; }
      tr.real td { background: #ffffff; }
      tr.proy td { background: #fff8e1; font-weight: bold; }
      .semana-title {
        background: #1a3a5c; color: white;
        padding: 6px 14px; font-size: 14px; font-weight: bold;
        border-radius: 4px 4px 0 0;
      }
      .footer { margin-top: 28px; font-size: 11px; color: #999; border-top: 1px solid #eee; padding-top: 8px; }
    </style>
    """

    html_blocks = []
    for nombre, e in resumen_campañas.items():
        tiene_monto = "proy_monto" in e
        if tiene_monto:
            header_row = '<th>CANTIDAD (Nro. Op.)</th><th>MONTO (MM$)</th><th>TICKET PROM. (MM$)</th>'
            real_vals  = f"<td>{fmt_n(e['real_qty'])}</td><td>{fmt_m(e['real_monto'])}</td><td>{fmt_t(e['real_ticket'])}</td>"
            proy_vals  = f"<td>{fmt_n(e['proy_qty'])}</td><td>{fmt_m(e['proy_monto'])}</td><td>{fmt_t(e['proy_ticket'])}</td>"
        else:
            header_row = '<th>CANTIDAD (Nro. Op.)</th>'
            real_vals  = f"<td>{fmt_n(e['real_qty'])}</td>"
            proy_vals  = f"<td>{fmt_n(e['proy_qty'])}</td>"

        ncols = 4 if tiene_monto else 2
        block = f"""
        <table style="margin-top:26px;">
          <tr>
            <td colspan="{ncols}" class="campana-title">{nombre}</td>
          </tr>
          <tr>
            <th class="lbl" style="width:200px;">INDICADOR</th>
            {header_row}
          </tr>
          <tr class="real">
            <td class="label">Real parcial (a la fecha)</td>{real_vals}
          </tr>
          <tr class="proy">
            <td class="label">Proyección de cierre</td>{proy_vals}
          </tr>
        </table>
        """
        html_blocks.append(block)

    semana_blocks = []
    for nombre_sw, e_sw in resumen_semana.items():
        tiene_monto_sw = "proy_monto" in e_sw
        if tiene_monto_sw:
            header_sw = '<th>CANTIDAD (Nro. Op.)</th><th>MONTO (MM$)</th><th>TICKET PROM. (MM$)</th>'
            real_sw   = f"<td>{fmt_n(e_sw['real_qty'])}</td><td>{fmt_m(e_sw['real_monto'])}</td><td>{fmt_t(e_sw['real_ticket'])}</td>"
            proy_sw   = f"<td>{fmt_n(e_sw['proy_qty'])}</td><td>{fmt_m(e_sw['proy_monto'])}</td><td>{fmt_t(e_sw['proy_ticket'])}</td>"
        else:
            header_sw = '<th>CANTIDAD (Nro. Op.)</th>'
            real_sw   = f"<td>{fmt_n(e_sw['real_qty'])}</td>"
            proy_sw   = f"<td>{fmt_n(e_sw['proy_qty'])}</td>"
        ncols_sw = 4 if tiene_monto_sw else 2
        semana_blocks.append(f"""
        <table style="margin-top:16px;">
          <tr><td colspan="{ncols_sw}" class="semana-title">{nombre_sw}</td></tr>
          <tr>
            <th class="lbl" style="width:200px;">INDICADOR</th>
            {header_sw}
          </tr>
          <tr class="real"><td class="label">Real parcial (semana)</td>{real_sw}</tr>
          <tr class="proy"><td class="label">Proyección de cierre</td>{proy_sw}</tr>
        </table>
        """)

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">{styles}</head>
<body>
  <h2>Proyección de Cierre Mensual &mdash; Campañas Comerciales</h2>
  <p class="meta">
    <b>Mes en curso:</b> {mes_actual} &nbsp;|&nbsp;
    <b>Avance:</b> {fraccion:.1%} ({dias_registrados} días registrados) &nbsp;|&nbsp;
    <b>Generado:</b> {fecha_gen}
  </p>
  {''.join(html_blocks)}
  <hr style="border:none;border-top:2px solid #ddd;margin:36px 0 20px 0;">
  <h3 style="color:#1a3a5c;margin-bottom:4px;">Proyección de Cierre Semanal</h3>
  <p class="meta">
    <b>Semana del:</b> {lunes_actual.strftime('%d/%m/%Y')} &nbsp;|&nbsp;
    <b>Avance:</b> {fraccion_semana:.1%} ({dias_semana_registrados} días registrados)
  </p>
  {''.join(semana_blocks)}
  <p class="footer">
    Proyección generada automáticamente combinando: ritmo del mes actual,
    referencia al mes anterior y modelo estacional Prophet (Meta).<br>
    El intervalo 80% corresponde al rango de confianza del modelo Prophet.
  </p>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
# 8. ENVIAR CORREO
# ─────────────────────────────────────────────────────────────
def send_email(html_body):
    from_header = f"{SMTP_FROM_NAME} <{SMTP_USER}>"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Proyección Cierre Campañas — {mes_actual}"
    msg["From"]    = from_header
    msg["To"]      = ", ".join(TO_LIST)
    if CC_LIST:
        msg["Cc"]  = ", ".join(CC_LIST)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    all_recipients = TO_LIST + CC_LIST
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, all_recipients, msg.as_string())
        print(f"✅  Correo enviado a: {', '.join(all_recipients)}")
    except smtplib.SMTPAuthenticationError:
        print("❌  Error de autenticación. Revisa SMTP_USER y SMTP_PASSWORD.")
    except smtplib.SMTPException as exc:
        print(f"❌  Error al enviar correo: {exc}")


html_email = build_html_email(
    mes_actual, fraccion, dias_registrados, resumen_campañas,
    resumen_semana, lunes_actual, fraccion_semana, dias_semana_registrados
)
send_email(html_email)

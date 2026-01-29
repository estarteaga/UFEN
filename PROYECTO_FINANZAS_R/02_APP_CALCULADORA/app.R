# app.R — Cotizador (Shiny) + Cotización imprimible (solo PDF por impresión)
# - Se eliminó el botón "Guardar HTML en carpeta"
# - Se eliminó la visual "Carpeta de guardado fijo"
# - Impresión multipágina: imprime SOLO la cotización en ventana limpia

OUTPUT_DIR <- "C:/Users/sclear/OneDrive/FEN CIENCIA DATOS/RSTUDIO/UFEN/PROYECTO_FINANZAS_R/02_APP_CALCULADORA"

# ---------------- Paquetes ----------------
pkgs <- c("shiny","bslib","tidyverse","glue","lubridate","readr","htmltools","base64enc")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}

library(shiny)
library(bslib)
library(tidyverse)
library(glue)
library(lubridate)
library(readr)
library(htmltools)
library(base64enc)

# ---------------- Carpetas ----------------
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUTPUT_DIR, "assets"), recursive = TRUE, showWarnings = FALSE)

# ---------------- README (requisito) ----------------
readme_path <- file.path(OUTPUT_DIR, "README_APP.txt")
if (!file.exists(readme_path)) {
  writeLines(
    c(
      "INSTRUCCIONES — Ejecutar la app Shiny",
      "",
      "Ejecuta en la consola (cambia la ruta según donde lo ejecutes):",
      "",
      "setwd(\"C:/Users/sclear/OneDrive/FEN CIENCIA DATOS/RSTUDIO/UFEN/PROYECTO_FINANZAS_R/02_APP_CALCULADORA\")",
      "shiny::runApp(launch.browser = TRUE)",
      "",
      "Luego: completa los inputs y usa la pestaña 'Cotización (Imprimible)'.",
      "Para PDF: botón 'Imprimir / Guardar PDF' -> guardar como PDF en el navegador.",
      "",
      "Opcional: agrega un logo en OUTPUT_DIR/assets/logo.png."
    ),
    con = readme_path
  )
}

# ---------------- Helpers ----------------
parse_num <- function(text, name = "") {
  t <- as.character(text) %>% trimws()
  if (t == "" || is.na(t)) stop(glue("Falta completar: {name}"))

  # Soporta "3.600.000" y "0,5"
  if (grepl(",", t) && !grepl("\\d+,\\d{3}$", t)) {
    t <- gsub("\\.", "", t)
    t <- gsub(",", ".", t)
  } else {
    t <- gsub("\\.", "", t)
    t <- gsub(" ", "", t)
  }

  val <- suppressWarnings(as.numeric(t))
  if (is.na(val)) stop(glue("'{name}' debe ser numérico (valor: {text})"))
  val
}

fmt_int <- function(x) formatC(round(as.numeric(x),0), format="f", digits=0, big.mark=".", decimal.mark=",")
fmt_num <- function(x) formatC(as.numeric(x), format="f", digits=2, big.mark=".", decimal.mark=",")
fmt_clp <- function(x) paste0("CLP ", fmt_int(x))

# ---------------- Factores predefinidos ----------------
factor_ejes <- c("4x2"=1.01, "6x2"=1.02, "4x4"=1.04, "6x4"=1.05, "8x4"=1.10)
factor_zona <- c("Zona 01 Norte"=1.05, "Zona 02 Centro"=1.02, "Zona 03 Sur"=1.02)
factor_operacion <- c("Larga distancia"=1.00, "Forestal"=1.15, "Minería"=1.40)

# ---------------- Cálculo principal ----------------
calcular_cotizacion <- function(
  client_name, start_date,
  intervalo_km, costo_ciclo,
  usd_mensual_pct, uf_mensual_pct,
  peso_usd_pct,
  uso_mensual_km, meses_contrato,
  var_ejes, var_zona, var_operacion
) {

  intervalo_km <- as.integer(parse_num(intervalo_km, "Intervalo MTTO (km)"))
  costo_ciclo  <- parse_num(costo_ciclo, "Costo por ciclo (CLP)")
  usd_pct      <- parse_num(usd_mensual_pct, "USD mensual (%)") / 100
  uf_pct       <- parse_num(uf_mensual_pct, "UF mensual (%)") / 100
  uso_mensual  <- as.integer(parse_num(uso_mensual_km, "Uso mensual (km/mes)"))
  meses        <- as.integer(parse_num(meses_contrato, "Plazo (meses)"))

  if (intervalo_km <= 0 || costo_ciclo <= 0 || uso_mensual <= 0 || meses <= 0) {
    stop("Intervalo, costo, uso mensual y meses deben ser > 0.")
  }
  if (usd_pct <= -1 || uf_pct <= -1) stop("USD/UF mensual no puede ser <= -100%.")

  f_ejes <- factor_ejes[[var_ejes]]
  f_zona <- factor_zona[[var_zona]]
  f_op   <- factor_operacion[[var_operacion]]
  factor_total <- f_ejes * f_zona * f_op

  peso_usd <- as.numeric(peso_usd_pct)/100
  peso_uf  <- 1 - peso_usd

  km_ciclo <- intervalo_km * 4
  tarifa_base <- costo_ciclo / km_ciclo
  tarifa_indexada <- tarifa_base * factor_total

  idx_usd <- 1.0
  idx_uf  <- 1.0
  total_contrato <- 0

  fechas <- seq.Date(from = as.Date(start_date), by = "month", length.out = meses)

  detalle <- vector("list", meses)
  for (m in seq_len(meses)) {
    idx_usd <- idx_usd * (1 + usd_pct)
    idx_uf  <- idx_uf  * (1 + uf_pct)
    idx_blend <- peso_usd * idx_usd + peso_uf * idx_uf

    tarifa_mes <- tarifa_indexada * idx_blend
    pago_mes <- tarifa_mes * uso_mensual
    total_contrato <- total_contrato + pago_mes

    detalle[[m]] <- tibble(
      mes = m,
      fecha = fechas[m],
      tarifa_mes = tarifa_mes,
      km_mes = uso_mensual,
      pago_mes = pago_mes,
      usd_idx = idx_usd,
      uf_idx = idx_uf,
      blend_idx = idx_blend
    )
  }

  df_detalle <- bind_rows(detalle)

  metrics <- tibble(
    cliente = client_name,
    fecha_inicio = as.Date(start_date),
    ejes = var_ejes,
    zona = var_zona,
    operacion = var_operacion,
    intervalo_km = intervalo_km,
    costo_ciclo = costo_ciclo,
    uso_mensual_km = uso_mensual,
    meses_contrato = meses,
    usd_mensual_pct = usd_pct,
    uf_mensual_pct = uf_pct,
    peso_usd = peso_usd,
    peso_uf = peso_uf,
    factor_total = factor_total,
    tarifa_base = tarifa_base,
    tarifa_indexada = tarifa_indexada,
    pago_mes1 = df_detalle$pago_mes[1],
    total_contrato = total_contrato
  )

  list(detalle = df_detalle, metrics = metrics)
}

# ---------------- CSS (pantalla + impresión) ----------------
REPORT_CSS <- "
:root{
  --brand:#003A5D;
  --muted:#6c757d;
  --border:#e6e6e6;
}

html, body { height: 100%; width: 100%; margin: 0; overflow: hidden; }
.bslib-page-fill { height: 100vh; }
.bslib-sidebar { overflow-y: auto; max-height: 100vh; }

.report-wrap{
  max-width: 1000px;
  margin: 0 auto;
  padding: 16px 16px 40px 16px;
}
.banner{
  background: var(--brand);
  color: #fff;
  border-radius: 16px;
  padding: 18px 20px;
  margin-bottom: 14px;
}
.banner h1{ margin:0; font-size: 1.6rem; }
.banner p{ margin: 6px 0 0 0; opacity: .9; }

.meta-row{
  display:flex;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
  margin: 10px 0 16px 0;
}
.meta-box{
  flex: 1 1 260px;
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 14px;
  background:#fff;
}

.kpi-row{
  display:flex;
  flex-wrap:wrap;
  gap: 12px;
  margin: 10px 0 16px 0;
}
.kpi{
  flex: 1 1 220px;
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
  background:#fff;
  box-shadow: 0 1px 2px rgba(0,0,0,.06);
}
.kpi .label{ color: var(--muted); font-size:.85rem; margin:0; }
.kpi .value{ color: var(--brand); font-size:1.6rem; font-weight:700; margin:0; }

.section-title{
  margin-top: 18px;
  margin-bottom: 8px;
  font-size: 1.1rem;
  color: #111;
}

.tbl{
  width: 100%;
  border-collapse: collapse;
  margin: 6px 0 14px 0;
  font-size: .95rem;
}
.tbl th, .tbl td{
  border: 1px solid var(--border);
  padding: 8px 10px;
  vertical-align: top;
}
.tbl th{
  background: #f6f7f9;
  text-align: left;
}

.small-note{ color: var(--muted); font-size: .85rem; margin-top: 12px; }

/* --- PRINT (multipágina) --- */
@media print {
  @page { size: A4; margin: 12mm; }
  html, body { overflow: visible !important; height: auto !important; }
  body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }

  .banner, .meta-box, .kpi, img { break-inside: avoid; page-break-inside: avoid; }
  table, tr, td, th { break-inside: avoid; page-break-inside: avoid; }
  .tbl { break-inside: auto; page-break-inside: auto; }
  .tbl tr { break-inside: avoid; page-break-inside: avoid; }
}
"

plot_to_uri <- function(p, width=1200, height=420, res=150) {
  tf <- tempfile(fileext = ".png")
  png(tf, width = width, height = height, res = res)
  print(p)
  dev.off()
  base64enc::dataURI(file = tf, mime = "image/png")
}

build_report_tag <- function(res, inputs) {
  m <- res$metrics
  d <- res$detalle

  # Logo opcional
  logo_path <- file.path(OUTPUT_DIR, "assets", "logo.png")
  logo_uri <- if (file.exists(logo_path)) base64enc::dataURI(file=logo_path, mime="image/png") else NULL

  # Gráfico embebido
  p <- ggplot(d, aes(x = mes, y = pago_mes)) +
    geom_col(alpha = 0.9) +
    labs(x = "Mes", y = "Pago (CLP)") +
    theme_minimal()
  plot_uri <- plot_to_uri(p)

  # Tabla parámetros
  params_tbl <- tibble(
    Parámetro = c("Ejes","Zona","Operación","Intervalo MTTO (km)","Costo ciclo (CLP)","Uso mensual (km/mes)","Plazo (meses)",
                  "USD mensual (%)","UF mensual (%)","Peso USD (%)","Peso UF (%)","Factor total"),
    Valor = c(
      inputs$var_ejes,
      inputs$var_zona,
      inputs$var_operacion,
      inputs$intervalo_km,
      inputs$costo_ciclo,
      inputs$uso_mensual_km,
      inputs$meses_contrato,
      inputs$usd_mensual_pct,
      inputs$uf_mensual_pct,
      inputs$peso_usd_pct,
      100 - as.numeric(inputs$peso_usd_pct),
      fmt_num(m$factor_total[1])
    )
  )

  detalle12 <- d %>%
    select(mes, fecha, tarifa_mes, km_mes, pago_mes) %>%
    head(12) %>%
    mutate(
      fecha = format(as.Date(fecha), "%d-%m-%Y"),
      tarifa_mes = fmt_num(tarifa_mes),
      pago_mes = fmt_clp(pago_mes)
    )

  div(class="report-wrap",
    div(class="banner",
      tags$h1("Cotización — Contrato de Mantención"),
      tags$p("Tarifa $/km y proyección de costos")
    ),

    div(class="meta-row",
      div(class="meta-box",
        if (!is.null(logo_uri)) tags$img(src = logo_uri, style="max-height:55px; margin-bottom:8px; display:block;"),
        tags$b("Cliente:"), tags$div(inputs$client_name),
        tags$div(tags$small(class="small-note", paste0("Emisión: ", format(Sys.time(), "%d-%m-%Y %H:%M"))))
      ),
      div(class="meta-box",
        tags$b("Inicio contrato:"), tags$div(format(as.Date(inputs$start_date), "%d-%m-%Y")),
        tags$div(tags$small(class="small-note", paste0("Plazo: ", inputs$meses_contrato, " meses")))
      )
    ),

    div(class="kpi-row",
      div(class="kpi", tags$p(class="label","Tarifa base (CLP/km)"), tags$p(class="value", fmt_num(m$tarifa_base[1]))),
      div(class="kpi", tags$p(class="label","Tarifa indexada (CLP/km)"), tags$p(class="value", fmt_num(m$tarifa_indexada[1]))),
      div(class="kpi", tags$p(class="label","Pago mes 1 (CLP)"), tags$p(class="value", fmt_clp(m$pago_mes1[1]))),
      div(class="kpi", tags$p(class="label","Total contrato (CLP)"), tags$p(class="value", fmt_clp(m$total_contrato[1])))
    ),

    tags$h2(class="section-title","Proyección de pagos"),
    tags$img(src = plot_uri, style="width:100%; border:1px solid #e6e6e6; border-radius:12px;"),

    tags$h2(class="section-title","Parámetros seleccionados"),
    tags$table(class="tbl",
      tags$thead(tags$tr(tags$th("Parámetro"), tags$th("Valor"))),
      tags$tbody(
        lapply(seq_len(nrow(params_tbl)), function(i) {
          tags$tr(
            tags$td(params_tbl$Parámetro[i]),
            tags$td(as.character(params_tbl$Valor[i]))
          )
        })
      )
    ),

    tags$h2(class="section-title","Detalle (primeros 12 meses)"),
    tags$table(class="tbl",
      tags$thead(tags$tr(tags$th("Mes"), tags$th("Fecha"), tags$th("Tarifa (CLP/km)"), tags$th("Km mes"), tags$th("Pago mes"))),
      tags$tbody(
        lapply(seq_len(nrow(detalle12)), function(i) {
          tags$tr(
            tags$td(detalle12$mes[i]),
            tags$td(detalle12$fecha[i]),
            tags$td(detalle12$tarifa_mes[i]),
            tags$td(detalle12$km_mes[i]),
            tags$td(detalle12$pago_mes[i])
          )
        })
      )
    ),

    tags$div(class="small-note",
      "Notas: estimación basada en supuestos seleccionados por el usuario y factores predefinidos. Documento demostrativo."
    )
  )
}

# ---------------- UI ----------------
ui <- page_fillable(
  title = "Cotizador — Contrato de Mantención (MVP)",
  theme = bs_theme(version = 5),

  tags$style(id = "report_css", HTML(REPORT_CSS)),

  # JS: imprime SOLO el HTML de #report_print en una ventana nueva limpia
  tags$script(HTML("
    function printReportOnly() {
      var report = document.getElementById('report_print');
      if (!report) { window.print(); return; }

      var cssEl = document.getElementById('report_css');
      var css = cssEl ? cssEl.innerHTML : '';

      var w = window.open('', '_blank');
      if (!w) {
        alert('El navegador bloqueó la ventana de impresión (pop-up). Permite pop-ups para esta página.');
        return;
      }

      w.document.open();
      w.document.write('<!doctype html><html><head><meta charset=\"utf-8\">');
      w.document.write('<title>Cotización</title>');
      w.document.write('<style>' + css + '</style>');
      w.document.write('</head><body>');
      w.document.write(report.innerHTML);
      w.document.write('</body></html>');
      w.document.close();

      w.onafterprint = function() { w.close(); };

      w.focus();
      setTimeout(function(){
        w.print();
      }, 300);
    }
  ")),

  layout_sidebar(
    sidebar = sidebar(
      width = 380,
      title = "Inputs",
      accordion(
        accordion_panel(
          "Cliente y fechas",
          textInput("client_name", "Nombre cliente", value = "TRANSPORTES UFEN S.A."),
          dateInput("start_date", "Fecha inicio", value = Sys.Date(), format = "yyyy-mm-dd")
        ),
        accordion_panel(
          "Selecciones predefinidas",
          selectInput("var_ejes", "Configuración de ejes", choices = names(factor_ejes), selected = "6x2"),
          selectInput("var_zona", "Zona", choices = names(factor_zona), selected = "Zona 02 Centro"),
          selectInput("var_operacion", "Operación", choices = names(factor_operacion), selected = "Larga distancia")
        ),
        accordion_panel(
          "Supuestos de mantención",
          textInput("intervalo_km", "Intervalo MTTO (km)", value = "30000"),
          textInput("costo_ciclo", "Costo por ciclo (CLP)", value = "3.600.000"),
          textInput("uso_mensual_km", "Uso mensual (km/mes)", value = "10000"),
          textInput("meses_contrato", "Plazo (meses)", value = "36")
        ),
        accordion_panel(
          "Indexación (mensual)",
          textInput("usd_mensual_pct", "USD mensual (%)", value = "0,5"),
          textInput("uf_mensual_pct", "UF mensual (%)", value = "0,2"),
          sliderInput("peso_usd_pct", "Peso USD (%)", min = 0, max = 100, value = 80, step = 5),
          uiOutput("peso_uf_txt"),
          div(style="font-size:12px; color:#666;",
              "Puedes escribir con coma (0,5) o con puntos de miles (3.600.000).")
        ),
        open = 1
      ),
      hr(),
      actionButton("calc_btn", "Calcular / Actualizar"),
      actionButton("print_btn", "Imprimir / Guardar PDF", onclick = "printReportOnly();"),
      hr(),
      div(style="font-size:12px; color:#666;",
          "Impresión recomendada: Chrome/Edge. En el diálogo de impresión activa 'Gráficos de fondo'.")
    ),
    fill = TRUE,
    card(
      full_screen = TRUE,
      navset_card_tab(
        nav_panel("Resumen", tableOutput("metrics_table")),
        nav_panel("Gráfico", plotOutput("plot_pago", height = "320px")),
        nav_panel("Detalle (12 meses)", tableOutput("detalle_head")),
        nav_panel("Cotización (Imprimible)",
          div(id = "report_print", uiOutput("report_ui"))
        )
      )
    )
  )
)

# ---------------- Server ----------------
server <- function(input, output, session) {

  output$peso_uf_txt <- renderUI({
    tags$div(strong(paste0("Peso UF (%): ", 100 - input$peso_usd_pct)))
  })

  calc_res <- eventReactive(input$calc_btn, {
    calcular_cotizacion(
      client_name = input$client_name,
      start_date = input$start_date,
      intervalo_km = input$intervalo_km,
      costo_ciclo = input$costo_ciclo,
      usd_mensual_pct = input$usd_mensual_pct,
      uf_mensual_pct = input$uf_mensual_pct,
      peso_usd_pct = input$peso_usd_pct,
      uso_mensual_km = input$uso_mensual_km,
      meses_contrato = input$meses_contrato,
      var_ejes = input$var_ejes,
      var_zona = input$var_zona,
      var_operacion = input$var_operacion
    )
  }, ignoreInit = FALSE)

  output$metrics_table <- renderTable({
    res <- calc_res()
    res$metrics %>%
      transmute(
        Cliente = cliente,
        `Fecha inicio` = fecha_inicio,
        Ejes = ejes,
        Zona = zona,
        Operación = operacion,
        `Tarifa base (CLP/km)` = round(tarifa_base, 2),
        `Tarifa indexada (CLP/km)` = round(tarifa_indexada, 2),
        `Pago mes 1 (CLP)` = round(pago_mes1, 0),
        `Total contrato (CLP)` = round(total_contrato, 0)
      )
  })

  output$detalle_head <- renderTable({
    res <- calc_res()
    res$detalle %>%
      select(mes, fecha, tarifa_mes, km_mes, pago_mes) %>%
      head(12) %>%
      mutate(
        tarifa_mes = round(tarifa_mes, 2),
        pago_mes = round(pago_mes, 0)
      )
  })

  output$plot_pago <- renderPlot({
    res <- calc_res()
    ggplot(res$detalle, aes(x = mes, y = pago_mes)) +
      geom_col(alpha = 0.9) +
      labs(x = "Mes", y = "Pago (CLP)") +
      theme_minimal()
  })

  output$report_ui <- renderUI({
    res <- calc_res()
    build_report_tag(res, reactiveValuesToList(input))
  })
}

shinyApp(ui, server)

# ============================================================
# RAG over an Excel in R using OpenAI (Embeddings + Chat)
# - Read Excel -> turn rows into text chunks -> embed -> retrieve -> answer
# - No vector DB needed (in-memory). Save as .rds for reuse.
# ============================================================

install.packages(c("httr", "jsonlite", "readxl", "stringr"), quiet = TRUE)

library(httr)
library(jsonlite)
library(readxl)
library(stringr)

# ----------------------------
# Config
# ----------------------------
OPENAI_EMBED_MODEL <- "text-embedding-3-small"
OPENAI_CHAT_MODEL  <- "gpt-4.1-mini"

XLSX_PATH <- "data_20230107.xlsx"    # <- change this
SHEET <- 1                     # sheet name or index
ROWS_PER_CHUNK <- 40           # tune: 20–100 depending on row width
TOP_K <- 4

# ----------------------------
# OpenAI helpers
# ----------------------------
assert_api_key <- function() {
  key <- Sys.getenv("OPENAI_API_KEY")
  if (identical(key, "") || nchar(key) < 20) {
    stop(
      "OPENAI_API_KEY missing/invalid.\n",
      "Set it with: Sys.setenv(OPENAI_API_KEY='sk-...')\n",
      "Or put it in ~/.Renviron and restart R."
    )
  }
  key
}

as_numeric_vec <- function(x) as.numeric(unlist(x, use.names = FALSE))

openai_embed <- function(text, model = OPENAI_EMBED_MODEL) {
  key <- assert_api_key()
  
  res <- POST(
    "https://api.openai.com/v1/embeddings",
    add_headers(
      Authorization = paste("Bearer", key),
      "Content-Type" = "application/json"
    ),
    body = list(model = model, input = text),
    encode = "json"
  )
  
  txt <- content(res, as = "text", encoding = "UTF-8")
  if (http_error(res)) stop("Embeddings failed (HTTP ", status_code(res), ")\n\n", txt)
  
  out <- tryCatch(fromJSON(txt, simplifyVector = FALSE),
                  error = function(e) stop("Bad JSON from API.\n\nBody:\n", txt))
  
  if (!is.null(out$error)) stop("OpenAI embeddings error: ", out$error$message)
  
  emb <- out$data[[1]]$embedding
  if (is.null(emb)) stop("Unexpected embeddings response format.\n\nBody:\n", txt)
  
  as_numeric_vec(emb)
}

openai_chat <- function(prompt, model = OPENAI_CHAT_MODEL) {
  key <- assert_api_key()
  
  res <- POST(
    "https://api.openai.com/v1/chat/completions",
    add_headers(
      Authorization = paste("Bearer", key),
      "Content-Type" = "application/json"
    ),
    body = list(
      model = model,
      messages = list(list(role = "user", content = prompt))
    ),
    encode = "json"
  )
  
  txt <- content(res, as = "text", encoding = "UTF-8")
  if (http_error(res)) stop("Chat failed (HTTP ", status_code(res), ")\n\n", txt)
  
  out <- tryCatch(fromJSON(txt, simplifyVector = FALSE),
                  error = function(e) stop("Bad JSON from API.\n\nBody:\n", txt))
  
  if (!is.null(out$error)) stop("OpenAI chat error: ", out$error$message)
  
  out$choices[[1]]$message$content
}

# ----------------------------
# Excel -> text chunks
# ----------------------------
value_to_text <- function(x) {
  if (is.null(x) || length(x) == 0) return("")
  if (inherits(x, "POSIXct") || inherits(x, "Date")) return(as.character(x))
  if (is.na(x)) return("NA")
  as.character(x)
}

row_to_sentence <- function(row, colnames) {
  parts <- mapply(function(k, v) paste0(k, "=", value_to_text(v)),
                  colnames, row, SIMPLIFY = TRUE, USE.NAMES = FALSE)
  paste(parts, collapse = " | ")
}

excel_to_chunks <- function(path, sheet = 1, rows_per_chunk = 40) {
  df <- read_excel(path, sheet = sheet)
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  
  # ensure column names are present
  if (any(colnames(df) == "")) {
    colnames(df)[colnames(df) == ""] <- paste0("V", which(colnames(df) == ""))
  }
  
  n <- nrow(df)
  if (n == 0) stop("Excel sheet has 0 rows.")
  
  sentences <- vapply(seq_len(n), function(i) {
    row_to_sentence(df[i, , drop = FALSE], colnames(df))
  }, character(1))
  
  # group rows into chunks
  starts <- seq(1, n, by = rows_per_chunk)
  
  chunks <- vapply(starts, function(s) {
    e <- min(s + rows_per_chunk - 1, n)
    paste0(
      "Rows ", s, "-", e, ":\n",
      paste(sentences[s:e], collapse = "\n")
    )
  }, character(1))
  
  meta <- data.frame(
    chunk_id = seq_along(chunks),
    row_start = starts,
    row_end = pmin(starts + rows_per_chunk - 1, n)
  )
  
  list(df = df, chunks = chunks, meta = meta)
}

# ----------------------------
# Retrieval
# ----------------------------
cosine_sim <- function(a, b) {
  a <- as_numeric_vec(a)
  b <- as_numeric_vec(b)
  if (length(a) != length(b)) stop("Embedding length mismatch: ", length(a), " vs ", length(b))
  den <- sqrt(sum(a^2)) * sqrt(sum(b^2))
  if (den == 0) return(0)
  sum(a * b) / den
}

build_vector_store <- function(chunks, meta) {
  embs <- lapply(chunks, openai_embed)
  list(chunks = chunks, embs = embs, meta = meta)
}

retrieve_top_k <- function(store, query, k = 4) {
  q_emb <- openai_embed(query)
  scores <- sapply(store$embs, cosine_sim, b = q_emb)
  
  idx <- order(scores, decreasing = TRUE)[seq_len(min(k, length(scores)))]
  list(
    idx = idx,
    scores = scores[idx],
    chunks = store$chunks[idx],
    meta = store$meta[idx, , drop = FALSE]
  )
}

rag_answer <- function(store, question, k = 4) {
  ret <- retrieve_top_k(store, question, k = k)
  
  # include row ranges so the model can cite them
  context <- paste0(
    "[Rows ", ret$meta$row_start, "-", ret$meta$row_end, "]\n",
    ret$chunks,
    collapse = "\n---\n"
  )
  
  prompt <- paste0(
    "You are a helpful assistant.\n",
    "Answer using ONLY the context from the spreadsheet.\n",
    "If the context is insufficient, say you don't know.\n",
    "When you use facts, cite the row range like (Rows 10-20).\n\n",
    "CONTEXT:\n", context, "\n\n",
    "QUESTION: ", question, "\n",
    "ANSWER:"
  )
  
  list(
    answer = openai_chat(prompt),
    retrieved = ret
  )
}

# ============================================================
# Run: Excel -> store -> ask
# ============================================================

# 0) Set your key (or put in ~/.Renviron and restart)
# Sys.setenv(OPENAI_API_KEY = "sk-...")

excel_data <- excel_to_chunks(XLSX_PATH, sheet = "Plan de Operaciones", rows_per_chunk = ROWS_PER_CHUNK)

cat("Rows:", nrow(excel_data$df), "\n")
cat("Chunks created:", length(excel_data$chunks), "\n")

store <- build_vector_store(excel_data$chunks, excel_data$meta)

# Optional: save store (avoid re-embedding every time)
saveRDS(store, "excel_store.rds")

question <- "What is the value of Grua horquilla."
result <- rag_answer(store, question, k = TOP_K)

cat("\n--- Answer ---\n")
cat(result$answer, "\n")

cat("\n--- Retrieved row ranges ---\n")
print(result$retrieved$meta)

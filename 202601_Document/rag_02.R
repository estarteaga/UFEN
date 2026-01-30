# ============================================================
# RAG over a PDF in R using OpenAI (Embeddings + Chat)
# - Extract PDF text -> clean -> chunk -> embed -> retrieve -> answer
# - No vector DB needed (in-memory). Save as .rds for reuse.
# ============================================================

# Packages
install.packages(c("httr", "jsonlite", "pdftools", "stringr"), quiet = TRUE)

library(httr)
library(jsonlite)
library(pdftools)
library(stringr)

# ----------------------------
# Config
# ----------------------------
OPENAI_EMBED_MODEL <- "text-embedding-3-small"
OPENAI_CHAT_MODEL  <- "gpt-4.1-mini"

PDF_PATH <- "Results-SPA-1Q2025.pdf"   # <- change this
CHUNK_WORDS <- 500              # chunk size in words (adjust 300-800)
TOP_K <- 4                      # how many chunks to retrieve per question

# ----------------------------
# OpenAI helpers
# ----------------------------
assert_api_key <- function() {
  key <- Sys.getenv("OPENAI_API_KEY")
  if (identical(key, "") || nchar(key) < 20) {
    stop(
      "OPENAI_API_KEY missing/invalid.\n",
      "Set it with: Sys.setenv(OPENAI_API_KEY = 'sk-...')\n",
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
# Text processing: PDF -> clean -> chunk
# ----------------------------
clean_text <- function(x) {
  x |>
    str_replace_all("-\n", "") |>
    str_replace_all("\n", " ") |>
    str_squish()
}

chunk_words <- function(text, chunk_words = 500) {
  words <- str_split(text, "\\s+")[[1]]
  if (length(words) == 0) return(character(0))
  
  starts <- seq(1, length(words), by = chunk_words)
  vapply(starts, function(i) {
    paste(words[i:min(i + chunk_words - 1, length(words))], collapse = " ")
  }, character(1))
}

pdf_to_chunks <- function(pdf_path, chunk_words = 500) {
  pages_raw <- pdf_text(pdf_path)
  pages_clean <- vapply(pages_raw, clean_text, character(1))
  
  chunks <- character(0)
  meta_page <- integer(0)
  
  for (p in seq_along(pages_clean)) {
    page_chunks <- chunk_words(pages_clean[[p]], chunk_words = chunk_words)
    if (length(page_chunks) > 0) {
      chunks <- c(chunks, page_chunks)
      meta_page <- c(meta_page, rep.int(p, length(page_chunks)))
    }
  }
  
  list(chunks = chunks, page = meta_page)
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

build_vector_store <- function(chunks, page) {
  embs <- lapply(chunks, openai_embed)
  list(chunks = chunks, embs = embs, page = page)
}

retrieve_top_k <- function(store, query, k = 4) {
  q_emb <- openai_embed(query)
  scores <- sapply(store$embs, cosine_sim, b = q_emb)
  
  idx <- order(scores, decreasing = TRUE)[seq_len(min(k, length(scores)))]
  list(
    idx = idx,
    scores = scores[idx],
    chunks = store$chunks[idx],
    page = store$page[idx]
  )
}

# ----------------------------
# RAG answer with page citations
# ----------------------------
rag_answer <- function(store, question, k = 4) {
  ret <- retrieve_top_k(store, question, k = k)
  
  # Include page numbers in context so the model can cite them
  context <- paste0(
    "[Page ", ret$page, "] ", ret$chunks,
    collapse = "\n---\n"
  )
  
  prompt <- paste0(
    "You are a helpful assistant.\n",
    "Answer using ONLY the context.\n",
    "If the context is insufficient, say you don't know.\n",
    "When you use a fact, cite the page like (Page X).\n\n",
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
# Run: PDF -> store -> ask
# ============================================================

# 0) Set your API key (or put it in ~/.Renviron and restart)
# Sys.setenv(OPENAI_API_KEY = "sk-...")

# 1) Extract and chunk PDF
pdf_data <- pdf_to_chunks(PDF_PATH, chunk_words = CHUNK_WORDS)
chunks <- pdf_data$chunks
page   <- pdf_data$page

cat("Chunks created:", length(chunks), "\n")

# 2) Build store (this calls OpenAI embeddings for each chunk)
store <- build_vector_store(chunks, page)

# Optional: save so you don't re-embed every time
saveRDS(store, "pdf_store.rds")

# 3) Ask a question
question <- "Describe the aircraft fleet in latam"
result <- rag_answer(store, question, k = TOP_K)

cat("\n--- Answer ---\n")
cat(result$answer, "\n")

cat("\n--- Retrieved pages ---\n")
print(unique(result$retrieved$page))


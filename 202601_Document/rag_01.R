# ============================================================
# RAG in R using OpenAI (Embeddings + Chat)
# - No vector DB needed (in-memory)
# - Robust error handling + numeric-safe embeddings
# ============================================================

library(httr)
library(jsonlite)

# ----------------------------
# Config
# ----------------------------
OPENAI_EMBED_MODEL <- "text-embedding-3-small"  # embeddings model :contentReference[oaicite:4]{index=4}
OPENAI_CHAT_MODEL  <- "gpt-4.1-mini"            # chat model :contentReference[oaicite:5]{index=5}

# ----------------------------
# Utilities
# ----------------------------
assert_api_key <- function() {
  key <- Sys.getenv("OPENAI_API_KEY")
  if (identical(key, "") || nchar(key) < 20) {
    stop(
      "OPENAI_API_KEY is missing/invalid in this R session.\n",
      "Set it with: Sys.setenv(OPENAI_API_KEY = 'sk-...')\n",
      "Then re-run."
    )
  }
  key
}

as_numeric_vec <- function(x) as.numeric(unlist(x, use.names = FALSE))

stop_with_body <- function(res, prefix = "OpenAI API error") {
  txt <- tryCatch(content(res, as = "text", encoding = "UTF-8"), error = function(e) "")
  stop(prefix, " (HTTP ", status_code(res), ")\n\nBody:\n", txt)
}

cosine_sim <- function(a, b) {
  a <- as_numeric_vec(a)
  b <- as_numeric_vec(b)
  
  if (length(a) != length(b)) {
    stop("Embedding length mismatch: ", length(a), " vs ", length(b))
  }
  
  den <- sqrt(sum(a^2)) * sqrt(sum(b^2))
  if (den == 0) return(0)
  sum(a * b) / den
}

# ----------------------------
# OpenAI: embeddings
# ----------------------------
openai_embed <- function(text, model = OPENAI_EMBED_MODEL) {
  key <- assert_api_key()
  
  res <- POST(
    "https://api.openai.com/v1/embeddings",
    add_headers(
      Authorization = paste("Bearer", key),
      "Content-Type" = "application/json"
    ),
    body = list(
      model = model,
      input = text
    ),
    encode = "json"
  )
  
  txt <- content(res, as = "text", encoding = "UTF-8")
  
  if (http_error(res)) {
    stop("Embeddings call failed.\n\nHTTP ", status_code(res), "\n\n", txt)
  }
  
  out <- tryCatch(
    fromJSON(txt, simplifyVector = FALSE),
    error = function(e) stop("Could not parse JSON.\n\nRaw body:\n", txt)
  )
  
  if (!is.null(out$error)) stop("OpenAI embeddings error: ", out$error$message)
  
  # Expected: out$data[[1]]$embedding
  emb <- out$data[[1]]$embedding
  if (is.null(emb)) stop("Unexpected embeddings response format.\n\nRaw body:\n", txt)
  
  as_numeric_vec(emb)
}

# ----------------------------
# OpenAI: chat completions
# ----------------------------
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
      messages = list(
        list(role = "user", content = prompt)
      )
    ),
    encode = "json"
  )
  
  txt <- content(res, as = "text", encoding = "UTF-8")
  
  if (http_error(res)) {
    stop("Chat call failed.\n\nHTTP ", status_code(res), "\n\n", txt)
  }
  
  out <- tryCatch(
    fromJSON(txt, simplifyVector = FALSE),
    error = function(e) stop("Could not parse JSON.\n\nRaw body:\n", txt)
  )
  
  if (!is.null(out$error)) stop("OpenAI chat error: ", out$error$message)
  
  out$choices[[1]]$message$content
}

# ----------------------------
# RAG functions
# ----------------------------
build_vector_store <- function(docs) {
  embs <- lapply(docs, openai_embed)
  list(docs = docs, embs = embs)
}

retrieve_top_k <- function(store, query, k = 3) {
  q_emb <- openai_embed(query)
  scores <- sapply(store$embs, cosine_sim, b = q_emb)
  idx <- order(scores, decreasing = TRUE)[seq_len(min(k, length(scores)))]
  
  list(
    idx = idx,
    scores = scores[idx],
    top_docs = store$docs[idx]
  )
}

rag_answer <- function(store, question, k = 3) {
  ret <- retrieve_top_k(store, question, k = k)
  context <- paste(ret$top_docs, collapse = "\n---\n")
  
  prompt <- paste0(
    "You are a helpful assistant. Answer using ONLY the context.\n",
    "If the context is insufficient, say you don't know.\n\n",
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
# Example run
# ============================================================

# 1) Set your key (or put it in ~/.Renviron and restart R)
# Sys.setenv(OPENAI_API_KEY = "sk-...")

docs <- c(
  "RAG stands for Retrieval-Augmented Generation. It retrieves relevant text and uses it as context for an LLM.",
  "Embeddings map text into vectors so we can search by semantic similarity like cosine similarity.",
  "A vector store helps store embeddings and perform similarity search efficiently at scale, but you can start in-memory."
)

store <- build_vector_store(docs)

result <- rag_answer(store, "what is an airplane", k = 2)

cat("\n--- Retrieved docs ---\n")
print(result$retrieved$top_docs)

cat("\n--- Answer ---\n")
cat(result$answer, "\n")

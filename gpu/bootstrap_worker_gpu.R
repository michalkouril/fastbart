# One replicate of the GPU pipeline, one phase.   Usage: Rscript bootstrap_worker_gpu.R <rep> <prep|finish>
#   prep:   resample (rep > 0; as bootstrap_worker_bart_week6.R), run the estimator up to the titer fits, export the six titer
#           training sets and save the session in the replicate's scratch folder.
#   finish: reload the session, read the GPU fits, run the rest of the estimator and write rep_XXX.csv.
args <- commandArgs(trailingOnly = TRUE)
.w_rep <- as.integer(args[[1]]); .w_phase <- args[[2]]
BASE <- Sys.getenv("BOOT_BASE"); DATA_DIR <- Sys.getenv("BOOT_DATA_DIR"); OUTDIR <- Sys.getenv("BOOT_OUTDIR")
.w_scr <- file.path(Sys.getenv("BOOT_SCRATCH"), sprintf("rep_%03d", .w_rep))
.w_res <- file.path(OUTDIR, sprintf("rep_%03d.csv", .w_rep))
fail <- function(e) {
  res <- data.frame(interval_week = NA, Estimand = NA, Outcome = NA, Post_mean = NA, rep = .w_rep, error = paste(.w_phase, conditionMessage(e)))
  write.csv(res, paste0(.w_res, ".tmp"), row.names = FALSE); file.rename(paste0(.w_res, ".tmp"), .w_res)
  quit(save = "no", status = 1)
}
if (.w_phase == "prep") {
  dir.create(.w_scr, recursive = TRUE, showWarnings = FALSE)
  tryCatch({
    raw <- read.csv(file.path(DATA_DIR, "simulated_observed_week6.csv"))
    if (.w_rep > 0) {
      set.seed(2000000L + .w_rep)
      ids <- unique(raw$id)
      pick <- sample(ids, length(ids), replace = TRUE)
      rows_by_id <- split(seq_len(nrow(raw)), raw$id)
      sel <- rows_by_id[as.character(pick)]
      raw <- raw[unlist(sel, use.names = FALSE), ]
      raw$id <- rep(seq_along(pick), lengths(sel))
    }
    write.csv(raw, file.path(.w_scr, "simulated_observed_week6.csv"), row.names = FALSE)
    Sys.setenv(SIM_OUT_DIR = .w_scr)
    tf <- file.path(DATA_DIR, "true_censoring_survival_score_week6.csv")
    if (file.exists(tf)) Sys.setenv(TRUE_CSV = tf)
    source(file.path(BASE, "gpu_bart.R"))
    invisible(capture.output(suppressMessages(source(file.path(BASE, "estimator_gpu_prep.R"), echo = FALSE))))
  }, error = fail)
} else {
  tryCatch({
    tf <- file.path(DATA_DIR, "true_censoring_survival_score_week6.csv")   # environment variables are not in the saved session
    if (file.exists(tf)) Sys.setenv(TRUE_CSV = tf)
    load(file.path(.w_scr, "state.RData"), envir = globalenv())
    .w_phase <- "finish"   # the saved session also holds the prep run's .w_* values (same replicate, paths); only the phase differs
    invisible(capture.output(suppressMessages(source(file.path(BASE, "estimator_gpu_finish.R"), echo = FALSE))))
    out <- diff_tab; out$rep <- .w_rep
    write.csv(out, paste0(.w_res, ".tmp"), row.names = FALSE); file.rename(paste0(.w_res, ".tmp"), .w_res)
    unlink(.w_scr, recursive = TRUE)
  }, error = fail)
}

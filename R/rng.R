# Which random-number generator the samplers use, from the environment variable FASTBART_RNG:
#   "R" (default): R's generator (unif_rand, norm_rand, exp_rand, rchisq) -- follows set.seed() and gives the same
#                  results on every platform;
#   "internal":    the built-in xoshiro128** generator -- somewhat faster, seeded from R's generator, but its normal
#                  draws use the platform's float math library, so results differ between platforms (by Monte Carlo
#                  error only).
fastbart_rng <- function() {
  v <- Sys.getenv("FASTBART_RNG", "R")
  if (v == "") v <- "R"
  if (!v %in% c("R", "internal")) stop("FASTBART_RNG must be \"R\" (default) or \"internal\", not \"", v, "\"")
  v
}

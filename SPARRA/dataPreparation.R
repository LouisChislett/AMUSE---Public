library(fst)
library(lubridate)

## --- Read data ---
in_path <- "PATH"
df <- read_fst(in_path)

## --- Output folder (AMUSE next to the original file) ---
base_dir <- dirname(in_path)
out_dir  <- file.path(base_dir, "AMUSE")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

## --- Parse time robustly ---
if (inherits(df$time, "Date")) {
  df$time <- as.POSIXct(df$time, tz = "UTC")
} else if (inherits(df$time, c("POSIXct", "POSIXt"))) {
  df$time <- as.POSIXct(df$time, tz = "UTC")
} else if (is.numeric(df$time)) {
  df$time <- as.POSIXct(df$time, origin = "1970-01-01", tz = "UTC") # if epoch ms, use df$time/1000
} else {
  df$time <- suppressWarnings(ymd_hms(as.character(df$time), tz = "UTC"))
  if (all(is.na(df$time))) {
    df$time <- ymd(as.character(df$time), tz = "UTC")
    df$time <- as.POSIXct(df$time, tz = "UTC")
  }
}

## --- Keep only needed columns + month key ---
df$ym <- format(df$time, "%Y-%m")
keep <- c("target", "age", "decile", "sexM", "ym")
df2 <- df[, keep]

# SIMD decile coded as 11 should be treated as missing
df2$decile[df2$decile == 11] <- NA

# Remove rows with NA in any of the kept columns
df2 <- df2[complete.cases(df2), ]

## --- Write one CSV per month, suffix = YYYY-MM ---
months_ym <- sort(unique(df2$ym))

for (m in months_ym) {
  dat_m <- df2[df2$ym == m, c("target", "age", "decile", "sexM"), drop = FALSE]
  out_file <- file.path(out_dir, paste0("AMUSE_", m, ".csv"))
  write.csv(dat_m, out_file, row.names = FALSE)
}

cat("Wrote", length(months_ym), "files to:", out_dir, "\n")
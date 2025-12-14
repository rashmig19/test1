METRICS_XLSX_PATH = Path(getattr(settings, "METRICS_XLSX_PATH", str(METRICS_DIR / "api_metrics.xlsx")))
METRICS_MASTER_CSV = Path(getattr(settings, "METRICS_MASTER_CSV", str(METRICS_DIR / "api_metrics_master.csv")))
METRICS_QUEUE_CSV = Path(getattr(settings, "METRICS_QUEUE_CSV", str(METRICS_DIR / "api_metrics_queue.csv")))
